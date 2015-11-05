require 'torch'
require 'optim'
require 'image'
--require 'datasets'
require 'pl'
require 'paths'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
ADVERSARIAL = require 'adversarial'
DATASET = require 'dataset'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models'

----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  --batchSize        (default 128)         batch size (must be even, must be >= 4)
  --save             (default "logs")      subdirectory to save logs
  --saveFreq         (default 30)          save every saveFreq epochs
  --network          (default "")          reload pretrained network
  --noplot                                 Whether to not plot while training
  --N_epoch          (default 1000)        Number of examples per epoch (-1 means all)
  --G_SGD_lr         (default 0.02)        SGD learning rate for G
  --G_SGD_momentum   (default 0)           SGD momentum for G
  --D_SGD_lr         (default 0.02)        SGD learning rate for D
  --D_SGD_momentum   (default 0)           SGD momentum for D
  --G_adam_lr        (default -1)          Adam learning rate for G (-1 is automatic)
  --D_adam_lr        (default -1)          Adam learning rate for D (-1 is automatic)
  --G_L1             (default 0e-6)        L1 penalty on the weights of G
  --G_L2             (default 0e-6)        L2 penalty on the weights of G
  --D_L1             (default 0e-7)        L1 penalty on the weights of D
  --D_L2             (default 1e-4)        L2 penalty on the weights of D
  --D_iterations     (default 1)           How often to optimize D per batch
  --G_iterations     (default 1)           How often to optimize G per batch
  --D_maxAcc         (default 1.01)        stop training of D roughly around that accuracy level
  --D_clamp          (default 1)           To which value to clamp D's gradients (e.g. 5 means -5 to +5, 0 is off)
  --G_clamp          (default 5)           To which value to clamp G's gradients (e.g. 5 means -5 to +5, 0 is off)
  --D_optmethod      (default "adam")      Optimizer to use for D, either "sgd" or "adam" or "adagrad"
  --G_optmethod      (default "adam")      Optimizer to use for D, either "sgd" or "adam" or "adagrad"
  --threads          (default 8)           number of threads
  --gpu              (default 0)           gpu to run on (0-4 or -1 for cpu)
  --noiseDim         (default 100)         dimensionality of noise vector
  --window           (default 3)           ID of the first plotting window, will also use some window-ids beyond that
  --scale            (default 16)          scale of images to train on (height, width)
  --autoencoder      (default "")          path to autoencoder to load (optional)
  --seed             (default 1)           Seed to use for the RNG
  --weightsVisFreq   (default 0)           How often to update the windows showing the weights (only if >0; implies starting with qlua if >0)
  --grayscale                              Whether to activate grayscale mode on the images
  --denoise                                Whether to apply the denoiser trained with train_denoiser to the generated images
  --aws                                    Activate AWS settings
]]

if OPT.scale ~= 16 and OPT.scale ~= 32 then
    print("[Warning] models are not optimized for chosen scale")
end

-- check is batch size is valid (x >= 4 and an even number)
if OPT.batchSize % 2 ~= 0 or OPT.batchSize < 4 then
    print("[ERROR] batch size must be a multiple of 2 and higher or equal to 4")
    sys.exit()
end

-- GPU, seed, threads
if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)
torch.setnumthreads(OPT.threads)
print(OPT)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- run on gpu if chosen
if OPT.gpu then
    print("<trainer> starting gpu support...")
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
else
    require 'nn'
end
require 'dpnn'
require 'LeakyReLU'
torch.setdefaulttensortype('torch.FloatTensor')

CLASSES = {"0", "1"} -- possible output of disciminator, used for confusion matrix
Y_GENERATOR = 0 -- Y=Image was created by generator
Y_NOT_GENERATOR = 1 -- Y=Image was from training dataset
IMG_DIMENSIONS = {3, OPT.scale, OPT.scale} -- axis of images: 1 or 3 channels, <scale> px height, <scale> px width
if OPT.grayscale then IMG_DIMENSIONS[1] = 1 end
INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3] -- size in values/pixels per input image (channels*height*width)

-- Main function, creates/loads networks, loads dataset, starts training
function main()
    ----------------------------------------------------------------------
    -- Load / Define network
    ----------------------------------------------------------------------
    if OPT.denoise then
        -- load denoiser
        local filename = paths.concat(OPT.save, string.format('denoiser_%dx%dx%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
        local tmp = torch.load(filename)
        DENOISER = nn.Sequential()
        DENOISER:add(tmp.AE1_DECODER)
        --DENOISER:add(tmp.AE2_DECODER)
        DENOISER:float()
        DENOISER:evaluate()
    end
    
    -- load previous networks (D and G)
    -- or initialize them new
    if OPT.network ~= "" then
        print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
        require 'cutorch'
        require 'cunn'
        
        local tmp = torch.load(OPT.network)
        MODEL_D = tmp.D
        MODEL_G = tmp.G
        --OPTSTATE = tmp.optstate
        EPOCH = tmp.epoch
        
        if OPT.gpu == false then
            MODEL_D:float()
            MODEL_G:float()
        end
    else
        --------------
        -- D
        --------------
        MODEL_D = MODELS.create_D(IMG_DIMENSIONS)

        --------------
        -- G
        --------------
        if OPT.autoencoder ~= "" then
            -- Merge of autoencoder and G
            -- Autoencoder generates images, G tries to generate refined versions
            -- Concat layer then merges both
            local left = nn.Sequential()
            left:add(nn.View(INPUT_SZ))
            local right = nn.Sequential()
            right:add(nn.View(INPUT_SZ))
            right:add(nn.Linear(INPUT_SZ, 1024))
            right:add(nn.PReLU())
            right:add(nn.Linear(1024, INPUT_SZ))
            right:add(nn.Tanh())
            right:add(nn.MulConstant(0.25)) -- Limit the output to -0.25 to +0.25 (refine)

            local concat = nn.ConcatTable()
            concat:add(left)
            concat:add(right)
            MODEL_G = nn.Sequential()
            MODEL_G:add(concat)
            MODEL_G:add(nn.CAddTable()) -- add refined version to original
            MODEL_G:add(nn.View(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
        else
            -- No autoencoder chosen, just build a standard G
            MODEL_G = MODELS.create_G(IMG_DIMENSIONS, OPT.noiseDim)
        end
      
        NN_UTILS.initializeWeights(MODEL_D)
        NN_UTILS.initializeWeights(MODEL_G)
    end

    -- if we use an autoencoder then initialize it here
    if OPT.autoencoder == "" then
        print("[INFO] No Autoencoder network specified, will not use an autoencoder.")
    else
        print("<trainer> Loading autoencoder")
        local tmp = torch.load(OPT.autoencoder)
        local savedAutoencoder = tmp.AE

        MODEL_AE = nn.Sequential()
        MODEL_AE:add(nn.Linear(OPT.noiseDim, 256))
        MODEL_AE:add(nn.ReLU())
        MODEL_AE:add(nn.Linear(256, INPUT_SZ))
        MODEL_AE:add(nn.Sigmoid())
        MODEL_AE:add(nn.View(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))

        -- extract the decoder part from the autoencoder and set the weights of MODEL_AE
        -- to the ones of the decoder
        local mapping = {{1,6+1}, {3,6+3}, {5,6+5}}
        for i=1, #mapping do
            print(string.format("Loading AE layer %d from autoencoder layer %d ...", mapping[i][1], mapping[i][2]))
            local mapTo = mapping[i][1]
            local mapFrom = mapping[i][2]
            if MODEL_AE.modules[mapTo].weight and savedAutoencoder.modules[mapFrom].weight then
                MODEL_AE.modules[mapTo].weight = savedAutoencoder.modules[mapFrom].weight
            end
            if MODEL_AE.modules[mapTo].bias and savedAutoencoder.modules[mapFrom].bias then
                MODEL_AE.modules[mapTo].bias = savedAutoencoder.modules[mapFrom].bias
            end
        end
    end

    -- Activate GPU mode
    if OPT.gpu then
        if MODEL_AE then MODEL_AE = NN_UTILS.activateCuda(MODEL_AE) end
        MODEL_D = NN_UTILS.activateCuda(MODEL_D)
        MODEL_G = NN_UTILS.activateCuda(MODEL_G)
    end

    -- loss function: negative log-likelihood
    CRITERION = nn.BCECriterion()

    -- retrieve parameters and gradients
    PARAMETERS_D, GRAD_PARAMETERS_D = MODEL_D:getParameters()
    PARAMETERS_G, GRAD_PARAMETERS_G = MODEL_G:getParameters()

    -- print networks
    print("Autoencoder network:")
    print(MODEL_AE)
    print('Discriminator network:')
    print(MODEL_D)
    print('Generator network:')
    print(MODEL_G)

    ----------------------------------------------------------------------
    -- get/create dataset
    ----------------------------------------------------------------------
    -- adjust dataset
    if OPT.aws then
        DATASET.setDirs({"/mnt/datasets/out_aug_64x64"})
    else
        DATASET.setDirs({"dataset/out_aug_64x64"})
    end
    DATASET.setFileExtension("jpg")
    DATASET.setScale(OPT.scale)
    DATASET.setNbChannels(IMG_DIMENSIONS[1])
    ----------------------------------------------------------------------

    -- this matrix records the current confusion across classes
    CONFUSION = optim.ConfusionMatrix(CLASSES)

    -- Set optimizer state
    OPTSTATE = {
        adagrad = {D = {}, G = {}},
        adam = {D = {}, G = {}},
        rmsprop = {D = {}, G = {}},
        sgd = {
            D = {learningRate = OPT.D_SGD_lr, momentum = OPT.D_SGD_momentum},
            G = {learningRate = OPT.G_SGD_lr, momentum = OPT.G_SGD_momentum}
        }
    }
    
    if OPT.D_adam_lr ~= -1 then OPTSTATE.adam.D["learningRate"] = OPT.D_adam_lr end
    if OPT.G_adam_lr ~= -1 then OPTSTATE.adam.G["learningRate"] = OPT.G_adam_lr end

    -- Visualize initially, before the first epoch (usually just white noise, more if
    -- the networks were loaded from file)
    VIS_NOISE_INPUTS = NN_UTILS.createNoiseInputs(100)

    -- training loop
    EPOCH = 1
    while true do
        print('Loading new training data...')
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch)
        
        if not OPT.noplot then
            NN_UTILS.visualizeProgress(VIS_NOISE_INPUTS)
        end
    
        ADVERSARIAL.train(TRAIN_DATA, OPT.D_maxAcc, math.max(20, math.min(1000/OPT.batchSize, 250)))
    end
end

--------------------------------------
main()
