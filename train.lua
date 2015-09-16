require 'torch'
require 'nn'
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

----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  -b,--batchSize     (default 100)         batch size
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 10)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  --N_epoch          (default -1)          Number of examples per epoch (-1 means all)
  --G_SGD_lr         (default 0.02)        SGD learning rate for G
  --G_SGD_momentum   (default 0)           SGD momentum for G
  --D_SGD_lr         (default 0.02)        SGD learning rate for D
  --D_SGD_momentum   (default 0)           SGD momentum for D
  --G_adam_lr        (default -1)          Adam learning rate for G (-1 is automatic)
  --D_adam_lr        (default -1)          Adam learning rate for D (-1 is automatic)
  --G_L1             (default 0)           L1 penalty on the weights of G
  --G_L2             (default 0.000001)    L2 penalty on the weights of G
  --D_L1             (default 0)           L1 penalty on the weights of D
  --D_L2             (default 0.0001)      L2 penalty on the weights of D
  --D_iterations     (default 1)           number of iterations to optimize D for
  --G_iterations     (default 1)           number of iterations to optimize G for
  --D_maxAcc         (default 0.99)        stop training of D roughly around that accuracy level
  --D_clamp          (default 1)           To which value to clamp D's gradients (e.g. 5 means -5 to +5, 0 is off)
  --G_clamp          (default 5)           To which value to clamp G's gradients (e.g. 5 means -5 to +5, 0 is off)
  -t,--threads       (default 8)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  -d,--noiseDim      (default 256)         dimensionality of noise vector
  -w, --window       (default 3)           ID of the first plotting window, will also use some window-ids beyond that
  --scale            (default 32)          scale of images to train on
  --autoencoder      (default "")          path to autoencoder to load (optional)
  --rebuildOptstate                        force rebuild of the optimizer state even when loading from save
]]

-- GPU, seed, threads
if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
torch.manualSeed(1)
torch.setnumthreads(OPT.threads)
print(OPT)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- run on gpu if chosen
if OPT.gpu then
    print("<trainer> starting gpu support...")
    require 'cunn'
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(1)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
    torch.setdefaulttensortype('torch.CudaTensor')
else
    torch.setdefaulttensortype('torch.FloatTensor')
end



CLASSES = {"0", "1"} -- possible output of disciminator, used for confusion matrix
Y_GENERATOR = 0 -- Y=Image was created by generator
Y_NOT_GENERATOR = 1 -- Y=Image was from training dataset
IMG_DIMENSIONS = {1, OPT.scale, OPT.scale} -- axis of images: 1 or 3 channels, <scale> px height, <scale> px width
INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3] -- size in values/pixels per input image (channels*height*width)

function main()
    ----------------------------------------------------------------------
    -- Load / Define network
    ----------------------------------------------------------------------
    -- load previous networks (D and G)
    -- or initialize them new
    if OPT.network ~= "" then
        print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
        tmp = torch.load(OPT.network)
        MODEL_D = tmp.D
        MODEL_G = tmp.G
        OPTSTATE = tmp.optstate
    else
        --------------
        -- D
        --------------
        -- One branch with convolutions, one with dense layers
        -- merge them at the end
        local branch_conv = nn.Sequential()
        branch_conv:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 32, 3, 3, 1, 1, (3-1)/2))
        branch_conv:add(nn.PReLU())
        branch_conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
        branch_conv:add(nn.PReLU())
        branch_conv:add(nn.SpatialMaxPooling(2, 2))
        branch_conv:add(nn.View(32 * (1/4) * INPUT_SZ))
        branch_conv:add(nn.Linear(32 * (1/4) * INPUT_SZ, 1024))
        branch_conv:add(nn.PReLU())

        local branch_dense = nn.Sequential()
        branch_dense:add(nn.View(INPUT_SZ))
        branch_dense:add(nn.Linear(INPUT_SZ, 1024))
        branch_dense:add(nn.PReLU())
        branch_dense:add(nn.Linear(1024, 1024))
        branch_dense:add(nn.PReLU())

        local concat = nn.ConcatTable()
        concat:add(branch_conv)
        concat:add(branch_dense)

        MODEL_D = nn.Sequential()
        MODEL_D:add(concat)
        MODEL_D:add(nn.JoinTable(2))
        MODEL_D:add(nn.Linear(1024*2, 512))
        MODEL_D:add(nn.PReLU())
        MODEL_D:add(nn.Dropout())
        MODEL_D:add(nn.Linear(512, 1))
        MODEL_D:add(nn.Sigmoid())

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
            right:add(nn.Linear(INPUT_SZ, 4096))
            right:add(nn.PReLU())
            right:add(nn.Linear(4096, INPUT_SZ))
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
            MODEL_G = nn.Sequential()
            MODEL_G:add(nn.Linear(OPT.noiseDim, 4096))
            MODEL_G:add(nn.PReLU())
            MODEL_G:add(nn.Linear(4096, INPUT_SZ))
            MODEL_G:add(nn.Sigmoid())
            MODEL_G:add(nn.View(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
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

    if OPT.gpu then
        print("Copying model to gpu...")
        if MODEL_AE then
            MODEL_AE:cuda()
        end
        MODEL_D:cuda()
        MODEL_G:cuda()
    end


    ----------------------------------------------------------------------
    -- get/create dataset
    ----------------------------------------------------------------------
    -- adjust dataset
    DATASET.setDirs({"/media/aj/grab/ml/datasets/lfwcrop_grey/faces"})
    DATASET.setFileExtension("pgm")
    DATASET.setScale(OPT.scale)

    -- create training set
    print('Loading training dataset...')
    TRAIN_DATA = DATASET.loadImages(1, 12000)
    ----------------------------------------------------------------------

    -- this matrix records the current confusion across classes
    CONFUSION = optim.ConfusionMatrix(CLASSES)

    -- log results to files
    TRAIN_LOGGER = optim.Logger(paths.concat(OPT.save, 'train.log'))
    TEST_LOGGER = optim.Logger(paths.concat(OPT.save, 'test.log'))

    -- Set optimizer state if it hasn't been loaded from file
    if OPTSTATE == nil or OPT.rebuildOptstate then
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
    end

    -- Visualize initially, before the first epoch (usually just white noise, more if
    -- the networks were loaded from file)
    VIS_NOISE_INPUTS = NN_UTILS.createNoiseInputs(100)
    if OPT.plot then
        NN_UTILS.visualizeProgress(VIS_NOISE_INPUTS)
    end

    -- training loop
    EPOCH = 1
    while true do
        ADVERSARIAL.train(TRAIN_DATA, OPT.D_maxAcc, math.max(20, math.min(1000/OPT.batchSize, 250)))
        --OPTSTATE.adam.G.learningRate = OPTSTATE.adam.G.learningRate * 0.99

        if OPT.plot and EPOCH and EPOCH % 1 == 0 then
            NN_UTILS.visualizeProgress(VIS_NOISE_INPUTS)

            TRAIN_LOGGER:style{['% mean class accuracy (train set)'] = '-'}
            TEST_LOGGER:style{['% mean class accuracy (test set)'] = '-'}
        end
    end
end

--------------------------------------
main()
