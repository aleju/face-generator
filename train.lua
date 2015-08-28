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


----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 10)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.02)        learning rate
  --forceLearningRate (default -1)
  -b,--batchSize     (default 100)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --forceMomentum    (default -1)         
  --GL1              (default 0)           L1 penalty on the weights of G
  --GL2              (default 0)           L2 penalty on the weights of G
  --DL1              (default 0)           L1 penalty on the weights of D
  --DL2              (default 0)           L2 penalty on the weights of D
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           window id of sample image
  --hidden_G         (default 8000)        number of units in hidden layers of G
  --hidden_D         (default 1600)        number of units in hidden layers of D
  --scale            (default 32)          scale of images to train on
  --autoencoder      (default "")          path to autoencoder to load weights from
]]

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(OPT.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- adjust dataset
DATASET.setDirs({"/media/aj/grab/ml/datasets/lfwcrop_grey/faces"})
DATASET.setFileExtension("pgm")
DATASET.setScale(OPT.scale)

-- run on gpu if chosen
if OPT.gpu then
    print("<trainer> starting gpu support...")
    require 'cunn'
    cutorch.setDevice(OPT.gpu + 1)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
    torch.setdefaulttensortype('torch.CudaTensor')
else
    torch.setdefaulttensortype('torch.FloatTensor')
end

-- possible output of disciminator
Y_GENERATOR = 0
Y_NOT_GENERATOR = 1

-- axis of images: 3 channels, <scale> height, <scale> width
OPT.geometry = {1, OPT.scale, OPT.scale}
-- size in values/pixels per input image (channels*height*width)
local INPUT_SZ = OPT.geometry[1] * OPT.geometry[2] * OPT.geometry[3]

function setWeights(weights, std)
    weights:randn(weights:size())
    weights:mul(std)
end

function initializeWeights(model, stdWeights, stdBias)
    for m = 1, #model.modules do
        if model.modules[m].weight then
            setWeights(model.modules[m].weight, stdWeights)
        end
        if model.modules[m].bias then
            setWeights(model.modules[m].bias, stdBias)
        end
    end
end

----------------------------------------------------------------------
-- Load / Define network
----------------------------------------------------------------------
-- load previous networks (D and G)
-- or initialize them new
if OPT.network ~= "" then
    print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
    tmp = torch.load(opt.network)
    MODEL_D = tmp.D
    MODEL_G = tmp.G
    OPTSTATE = tmp.optstate
else
  --------------
  -- D
  --------------
  local branch_conv = nn.Sequential()
  branch_conv:add(nn.SpatialConvolution(OPT.geometry[1], 32, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(nn.PReLU())
  branch_conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(nn.PReLU())
  branch_conv:add(nn.SpatialMaxPooling(2, 2))
  branch_conv:add(nn.View(32 * (1/4) * OPT.geometry[2] * OPT.geometry[3]))
  branch_conv:add(nn.Linear(32 * (1/4) * OPT.geometry[2] * OPT.geometry[3], 32))
  branch_conv:add(nn.PReLU())
  
  local branch_dense = nn.Sequential()
  branch_dense:add(nn.View(INPUT_SZ))
  branch_dense:add(nn.Linear(INPUT_SZ, 512))
  branch_dense:add(nn.PReLU())
  branch_dense:add(nn.Linear(512, 32))
  branch_dense:add(nn.PReLU())
  
  local concat = nn.ConcatTable()
  concat:add(branch_conv)
  concat:add(branch_dense)
  
  MODEL_D = nn.Sequential()
  MODEL_D:add(concat)
  MODEL_D:add(nn.JoinTable(2))
  MODEL_D:add(nn.Linear(32*2, 64))
  MODEL_D:add(nn.PReLU())
  MODEL_D:add(nn.Linear(64, 1))
  MODEL_D:add(nn.Sigmoid())
  
  --------------
  -- G
  --------------
  local left = nn.Sequential()
  left:add(nn.View(INPUT_SZ))
  local right = nn.Sequential()
  right:add(nn.View(INPUT_SZ))
  right:add(nn.Linear(INPUT_SZ, 128))
  right:add(nn.ReLU())
  right:add(nn.BatchNormalization(128))
  right:add(nn.Linear(128, INPUT_SZ))
  right:add(nn.Tanh())
  right:add(nn.MulConstant(0.25))
  
  local concat = nn.ConcatTable()
  concat:add(left)
  concat:add(right)
  MODEL_G = nn.Sequential()
  MODEL_G:add(concat)
  MODEL_G:add(nn.CAddTable())
  MODEL_G:add(nn.View(OPT.geometry[1], OPT.geometry[2], OPT.geometry[3]))
  
  initializeWeights(MODEL_D)
  initializeWeights(MODEL_G)
end

if OPT.autoencoder == "" then
    print("[ERROR] Autoencoder network required but not set in opt.autoencoder.")
    os.exit()
end
print("<trainer> Loading autoencoder")
local tmp = torch.load(OPT.autoencoder)
local savedAutoencoder = tmp.AE

MODEL_AE = nn.Sequential()
MODEL_AE:add(nn.Linear(OPT.noiseDim, 128))
MODEL_AE:add(nn.ReLU())
MODEL_AE:add(nn.Linear(128, 128))
MODEL_AE:add(nn.ReLU())
MODEL_AE:add(nn.Linear(128, INPUT_SZ))
MODEL_AE:add(nn.Sigmoid())
MODEL_AE:add(nn.View(OPT.geometry[1], OPT.geometry[2], OPT.geometry[3]))

local mapping = {{1,6+1}, {3,6+3}, {5,6+5}}
for i=1, #mapping do
    print("Loading AE layer " .. mapping[i][1] .. " from autoencoder layer " .. mapping[i][2] .. "...")
    local mapTo = mapping[i][1]
    local mapFrom = mapping[i][2]
    if model_AE.modules[mapTo].weight and savedAutoencoder.modules[mapFrom].weight then
        model_AE.modules[mapTo].weight = savedAutoencoder.modules[mapFrom].weight
    end
    if model_AE.modules[mapTo].bias and savedAutoencoder.modules[mapFrom].bias then
        model_AE.modules[mapTo].bias = savedAutoencoder.modules[mapFrom].bias
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



----------------------------------------------------------------------
-- get/create dataset
----------------------------------------------------------------------

-- create training set and normalize
print('Loading training dataset...')
TRAIN_DATA = dataset.loadImages(1, 10000)

-- create validation set and normalize
print('Loading validation dataset...')
VAL_DATA = ds_cats.loadValidationSet(10001, 512)

-- this matrix records the current confusion across classes
CONFUSION = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if OPT.gpu then
  print("Copying model to gpu...")
  MODEL_AE:cuda()
  MODEL_D:cuda()
  MODEL_G:cuda()
end

-- Set optimizer state if it hasn't been loaded from file
if OPTSTATE == nil then
    OPTSTATE = {
        adagrad = {D = {}, G = {}},
        adam = {D = {}, G = {}},
        rmsprop = {D = {}, G = {}},
        sgd = {
            D = {learningRate = OPT.learningRate, momentum = OPT.momentum},
            G = {learningRate = OPT.learningRate, momentum = OPT.momentum}
        }
    }
end

if OPT.forceLearningRate >= 0 then
    print("Forcing learning rate to " .. OPT.forceLearningRate)
    OPTSTATE.sgd.D.learningRate = OPT.forceLearningRate
    OPTSTATE.sgd.G.learningRate = OPT.forceLearningRate
end

if OPT.forceMomentum >= 0 then
    print("Forcing momentum to " .. OPT.forceMomentum)
    OPTSTATE.sgd.D.momentum = OPT.forceMomentum
    OPTSTATE.sgd.G.momentum = OPT.forceMomentum
end

function createNoiseInputs(N)
    local noiseInputs = torch.Tensor(N, OPT.noiseDim)
    noiseInputs:normal(0.0, 0.35)
    return noiseInputs
end

function createImagesFromNoise(noiseInputs, outputAsList)
    local images = MODEL_AE:forward(noiseInputs)
    images = MODEL_G:forward(samples)
    if outputAsList then
        local imagesList = {}
        for i=1,N do
            imagesList[#imagesList+1] = images[i]:float()
        end 
        return imagesList
    else
        return images
    end
end

function createImages(N, outputAsList)
    return createImagesFromNoise(createNoiseInputs(N, outputAsList))
end

function sortImagesByPrediction(images, ascending, nbMaxOut)
    local predictions = MODEL_D:forward(images)
    local imagesWithPreds = {}
    for i=1,#images do
        imagesWithPreds[i] = {images[i], predictions[i]}
    end
    
    if ascending then
        table.sort(imagesWithPreds, function (a,b) return a[2] < b[2] end)
    else
        table.sort(imagesWithPreds, function (a,b) return a[2] > b[2] end)
    end
    
    resultImages = {}
    resultPredictions = {}
    for i=1,math.min(nbMaxOut,#imagesWithPreds) do
        resultImages[i] = imagesWithPreds[i][1]
        resultPredictions[i] = imagesWithPreds[i][2]
    end
    
    return resultImages, resultPredictions
end

function visualizeProgress(noiseInputs)
    local semiRandomImages = createImagesFromNoise(noiseInputs, true)
    
    local randomImages = createImages(300, true)
    local badImages = sortImagesByPrediction(randomImages, true, 50)
    local goodImages = sortImagesByPrediction(randomImages, false, 50)
    
    if OPT.gpu then
        torch.setdefaulttensortype('torch.FloatTensor')
    end

    DISP.image(semiRandomImages, {win=opt.window, width=OPT.geometry[3]*10, title="semi-random generated images"})
    DISP.image(badImages, {win=opt.window+1, width=OPT.geometry[3]*10, title="best samples"})
    DISP.image(goodImages, {win=opt.window+2, width=OPT.geometry[3]*10, title="worst samples"})
    
    if OPT.gpu then
        torch.setdefaulttensortype('torch.CudaTensor')
    else
        torch.setdefaulttensortype('torch.FloatTensor')
    end
end


EPOCH = 1
VIS_NOISE_INPUTS = createNoiseInputs(100)
if opt.plot then
    visualizeProgress(VIS_NOISE_INPUTS)
end

-- training loop
while true do
    ADVERSARIAL.train(TRAIN_DATA, 1.01, math.max(10, math.min(1000/OPT.batchSize, 250)))

    if OPT.plot and EPOCH and EPOCH % 1 == 0 then
        visualizeProgress(VIS_NOISE_INPUTS)

        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        testLogger:style{['% mean class accuracy (test set)'] = '-'}
    end
end
