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
  --SGD_lr           (default 0.02)        SGD learning rate
  -b,--batchSize     (default 100)         batch size
  --SGD_momentum     (default 0)           SGD momentum
  --G_L1             (default 0)           L1 penalty on the weights of G
  --G_L2             (default 0)           L2 penalty on the weights of G
  --D_L1             (default 0)           L1 penalty on the weights of D
  --D_L2             (default 0)           L2 penalty on the weights of D
  --D_iterations     (default 1)           number of iterations to optimize D for
  --G_iterations     (default 1)           number of iterations to optimize G for
  -t,--threads       (default 8)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  -d,--noiseDim      (default 256)         dimensionality of noise vector
  -w, --window       (default 3)           window id of sample image
  --scale            (default 32)          scale of images to train on
  --autoencoder      (default "")          path to autoencoder to load weights from
  --rebuildOptstate  (default 0)           whether to force a rebuild of the optimizer state
]]

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(OPT.threads)
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

-- possible output of disciminator
CLASSES = {"0", "1"}
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
    stdWeights = stdWeights or 0.005
    stdBias = stdBias or 0.001
    
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
    tmp = torch.load(OPT.network)
    MODEL_D = tmp.D
    MODEL_G = tmp.G
    OPTSTATE = tmp.optstate
else
  --------------
  -- D
  --------------
  
  --[[
  MODEL_D = nn.Sequential()
  MODEL_D:add(nn.View(INPUT_SZ))
  MODEL_D:add(nn.Linear(INPUT_SZ, 1024))
  MODEL_D:add(nn.PReLU())
  MODEL_D:add(nn.Linear(1024, 1))
  MODEL_D:add(nn.Sigmoid())
  --]]
  
  -- scale 32 network
  local branch_conv = nn.Sequential()
  branch_conv:add(nn.SpatialConvolution(OPT.geometry[1], 32, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(nn.PReLU())
  branch_conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(nn.PReLU())
  branch_conv:add(nn.SpatialMaxPooling(2, 2))
  branch_conv:add(nn.View(32 * (1/4) * OPT.geometry[2] * OPT.geometry[3]))
  branch_conv:add(nn.Linear(32 * (1/4) * OPT.geometry[2] * OPT.geometry[3], 1024))
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
  
  -- scale 64 network
  --[[
  local branch_conv = nn.Sequential()
  branch_conv:add(nn.SpatialConvolution(OPT.geometry[1], 4, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(nn.PReLU())
  branch_conv:add(nn.SpatialMaxPooling(2, 2))
  branch_conv:add(nn.SpatialConvolution(4, 8, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(nn.PReLU())
  branch_conv:add(nn.SpatialMaxPooling(2, 2))
  branch_conv:add(nn.Dropout())
  branch_conv:add(nn.View((1/4) * 8 * (1/4) * OPT.geometry[2] * OPT.geometry[3]))
  branch_conv:add(nn.Linear((1/4) * 8 * (1/4) * OPT.geometry[2] * OPT.geometry[3], 128))
  branch_conv:add(nn.PReLU())
  
  local branch_dense = nn.Sequential()
  branch_dense:add(nn.View(INPUT_SZ))
  branch_dense:add(nn.Linear(INPUT_SZ, 512))
  branch_dense:add(nn.PReLU())
  branch_dense:add(nn.Dropout())
  branch_dense:add(nn.Linear(512, 128))
  branch_dense:add(nn.PReLU())
  
  local concat = nn.ConcatTable()
  concat:add(branch_conv)
  concat:add(branch_dense)
  
  MODEL_D = nn.Sequential()
  MODEL_D:add(concat)
  MODEL_D:add(nn.JoinTable(2))
  MODEL_D:add(nn.Linear(128*2, 128))
  MODEL_D:add(nn.PReLU())
  MODEL_D:add(nn.Dropout())
  MODEL_D:add(nn.Linear(128, 1))
  MODEL_D:add(nn.Sigmoid())
  --]]
  
  -- scale 64 sgd network
  --[[
  local branch_conv = nn.Sequential()
  branch_conv:add(nn.SpatialConvolution(OPT.geometry[1], 8, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(nn.PReLU())
  --branch_conv:add(nn.SpatialMaxPooling(2, 2))
  branch_conv:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(nn.PReLU())
  branch_conv:add(nn.SpatialMaxPooling(2, 2))
  branch_conv:add(nn.Dropout())
  branch_conv:add(nn.View(8 * (1/4) * OPT.geometry[2] * OPT.geometry[3]))
  branch_conv:add(nn.Linear(8 * (1/4) * OPT.geometry[2] * OPT.geometry[3], 128))
  branch_conv:add(nn.PReLU())
  
  local branch_dense = nn.Sequential()
  branch_dense:add(nn.View(INPUT_SZ))
  branch_dense:add(nn.Linear(INPUT_SZ, 512))
  branch_dense:add(nn.PReLU())
  branch_dense:add(nn.Dropout())
  branch_dense:add(nn.Linear(512, 128))
  branch_dense:add(nn.PReLU())
  
  local concat = nn.ConcatTable()
  concat:add(branch_conv)
  concat:add(branch_dense)
  
  MODEL_D = nn.Sequential()
  MODEL_D:add(concat)
  MODEL_D:add(nn.JoinTable(2))
  MODEL_D:add(nn.Linear(128*2, 128))
  MODEL_D:add(nn.PReLU())
  --MODEL_D:add(nn.Dropout())
  --MODEL_D:add(nn.BatchNormalization(128))
  MODEL_D:add(nn.Linear(128, 1))
  MODEL_D:add(nn.Sigmoid())
  --]]
  
  --------------
  -- G
  --------------
  if OPT.autoencoder ~= "" then
      local left = nn.Sequential()
      left:add(nn.View(INPUT_SZ))
      local right = nn.Sequential()
      right:add(nn.View(INPUT_SZ))
      right:add(nn.Linear(INPUT_SZ, 1024))
      right:add(nn.PReLU())
      right:add(nn.BatchNormalization(1024))
      right:add(nn.Linear(1024, 1024))
      right:add(nn.PReLU())
      right:add(nn.BatchNormalization(1024))
      right:add(nn.Linear(1024, INPUT_SZ))
      right:add(nn.Tanh())
      right:add(nn.MulConstant(0.25))
      
      local concat = nn.ConcatTable()
      concat:add(left)
      concat:add(right)
      MODEL_G = nn.Sequential()
      MODEL_G:add(concat)
      MODEL_G:add(nn.CAddTable())
      MODEL_G:add(nn.View(OPT.geometry[1], OPT.geometry[2], OPT.geometry[3]))
  else
      --[[
      MODEL_G = nn.Sequential()
      MODEL_G:add(nn.Linear(OPT.noiseDim, 1024))
      MODEL_G:add(nn.PReLU())
      MODEL_G:add(nn.BatchNormalization(1024))
      MODEL_G:add(nn.Linear(1024, 1024))
      MODEL_G:add(nn.PReLU())
      MODEL_G:add(nn.BatchNormalization(1024))
      MODEL_G:add(nn.Linear(1024, INPUT_SZ))
      MODEL_G:add(nn.Sigmoid())
      MODEL_G:add(nn.View(OPT.geometry[1], OPT.geometry[2], OPT.geometry[3]))
      --]]
      MODEL_G = nn.Sequential()
      MODEL_G:add(nn.Linear(OPT.noiseDim, 6500))
      MODEL_G:add(nn.PReLU())
      --MODEL_G:add(nn.Dropout())
      --MODEL_G:add(nn.Linear(256, 4096))
      --MODEL_G:add(nn.PReLU())
      MODEL_G:add(nn.Linear(6500, INPUT_SZ))
      MODEL_G:add(nn.Sigmoid())
      MODEL_G:add(nn.View(OPT.geometry[1], OPT.geometry[2], OPT.geometry[3]))
  end
  
  initializeWeights(MODEL_D)
  initializeWeights(MODEL_G)
end

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
    MODEL_AE:add(nn.View(OPT.geometry[1], OPT.geometry[2], OPT.geometry[3]))

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

-- create training set and normalize
print('Loading training dataset...')
TRAIN_DATA = DATASET.loadImages(1, 8192+2048)

-- create validation set and normalize
--print('Loading validation dataset...')
--VAL_DATA = DATASET.loadImages(8192+2048, 512)
----------------------------------------------------------------------

-- this matrix records the current confusion across classes
CONFUSION = optim.ConfusionMatrix(CLASSES)

-- log results to files
trainLogger = optim.Logger(paths.concat(OPT.save, 'train.log'))
testLogger = optim.Logger(paths.concat(OPT.save, 'test.log'))

-- Set optimizer state if it hasn't been loaded from file
if OPTSTATE == nil or OPT.rebuildOptstate == 1 then
    OPTSTATE = {
        adagrad = {D = {}, G = {}},
        adam = {
            --D = {learningRate = 0.0005},
            --G = {learningRate = 0.0010}
            D = {},
            G = {}
        },
        rmsprop = {D = {}, G = {}},
        sgd = {
            D = {learningRate = OPT.SGD_lr, momentum = OPT.SGD_momentum},
            G = {learningRate = OPT.SGD_lr, momentum = OPT.SGD_momentum}
        }
    }
end

function createNoiseInputs(N)
    local noiseInputs = torch.Tensor(N, OPT.noiseDim)
    --noiseInputs:normal(0.0, 0.35)
    noiseInputs:uniform(-1.0, 1.0)
    return noiseInputs
end

function createImagesFromNoise(noiseInputs, outputAsList, refineWithG)
    local images
    if MODEL_AE then
        images = MODEL_AE:forward(noiseInputs)
        if refineWithG == nil or refineWithG ~= false then
            images = MODEL_G:forward(images)
        end
    else
        images = MODEL_G:forward(noiseInputs)
    end
    
    if outputAsList then
        local imagesList = {}
        for i=1, images:size(1) do
            imagesList[#imagesList+1] = images[i]:float()
        end
        return imagesList
    else
        return images
    end
end

function createImages(N, outputAsList, refineWithG)
    return createImagesFromNoise(createNoiseInputs(N), outputAsList, refineWithG)
end

function sortImagesByPrediction(images, ascending, nbMaxOut)
    local predictions = MODEL_D:forward(images)
    local imagesWithPreds = {}
    for i=1,images:size(1) do
        imagesWithPreds[i] = {images[i], predictions[i][1]}
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
    switchToEvaluationMode()
    
    local semiRandomImagesUnrefined
    if MODEL_AE then
        semiRandomImagesUnrefined = createImagesFromNoise(noiseInputs, true, false)
    end
    local semiRandomImagesRefined = createImagesFromNoise(noiseInputs, true, true)
    
    local sanityTestImage = torch.Tensor(1, OPT.scale, OPT.scale)
    sanityTestImage:uniform(0.0, 0.50)
    for i=1,OPT.scale do
        for j=1,OPT.scale do
            if i == j then
                sanityTestImage[1][i][j] = 1.0
            elseif i % 4 == 0 and j % 4 == 0 then
                sanityTestImage[1][i][j] = 0.5
            end
        end
    end
    
    --[[
    for i=1,OPT.scale do
        for j=1,OPT.scale do
            if i == j then
                randomImages[300][1][i][j] = 255
            elseif i % 4 == 0 and j % 4 == 0 then
                randomImages[300][1][i][j] = 125
            end
        end
    end
    --]]
    
    
    local randomImages = createImages(300, false)
    --randomImages[298] = TRAIN_DATA[2] -- one real face as sanity test
    randomImages[299] = TRAIN_DATA[3] -- one real face as sanity test
    randomImages[300] = sanityTestImage -- synthetic non-face as sanity test
    local badImages, _ = sortImagesByPrediction(randomImages, true, 50)
    local goodImages, _ = sortImagesByPrediction(randomImages, false, 50)
    
    if OPT.gpu then
        torch.setdefaulttensortype('torch.FloatTensor')
    end

    if semiRandomImagesUnrefined then
        DISP.image(semiRandomImagesUnrefined, {win=OPT.window, width=OPT.geometry[3]*15, title="semi-random generated images (before G)"})
    end
    DISP.image(semiRandomImagesRefined, {win=OPT.window+1, width=OPT.geometry[3]*15, title="semi-random generated images (after G)"})
    DISP.image(goodImages, {win=OPT.window+2, width=OPT.geometry[3]*15, title="best samples (first is best)"})
    DISP.image(badImages, {win=OPT.window+3, width=OPT.geometry[3]*15, title="worst samples (first is worst)"})
    
    if OPT.gpu then
        torch.setdefaulttensortype('torch.CudaTensor')
    else
        torch.setdefaulttensortype('torch.FloatTensor')
    end
    
    switchToTrainingMode()
end

function switchToTrainingMode()
    if MODEL_AE then
        MODEL_AE:training()
    end
    MODEL_G:training()
    MODEL_D:training()
end

function switchToEvaluationMode()
    if MODEL_AE then
        MODEL_AE:evaluate()
    end
    MODEL_G:evaluate()
    MODEL_D:evaluate()
end

EPOCH = 1
VIS_NOISE_INPUTS = createNoiseInputs(100)
if OPT.plot then
    visualizeProgress(VIS_NOISE_INPUTS)
end

-- training loop
while true do
    ADVERSARIAL.train(TRAIN_DATA, 1.01, math.max(20, math.min(1000/OPT.batchSize, 250)))
    --OPTSTATE.adam.G.learningRate = OPTSTATE.adam.G.learningRate * 0.99

    if OPT.plot and EPOCH and EPOCH % 1 == 0 then
        visualizeProgress(VIS_NOISE_INPUTS)

        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        testLogger:style{['% mean class accuracy (test set)'] = '-'}
    end
end
