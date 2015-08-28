require 'torch'
require 'nn'
require 'optim'
require 'image'
--require 'datasets'
require 'pl'
require 'paths'
image_utils = require 'utils.image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
DATASET = require 'dataset'

----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 50)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.02)        learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 8)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  -d,--noiseDim      (default 256)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           window id of sample image
  --hidden_G         (default 8000)        number of units in hidden layers of G
  --hidden_D         (default 1600)        number of units in hidden layers of D
  --scale            (default 32)          scale of images to train on
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
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
    torch.setdefaulttensortype('torch.CudaTensor')
else
    torch.setdefaulttensortype('torch.FloatTensor')
end

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

MODEL_AE = nn.Sequential()

MODEL_AE:add(nn.View(INPUT_SZ))
MODEL_AE:add(nn.Linear(INPUT_SZ, 512))
MODEL_AE:add(nn.ReLU())
MODEL_AE:add(nn.Linear(512, OPT.noiseDim))
MODEL_AE:add(nn.Tanh())
MODEL_AE:add(nn.Dropout(0.50))
MODEL_AE:add(nn.Linear(OPT.noiseDim, 256))
MODEL_AE:add(nn.ReLU())
MODEL_AE:add(nn.Linear(256, INPUT_SZ))
MODEL_AE:add(nn.Sigmoid())
MODEL_AE:add(nn.View(OPT.geometry[1], OPT.geometry[2], OPT.geometry[3]))

initializeWeights(MODEL_AE)

-- loss function
--criterion = nn.MSECriterion()
CRITERION = nn.AbsCriterion()

-- retrieve parameters and gradients
PARAMETERS_AE, GRAD_PARAMETERS_AE = MODEL_AE:getParameters()

if OPT.gpu then
    print('<trainer> Copy model to gpu')
    MODEL_AE:cuda()
end

-- print networks
print("<trainer> Autoencoder network:")
print(MODEL_AE)

----------------------------------------------------------------------
-- get/create dataset
----------------------------------------------------------------------
-- adjust dataset
DATASET.setDirs({"/media/aj/grab/ml/datasets/lfwcrop_grey/faces"})
DATASET.setFileExtension("pgm")
DATASET.setScale(OPT.scale)

-- create training set and normalize
print('<trainer> Loading training dataset...')
TRAIN_DATA = DATASET.loadImages(1, 10000)
print('<trainer> Loading validation dataset...')
VAL_DATA = DATASET.loadImages(10001, 512)
----------------------------------------------------------------------


-- Set optimizer state if it hasn't been loaded from file
OPTSTATE = {
    adagrad = {},
    adam = {},
    rmsprop = {},
    sgd = {learningRate = OPT.learningRate, momentum = OPT.momentum}
}

-- Get examples to plot
function getSamples(dataset, N)
    local images = torch.Tensor(N, OPT.geometry[1], OPT.geometry[2], OPT.geometry[3])
    for i=1, N do
        images[i] = dataset[i]
    end

    local decoded = MODEL_AE:forward(images)
    return {images:float(), decoded:float()}
end

-- train one epoch
function train(usedDataset)
    EPOCH = EPOCH or 1
    local N = usedDataset:size()
    local time = sys.clock()

    local shuffle
    if OPT.gpu then
        torch.setdefaulttensortype('torch.FloatTensor')
        shuffle = torch.randperm(N)
        torch.setdefaulttensortype('torch.CudaTensor')
    else
        shuffle = torch.randperm(N)
    end

    -- do one epoch
    print('\n<trainer> on training set:')
    print("<trainer> online epoch # " .. EPOCH .. ' [batchSize = ' .. OPT.batchSize .. ']')
    for t = 1,N,OPT.batchSize do
        -- if the last batch has a size smaller than opt.batchSize, adjust for that
        local thisBatchSize = math.min(OPT.batchSize, N - t + 1)
        local inputs = torch.Tensor(thisBatchSize, OPT.geometry[1], OPT.geometry[2], OPT.geometry[3])
        local targets = torch.Tensor(thisBatchSize, OPT.geometry[1], OPT.geometry[2], OPT.geometry[3])
        
        for i=1,thisBatchSize do
            inputs[{i, {}, {}, {}}] = usedDataset[shuffle[t+i-1]]
            targets[{i, {}, {}, {}}] = usedDataset[shuffle[t+i-1]]
        end

        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of discriminator
        local fevalAE = function(x)
          collectgarbage()

          if x ~= PARAMETERS_AE then -- get new parameters
            PARAMETERS_AE:copy(x)
          end

          GRAD_PARAMETERS_AE:zero() -- reset gradients

          --  forward pass
          local outputs = MODEL_AE:forward(inputs)
          local f = CRITERION:forward(outputs, targets)

          -- backward pass 
          local df_do = CRITERION:backward(outputs, targets)
          MODEL_AE:backward(inputs, df_do)

          -- penalties (L1 and L2):
          if OPT.coefL1 ~= 0 or OPT.coefL2 ~= 0 then
            local norm = torch.norm
            local sign = torch.sign
            -- Loss:
            f = f + OPT.coefL1 * torch.norm(PARAMETERS_AE, 1)
            f = f + OPT.coefL2 * torch.norm(PARAMETERS_AE, 2)^2/2
            -- Gradients:
            GRAD_PARAMETERS_AE:add( torch.sign(PARAMETERS_AE):mul(OPT.coefL1) + PARAMETERS_AE:clone():mul(OPT.coefL2) )
          end

          return f, GRAD_PARAMETERS_AE
        end

        optim.adam(fevalAE, PARAMETERS_AE, OPTSTATE.adam)

        -- display progress
        xlua.progress(t, usedDataset:size())
    end -- end for loop over dataset

    -- fill out progress bar completely,
    -- for some reason that doesn't happen in the previous loop
    -- probably because it progresses to t instead of t+dataBatchSize
    xlua.progress(usedDataset:size(), usedDataset:size())

    -- time taken
    time = sys.clock() - time
    print("<trainer> time required for this epoch = " .. (time) .. "s")
    time = time / usedDataset:size()
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- save/log current net
    if EPOCH % OPT.saveFreq == 0 then
        local filename = paths.concat(OPT.save, 'autoencoder.net')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        if paths.filep(filename) then
            os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
        end
        print('<trainer> saving network to '..filename)
        torch.save(filename, {AE = MODEL_AE, optstate = OPTSTATE})
    end

    -- next epoch
    EPOCH = EPOCH + 1
end

function exitIfNaNs(checkIn)
    local nanCount = checkIn:ne(checkIn):sum()
    if nanCount > 0 then
        print("[ERROR] Detected " .. nanCount .. " NaNs. Exiting.")
        os.exit()
    end
end

-- training loop
while true do
    train(TRAIN_DATA)

    if OPT.plot and EPOCH and EPOCH % 1 == 0 then
        local samplesTrain = getSamples(TRAIN_DATA, 200)
        local samplesVal = getSamples(VAL_DATA, 200)
        torch.setdefaulttensortype('torch.FloatTensor')

        exitIfNaNs(samplesTrain[2])

        disp.image(samplesTrain[1], {win=OPT.window, width=OPT.geometry[3]*10, title=OPT.save .. " (originals train)"})
        disp.image(samplesTrain[2], {win=OPT.window+1, width=OPT.geometry[3]*10, title=OPT.save .. " (decoded train)"})

        disp.image(samplesVal[1], {win=OPT.window+2, width=OPT.geometry[3]*10, title=OPT.save .. " (originals val)"})
        disp.image(samplesVal[2], {win=OPT.window+3, width=OPT.geometry[3]*10, title=OPT.save .. " (decoded val)"})

        if OPT.gpu then
            torch.setdefaulttensortype('torch.CudaTensor')
        else
            torch.setdefaulttensortype('torch.FloatTensor')
        end
    end
end


