require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'pl'
require 'paths'

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
DATASET = require 'dataset'
NN_UTILS = require 'utils/nn_utils'

----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  --save             (default "logs")      subdirectory to save logs
  --saveFreq         (default 50)          save every saveFreq epochs
  --network          (default "")          reload pretrained network
  --plot                                   plot while training
  --batchSize        (default 128)         batch size
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  --AE_clamp         (default 1)
  --threads          (default 8)           number of threads
  --gpu              (default 0)           gpu to run on (default cpu)
  --window           (default 10)          first window id (will block range ID to ID+3)
  --scale            (default 32)          scale of images to train on
  --seed             (default 1)           Seed to use for the RNG
  --grayscale                              Whether to activate grayscale mode on the images
]]

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

IMG_DIMENSIONS = {3, OPT.scale, OPT.scale} -- axis of images: 1 or 3 channels, <scale> px height, <scale> px width
if OPT.grayscale then IMG_DIMENSIONS[1] = 1 end
INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3] -- size in values/pixels per input image (channels*height*width)

function main()
    -- 8/8 convs into 2048/2048 mit lrelu und bnorm
    -- nach E350
    -- AE1: 0.0024
    -- AE2: 0.0023

    if OPT.network ~= "" then
        print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
        local filename = paths.concat(OPT.save, 'denoiser.net')
        local tmp = torch.load(filename)
        AE = nn.Sequential()
        AE:add(tmp.AE1_ENCODER)
        AE:add(tmp.AE1_DECODER)
        AE2 = nn.Sequential()
        AE2:add(tmp.AE2_DECODER)
        EPOCH = tmp.epoch
    else
        local ENCODER = nn.Sequential()
        ENCODER:add(nn.WhiteNoise(0.0, 0.1))
        
        --[[
        local DECODER = nn.Sequential()
        DECODER:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 32, 3, 3, 1, 1, 0))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 0))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 0))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.View(64 * (IMG_DIMENSIONS[2]-8) * (IMG_DIMENSIONS[3]-8)))
        DECODER:add(nn.Linear(64 * (IMG_DIMENSIONS[2]-8) * (IMG_DIMENSIONS[3]-8), 1024))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.Linear(1024, 1024))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.Linear(1024, IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]))
        DECODER:add(nn.Sigmoid())
        DECODER:add(nn.View(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
        --]]
        
        local DECODER = nn.Sequential()
        
        DECODER:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 8, 3, 3, 1, 1, 0))
        DECODER:add(nn.SpatialBatchNormalization(8))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 0))
        DECODER:add(nn.SpatialBatchNormalization(8))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.Dropout(0.2))
        
        DECODER:add(nn.View(8 * 28 * 28))
        
        DECODER:add(nn.Linear(8 * 28 * 28, 2048))
        DECODER:add(nn.BatchNormalization(2048))
        DECODER:add(nn.LeakyReLU(0.333))
        DECODER:add(nn.Dropout(0.2))
        
        DECODER:add(nn.Linear(2048, IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]))
        DECODER:add(nn.Sigmoid())
        DECODER:add(nn.View(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))

        NN_UTILS.initializeWeights(ENCODER)
        NN_UTILS.initializeWeights(DECODER)

        AE = nn.Sequential()
        AE:add(ENCODER)
        AE:add(DECODER)
        
        AE2 = AE:get(2):clone()
    end

    if OPT.gpu then
        AE = NN_UTILS.activateCuda(AE)
        AE2 = NN_UTILS.activateCuda(AE2)
    end

    -- loss function
    CRITERION = nn.BCECriterion()
    CRITERION2 = nn.BCECriterion()

    -- retrieve parameters and gradients
    PARAMETERS_AE, GRAD_PARAMETERS_AE = AE:getParameters()
    PARAMETERS_AE2, GRAD_PARAMETERS_AE2 = AE2:getParameters()

    -- print networks
    print("<trainer> Autoencoder network:")
    print(AE)

    ----------------------------------------------------------------------
    -- get/create dataset
    ----------------------------------------------------------------------
    -- adjust dataset
    DATASET.setDirs({"/media/aj/grab/ml/datasets/lfwcrop_color/faces"})
    DATASET.setFileExtension("ppm") -- pgm for lfwcrop_grey
    DATASET.setScale(OPT.scale)
    DATASET.setNbChannels(IMG_DIMENSIONS[1])

    -- create training set
    print('<trainer> Loading training dataset...')
    TRAIN_DATA = DATASET.loadImages(1, 10000)
    print('<trainer> Loading validation dataset...')
    VAL_DATA = DATASET.loadImages(10001, 256)
    VAL_DATA_TENSOR = imageListToTensor(VAL_DATA)
    ----------------------------------------------------------------------


    -- Set optimizer state
    OPTSTATE = {
        adagrad = {},
        adam = {}
    }
    
    -- training loop
    EPOCH = 1
    PLOT_DATA = {}
    MIN_LOSS = 9999999
    MIN_LOSS2 = 9999999
    while true do
        loss1, loss2 = train(TRAIN_DATA)

        if OPT.plot and EPOCH and EPOCH % 1 == 0 then
            AE:evaluate()
            AE2:evaluate()
            
            local valResult1 = AE:forward(VAL_DATA_TENSOR)
            local valResult2 = AE2:forward(valResult1)
            
            table.insert(PLOT_DATA, {EPOCH, loss1, loss2, CRITERION:forward(valResult1, VAL_DATA_TENSOR), CRITERION2:forward(valResult2, VAL_DATA_TENSOR)})
            if loss1 < MIN_LOSS then MIN_LOSS = loss1 end
            if loss2 < MIN_LOSS2 then MIN_LOSS2 = loss2 end
            
            local samplesTrain = getSamples(TRAIN_DATA, 100)
            local samplesVal = getSamples(VAL_DATA, 100)

            disp.image(samplesTrain[1], {win=OPT.window, width=IMG_DIMENSIONS[3]*15, title=OPT.save .. " (originals train)"})
            disp.image(samplesTrain[2], {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title=OPT.save .. " (decoded train)"})
            disp.image(samplesTrain[3], {win=OPT.window+2, width=IMG_DIMENSIONS[3]*15, title=OPT.save .. " (decoded train 2)"})

            disp.image(samplesVal[1], {win=OPT.window+3, width=IMG_DIMENSIONS[3]*15, title=OPT.save .. " (originals val)"})
            disp.image(samplesVal[2], {win=OPT.window+4, width=IMG_DIMENSIONS[3]*15, title=OPT.save .. " (decoded val)"})
            disp.image(samplesVal[3], {win=OPT.window+5, width=IMG_DIMENSIONS[3]*15, title=OPT.save .. " (decoded val 2)"})
            
            disp.plot(PLOT_DATA, {win=OPT.window+6, labels={'epoch', 'AE train loss', 'AE2 train loss', 'AE val loss', 'AE2 val loss'}, title=string.format('Loss at epoch %d (min1=%.5f, min2=%.5f)', EPOCH-1, MIN_LOSS, MIN_LOSS2)})
            
            AE:training()
            AE2:training()
        end
    end
end

function imageListToTensor(imageList)
    local tens = torch.Tensor(#imageList, imageList[1]:size(1), imageList[1]:size(2), imageList[1]:size(3))
    for i=1,#imageList do
        tens[i] = imageList[i]
    end
    return tens
end

-- Get examples to plot
function getSamples(ds, N)
    local images = torch.Tensor(N, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    for i=1, N do
        images[i] = ds[i]
    end

    local decoded = AE:forward(images)
    local decoded2 = AE2:forward(decoded)
    
    return {images:clone(), decoded:clone(), decoded2:clone()}
end

-- train one epoch
function train(usedDataset)
    EPOCH = EPOCH or 1
    local N = usedDataset:size()
    local time = sys.clock()

    local shuffle = torch.randperm(N)
    
    local sumLossCriterion = 0
    local sumLossCriterion2 = 0

    -- do one epoch
    print("<trainer> online epoch # " .. EPOCH .. ' [batchSize = ' .. OPT.batchSize .. ']')
    for t = 1,N,OPT.batchSize do
        -- if the last batch has a size smaller than opt.batchSize, adjust for that
        local thisBatchSize = math.min(OPT.batchSize, N - t + 1)
        local inputs = torch.Tensor(thisBatchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        local targets = torch.Tensor(thisBatchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        
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
            local outputs = AE:forward(inputs)
            local f = CRITERION:forward(outputs, targets)
            sumLossCriterion = sumLossCriterion + CRITERION.output

            -- backward pass 
            local df_do = CRITERION:backward(outputs, targets)
            AE:backward(inputs, df_do)

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

            if OPT.AE_clamp ~= 0 then
                GRAD_PARAMETERS_AE:clamp((-1)*OPT.AE_clamp, OPT.AE_clamp)
            end

            return f, GRAD_PARAMETERS_AE
        end

        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of discriminator
        local fevalAE2 = function(x)
            collectgarbage()

            if x ~= PARAMETERS_AE2 then -- get new parameters
                PARAMETERS_AE2:copy(x)
            end

            GRAD_PARAMETERS_AE2:zero() -- reset gradients

            --  forward pass
            local outputs1 = AE:forward(inputs)
            local outputs2 = AE2:forward(outputs1)
            local f = CRITERION2:forward(outputs2, targets)
            sumLossCriterion2 = sumLossCriterion2 + CRITERION2.output

            -- backward pass 
            local df_do = CRITERION2:backward(outputs2, targets)
            AE2:backward(outputs1, df_do)

            -- penalties (L1 and L2):
            if OPT.coefL1 ~= 0 or OPT.coefL2 ~= 0 then
                local norm = torch.norm
                local sign = torch.sign
                -- Loss:
                f = f + OPT.coefL1 * torch.norm(PARAMETERS_AE2, 1)
                f = f + OPT.coefL2 * torch.norm(PARAMETERS_AE2, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_AE2:add( torch.sign(PARAMETERS_AE2):mul(OPT.coefL1) + PARAMETERS_AE2:clone():mul(OPT.coefL2) )
            end

            if OPT.AE_clamp ~= 0 then
                GRAD_PARAMETERS_AE2:clamp((-1)*OPT.AE_clamp, OPT.AE_clamp)
            end

            return f, GRAD_PARAMETERS_AE2
        end

        optim.adam(fevalAE, PARAMETERS_AE, OPTSTATE.adam)
        optim.adam(fevalAE2, PARAMETERS_AE2, OPTSTATE.adam)
        --optim.adagrad(fevalAE, PARAMETERS_AE, OPTSTATE.adagrad)

        -- display progress
        xlua.progress(t + thisBatchSize, usedDataset:size())
    end

    -- time taken
    time = sys.clock() - time
    print("<trainer> time required for this epoch = " .. (time) .. "s")
    time = time / usedDataset:size()
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print("<trainer> loss1 = " .. (sumLossCriterion/(N/OPT.batchSize)))
    print("<trainer> loss2 = " .. (sumLossCriterion2/(N/OPT.batchSize)))

    -- save/log current net
    if EPOCH % OPT.saveFreq == 0 then
        local filename = paths.concat(OPT.save, string.format('denoiser_%dx%dx%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('<trainer> saving network to '..filename)
        local AE1_nocuda = NN_UTILS.deactivateCuda(AE)
        local AE2_nocuda = NN_UTILS.deactivateCuda(AE2)
        torch.save(filename, {AE1_ENCODER = AE1_nocuda:get(1),
                              AE1_DECODER = AE1_nocuda:get(2),
                              AE2_DECODER = AE2_nocuda})
    end

    -- next epoch
    EPOCH = EPOCH + 1
    
    return sumLossCriterion/(N/OPT.batchSize), sumLossCriterion2/(N/OPT.batchSize)
end

function exitIfNaNs(checkIn)
    local nanCount = checkIn:ne(checkIn):sum()
    if nanCount > 0 then
        print("[ERROR] Detected " .. nanCount .. " NaNs. Exiting.")
        os.exit()
    end
end


main()
