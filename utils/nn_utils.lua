require 'torch'

local nn_utils = {}

function nn_utils.setWeights(weights, std)
    weights:randn(weights:size())
    weights:mul(std)
end

function nn_utils.initializeWeights(model, stdWeights, stdBias)
    stdWeights = stdWeights or 0.005
    stdBias = stdBias or 0.001
    
    for m = 1, #model.modules do
        if model.modules[m].weight then
            nn_utils.setWeights(model.modules[m].weight, stdWeights)
        end
        if model.modules[m].bias then
            nn_utils.setWeights(model.modules[m].bias, stdBias)
        end
    end
end

function nn_utils.createNoiseInputs(N)
    local noiseInputs = torch.Tensor(N, OPT.noiseDim)
    --noiseInputs:normal(0.0, 0.35)
    noiseInputs:uniform(-1.0, 1.0)
    return noiseInputs
end

function nn_utils.createImagesFromNoise(noiseInputs, outputAsList, refineWithG)
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

function nn_utils.createImages(N, outputAsList, refineWithG)
    return nn_utils.createImagesFromNoise(createNoiseInputs(N), outputAsList, refineWithG)
end

function nn_utils.sortImagesByPrediction(images, ascending, nbMaxOut)
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

function nn_utils.visualizeProgress(noiseInputs)
    nn_utils.switchToEvaluationMode()
    
    local semiRandomImagesUnrefined
    if MODEL_AE then
        semiRandomImagesUnrefined = nn_utils.createImagesFromNoise(noiseInputs, true, false)
    end
    local semiRandomImagesRefined = nn_utils.createImagesFromNoise(noiseInputs, true, true)
    
    local sanityTestImage = torch.Tensor(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
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
    
    local trainImages = torch.Tensor(50, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    for i=1,50 do
        trainImages[i] = TRAIN_DATA[i]
    end
    
    local randomImages = createImages(300, false)
    randomImages[299] = TRAIN_DATA[3] -- one real face as sanity test
    randomImages[300] = sanityTestImage -- synthetic non-face as sanity test
    local badImages, _ = nn_utils.sortImagesByPrediction(randomImages, true, 50)
    local goodImages, _ = nn_utils.sortImagesByPrediction(randomImages, false, 50)

    if OPT.gpu then
        torch.setdefaulttensortype('torch.FloatTensor')
    end

    if semiRandomImagesUnrefined then
        DISP.image(semiRandomImagesUnrefined, {win=OPT.window, width=IMG_DIMENSIONS[3]*15, title="semi-random generated images (before G)"})
    end
    DISP.image(semiRandomImagesRefined, {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="semi-random generated images (after G)"})
    DISP.image(goodImages, {win=OPT.window+2, width=IMG_DIMENSIONS[3]*15, title="best samples (first is best)"})
    DISP.image(badImages, {win=OPT.window+3, width=IMG_DIMENSIONS[3]*15, title="worst samples (first is worst)"})
    DISP.image(trainImages, {win=OPT.window+4, width=IMG_DIMENSIONS[3]*15, title="original images from training set"})
    
    if OPT.gpu then
        torch.setdefaulttensortype('torch.CudaTensor')
    else
        torch.setdefaulttensortype('torch.FloatTensor')
    end
    
    nn_utils.switchToTrainingMode()
end

function nn_utils.switchToTrainingMode()
    if MODEL_AE then
        MODEL_AE:training()
    end
    MODEL_G:training()
    MODEL_D:training()
end

function nn_utils.switchToEvaluationMode()
    if MODEL_AE then
        MODEL_AE:evaluate()
    end
    MODEL_G:evaluate()
    MODEL_D:evaluate()
end

return nn_utils
