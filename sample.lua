require 'torch'
require 'image'
require 'paths'
require 'pl'
require 'layers.cudnnSpatialConvolutionUpsample'
NN_UTILS = require 'utils.nn_utils'

OPT = lapp[[
    --save_base     (default "logs")                 directory in which the networks are saved
    --save_c2f32    (default "logs")
    --G_base        (default "adversarial.net")      
    --D_base        (default "adversarial.net")      
    --G_c2f32       (default "adversarial_c2f_16_to_32.net")  
    --D_c2f32       (default "adversarial_c2f_16_to_32.net")  
    --writeto       (default "samples")              directory to save the images to
    --seed          (default 1)
    --gpu           (default 0)                      GPU to run on
    --runs          (default 3)                      How often to sample and save images
    --noiseDim      (default 100)
    --batchSize     (default 16)
]]

print(OPT)

if OPT.gpu < 0 then
    print("[ERROR] Sample script currently only runs on GPU, set --gpu=x where x is between 0 and 3.")
    exit()
end

torch.manualSeed(OPT.seed)
print("Starting gpu support...")
require 'cutorch'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)
cutorch.setDevice(OPT.gpu + 1)
cutorch.manualSeed(OPT.seed)

function main()
    MODEL_G, MODEL_D, MODEL_G_C2F_32, MODEL_D_C2F_32 = loadModels()
    
    MODEL_G = NN_UTILS.activateCuda(MODEL_G)
    MODEL_D = NN_UTILS.activateCuda(MODEL_D)
    MODEL_G_C2F_32 = NN_UTILS.activateCuda(MODEL_G_C2F_32)
    MODEL_D_C2F_32 = NN_UTILS.activateCuda(MODEL_D_C2F_32)
    
    print("Sampling...")
    for run=1,OPT.runs do
        local images = NN_UTILS.createImages(1000, false)
        local imagesBest, predictions = NN_UTILS.sortImagesByPrediction(images, false, 64)
        local imagesWorst, predictions = NN_UTILS.sortImagesByPrediction(images, true, 64)
        local imagesRandom = selectRandomImagesFrom(images, 64)
        image.save(paths.concat(OPT.writeto, string.format('best_%04d_base.jpg', run)), toGrid(imagesBest, 8))
        image.save(paths.concat(OPT.writeto, string.format('worst_%04d_base.jpg', run)), toGrid(imagesWorst, 8))
        image.save(paths.concat(OPT.writeto, string.format('random_%04d_base.jpg', run)), toGrid(imagesRandom, 8))
        
        local imagesBestC2F32 = c2f(imagesBest, MODEL_G_C2F_32, MODEL_D_C2F_32, 32)
        local imagesWorstC2F32 = c2f(imagesWorst, MODEL_G_C2F_32, MODEL_D_C2F_32, 32)
        local imagesRandomC2F32 = c2f(imagesRandom, MODEL_G_C2F_32, MODEL_D_C2F_32, 32)
        image.save(paths.concat(OPT.writeto, string.format('best_%04d_c2f_32.jpg', run)), toGrid(imagesBestC2F32, 8))
        image.save(paths.concat(OPT.writeto, string.format('worst_%04d_c2f_32.jpg', run)), toGrid(imagesWorstC2F32, 8))
        image.save(paths.concat(OPT.writeto, string.format('random_%04d_c2f_32.jpg', run)), toGrid(imagesRandomC2F32, 8))
        
        xlua.progress(run, OPT.runs)
    end
    
    print("Finished.")
end

function c2f(images, G, D, fineSize)
    local triesPerImage = 10
    local result = {}
    for i=1,#images do
        local imgTensor = torch.Tensor(triesPerImage, images[1]:size(1), fineSize, fineSize)
        local img = images[i]:clone()
        local height = img:size(2)
        local width = img:size(3)
        
        if height ~= fineSize or width ~= fineSize then
            img = image.scale(img, fineSize, fineSize)
        end
        
        for j=1,triesPerImage do
            imgTensor[j] = img:clone()
        end
        
        local noiseInputs = torch.Tensor(triesPerImage, 1, fineSize, fineSize)
        noiseInputs:uniform(-1, 1)
        
        local diffs = G:forward({noiseInputs, imgTensor})
        --diffs:float()
        local predictions = D:forward({diffs, imgTensor})
        
        local maxval = nil
        local maxdiff = nil
        for j=1,triesPerImage do
            if maxval == nil or predictions[j][1] > maxval then
                maxval = predictions[j][1]
                maxdiff = diffs[j]
            end
        end
        
        local imgRefined = torch.add(img, maxdiff)
        table.insert(result, imgRefined)
    end
    
    return result
end

function blur(img)
    local img2 = image.convolve(img:clone(), image.gaussian(3), "same")
    return img2
end

function toGrid(images, nrow)
    return image.toDisplayTensor{input=images, nrow=nrow}
end

function selectRandomImagesFrom(tensor, n)
    local shuffle = torch.randperm(tensor:size(1))
    local result = {}
    for i=1,math.min(n, tensor:size(1)) do
        table.insert(result, tensor[ shuffle[i] ])
    end
    return result
end

function loadModels()
    local file
    
    -- load G base
    file = torch.load(paths.concat(OPT.save_base, OPT.G_base))
    local G = file.G
    
    -- load D base
    if OPT.D_base ~= OPT.G_base then
        file = torch.load(paths.concat(OPT.save_base, OPT.D_base))
    end
    local D = file.D:float()
    
    -- load G c2f size 32
    file = torch.load(paths.concat(OPT.save_c2f32, OPT.G_c2f32))
    local G_c2f32 = file.G
    
    -- load D c2f size 32
    if OPT.D_c2f32 ~= OPT.G_c2f32 then
        file = torch.load(paths.concat(OPT.save_c2f32, OPT.D_c2f32))
    end
    local D_c2f32 = file.D
    
    return G, D, G_c2f32, D_c2f32
end

main()
