require 'torch'
require 'image'
NN_UTILS = require 'utils.nn_utils'

OPT = lapp[[
    --network       (default "")          reload pretrained network
    --dir           (default "samples")   directory to save the images to
]]

torch.manualSeed(OPT.seed)
print("<trainer> starting gpu support...")
require 'cutorch'
require 'cunn'
cutorch.setDevice(OPT.gpu + 1)
cutorch.manualSeed(OPT.seed)

function main()
    local tmp = torch.load(OPT.network)
    MODEL_D = tmp.D
    MODEL_G = tmp.G
    
    MODEL_D:cuda()
    MODEL_G:cuda()
    
    local images = NN_UTILS.createImagesFromNoise(1000, false)
    local images, local predictions = NN_UTILS.sortImagesByPrediction(images, false, 64)
    for i=1,#images do
        image.save(paths.concat(OPT.dir, string.format('%02d.png', i)), images[i])
    end
end

main()
