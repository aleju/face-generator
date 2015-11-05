require 'torch'
require 'nn'
require 'layers.cudnnSpatialConvolutionUpsample'

local models = {}

-- Create coarse to fine G network.
-- Color and grayscale currently get the same network, which is suboptimal.
-- @param dimensions Image dimensions as table of {count channels, height, width}
-- @param cuda Whether to start the network in CUDA/GPU mode (true).
-- @returns Sequential
function models.create_G(dimensions, cuda)
    local nplanes = 64
    local model_G = nn.Sequential()
    
    model_G:add(nn.JoinTable(2, 2))
    
    if cuda then
        model_G:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    inner:add(cudnn.SpatialConvolutionUpsample(dimensions[1]+1, 64, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(64, 128, 7, 7, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(128, dimensions[1], 5, 5, 1))
    inner:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))
    model_G:add(inner)
    if cuda then
        model_G:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
    end
    
    model_G = require('weight-init')(model_G, 'heuristic')
    
    if cuda then
        model_G:get(3):cuda()
    end
    
    return model_G
end

-- Create coarse to fine D network.
-- Color and grayscale currently get the same network, which is suboptimal.
-- @param dimensions Image dimensions as table of {count channels, height, width}
-- @param cuda Whether to start the network in CUDA/GPU mode (true).
-- @returns Sequential
function models.create_D(dimensions, cuda)
    local nplanes = 64
    local model_D = nn.Sequential()
    
    model_D:add(nn.CAddTable())
    if cuda then
        model_D:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    
    inner:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialMaxPooling(2, 2))
    inner:add(nn.Dropout())
    
    inner:add(nn.View(64 * 0.25 * dimensions[2] * dimensions[3]))
    
    inner:add(nn.Linear(64 * 0.25 * dimensions[2] * dimensions[3], 512))
    inner:add(nn.PReLU())
    inner:add(nn.Dropout())
    inner:add(nn.Linear(512, 1))
    inner:add(nn.Sigmoid())
    model_D:add(inner)
    if cuda then
        model_D:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
    end

    model_D = require('weight-init')(model_D, 'heuristic')

    if cuda then
        model_D:get(3):cuda()
    end

    return model_D
end

return models
