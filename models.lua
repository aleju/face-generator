require 'torch'
require 'nn'
require 'layers.cudnnSpatialConvolutionUpsample'

local models = {}

-- Create base G network.
-- Color and grayscale currently get the same network, which is suboptimal.
-- @param dimensions Image dimensions as table of {count channels, height, width}
-- @param noiseDim Size of the noise vector as integer, e.g. 100
-- @returns Sequential
function models.create_G(dimensions, noiseDim)
    local inputSz = dimensions[1] * dimensions[2] * dimensions[3]
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 2048))
    model:add(nn.PReLU())
    model:add(nn.Linear(2048, inputSz))
    model:add(nn.Sigmoid())
    model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))
    return model
end

-- Creates the decoder part of an upsampling height-16px G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder_upsampling16(dimensions, noiseDim)
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 128*4*4))
    model:add(nn.View(128, 4, 4))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())
    
    --model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates the decoder part of an upsampling height-32px G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder_upsampling32(dimensions, noiseDim)
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 128*8*8))
    model:add(nn.View(128, 8, 8))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())
    
    --model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates G, which is identical to the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G(dimensions, noiseDim)
    if dimensions[2] == 16 then
        return models.create_G_decoder_upsampling16(dimensions, noiseDim)
    else
        return models.create_G_decoder_upsampling32(dimensions, noiseDim)
    end
end

-- Create base D network.
-- @param dimensions Image dimensions as table of {count channels, height, width}
-- @returns Sequential
function models.create_D(dimensions)
    if dimensions[2] == 16 then
        return models.create_D16_d(dimensions)
    else
        return models.create_D32b(dimensions)
    end
end

-- Create base D network for 16x16 images.
-- Color and grayscale currently get the same network, which is suboptimal.
-- @param dimensions Image dimensions as table of {count channels, height, width}
-- @returns Sequential
function models.create_D16(dimensions)
    local inputSz = dimensions[1] * dimensions[2] * dimensions[3]
    local branch_conv_fine = nn.Sequential()
    branch_conv_fine:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialMaxPooling(2, 2))
    branch_conv_fine:add(nn.SpatialDropout())
    branch_conv_fine:add(nn.View(64 * (1/4) * dimensions[2] * dimensions[3]))
    branch_conv_fine:add(nn.Linear(64 * (1/4) * dimensions[2] * dimensions[3], 1024))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.Dropout())
    
    local branch_conv_coarse = nn.Sequential()
    branch_conv_coarse:add(nn.SpatialConvolution(dimensions[1], 32, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialMaxPooling(2, 2))
    branch_conv_coarse:add(nn.SpatialDropout())
    branch_conv_coarse:add(nn.View(64 * (1/4) * dimensions[2] * dimensions[3]))
    branch_conv_coarse:add(nn.Linear(64 * (1/4) * dimensions[2] * dimensions[3], 1024))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.Dropout())

    local branch_dense = nn.Sequential()
    branch_dense:add(nn.View(inputSz))
    branch_dense:add(nn.Linear(inputSz, 1024))
    branch_dense:add(nn.PReLU())
    branch_dense:add(nn.Dropout())
    branch_dense:add(nn.Linear(1024, 1024))
    branch_dense:add(nn.PReLU())

    local concat = nn.ConcatTable()
    concat:add(branch_conv_fine)
    concat:add(branch_conv_coarse)
    concat:add(branch_dense)

    local model = nn.Sequential()
    model:add(concat)
    model:add(nn.JoinTable(2))
    model:add(nn.Linear(1024 + 1024 + 1024, 1024))
    model:add(nn.PReLU())
    model:add(nn.Dropout())
    model:add(nn.Linear(1024, 1))
    model:add(nn.Sigmoid())
    
    return model
end

function models.create_D16_b(dimensions)
    local inputSz = dimensions[1] * dimensions[2] * dimensions[3]
    local branch_conv_fine = nn.Sequential()
    branch_conv_fine:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(128, 128, 3, 3, 2, 2, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialDropout())
    branch_conv_fine:add(nn.View(128 * 0.25 * dimensions[2] * dimensions[3]))
    branch_conv_fine:add(nn.Linear(128 * 0.25 * dimensions[2] * dimensions[3], 512))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.Dropout())
    
    local branch_conv_coarse = nn.Sequential()
    branch_conv_coarse:add(nn.SpatialConvolution(dimensions[1], 64, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(128, 128, 5, 5, 2, 2, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialDropout())
    branch_conv_coarse:add(nn.View(128 * 0.25 * dimensions[2] * dimensions[3]))
    branch_conv_coarse:add(nn.Linear(128 * 0.25 * dimensions[2] * dimensions[3], 512))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.Dropout())

    local branch_dense = nn.Sequential()
    branch_dense:add(nn.View(inputSz))
    branch_dense:add(nn.Linear(inputSz, 1024))
    branch_dense:add(nn.PReLU())
    branch_dense:add(nn.Dropout())
    branch_dense:add(nn.Linear(1024, 1024))
    branch_dense:add(nn.PReLU())

    local concat = nn.ConcatTable()
    concat:add(branch_conv_fine)
    concat:add(branch_conv_coarse)
    concat:add(branch_dense)

    local model = nn.Sequential()
    model:add(concat)
    model:add(nn.JoinTable(2))
    model:add(nn.Linear(512 + 512 + 1024, 1024))
    model:add(nn.PReLU())
    model:add(nn.Dropout())
    model:add(nn.Linear(1024, 1))
    model:add(nn.Sigmoid())
    
    return model
end

function models.create_D16_c(dimensions)
    local inputSz = dimensions[1] * dimensions[2] * dimensions[3]
    local branch_conv_fine = nn.Sequential()
    branch_conv_fine:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(128, 128, 3, 3, 2, 2, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(128, 512, 3, 3, 2, 2, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialDropout())
    local branch_conv_fine_size = 512 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    branch_conv_fine:add(nn.View(branch_conv_fine_size))
    branch_conv_fine:add(nn.Linear(branch_conv_fine_size, 1024))
    branch_conv_fine:add(nn.PReLU())
    
    local branch_conv_coarse = nn.Sequential()
    branch_conv_coarse:add(nn.SpatialConvolution(dimensions[1], 64, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(128, 128, 5, 5, 2, 2, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(128, 512, 5, 5, 2, 2, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialDropout())
    local branch_conv_coarse_size = 512 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    branch_conv_coarse:add(nn.View(branch_conv_coarse_size))
    branch_conv_coarse:add(nn.Linear(branch_conv_coarse_size, 1024))
    branch_conv_coarse:add(nn.PReLU())
    
    local branch_dense = nn.Sequential()
    branch_dense:add(nn.View(inputSz))
    branch_dense:add(nn.Linear(inputSz, 1024))
    branch_dense:add(nn.PReLU())
    branch_dense:add(nn.Dropout())
    branch_dense:add(nn.Linear(1024, 1024))
    branch_dense:add(nn.PReLU())

    local concat = nn.ConcatTable()
    concat:add(branch_conv_fine)
    concat:add(branch_conv_coarse)
    concat:add(branch_dense)

    local model = nn.Sequential()
    model:add(concat)
    model:add(nn.JoinTable(2))
    model:add(nn.Linear(1024 + 1024 + 1024, 1024))
    model:add(nn.PReLU())
    model:add(nn.Dropout())
    model:add(nn.Linear(1024, 1))
    model:add(nn.Sigmoid())
    
    return model
end

function models.create_D16_d(dimensions)
    local inputSz = dimensions[1] * dimensions[2] * dimensions[3]
    local branch_conv_fine = nn.Sequential()
    branch_conv_fine:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    branch_conv_fine:add(nn.SpatialConvolution(128, 512, 3, 3, 2, 2, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(512, 1024, 3, 3, 2, 2, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialDropout())
    local branch_conv_fine_size = 1024 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    branch_conv_fine:add(nn.View(branch_conv_fine_size))
    branch_conv_fine:add(nn.Linear(branch_conv_fine_size, 1024))
    branch_conv_fine:add(nn.PReLU())
    
    local branch_dense = nn.Sequential()
    branch_dense:add(nn.View(inputSz))
    branch_dense:add(nn.Linear(inputSz, 128))
    branch_dense:add(nn.PReLU())
    branch_dense:add(nn.Dropout())
    branch_dense:add(nn.Linear(128, 128))
    branch_dense:add(nn.PReLU())
    
    local concat = nn.ConcatTable()
    concat:add(branch_conv_fine)
    concat:add(branch_dense)
    
    local model = nn.Sequential()
    model:add(concat)
    model:add(nn.JoinTable(2))
    model:add(nn.Linear(1024 + 128, 1))
    model:add(nn.Sigmoid())
    
    return model
end

-- Create base D network for 32x32 images.
-- Color and grayscale currently get the same network, which is suboptimal.
-- @param dimensions Image dimensions as table of {count channels, height, width}
-- @returns Sequential
function models.create_D32(dimensions)
    local branch_conv_fine = nn.Sequential()
    branch_conv_fine:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 64, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch_conv_fine:add(nn.PReLU())
    branch_conv_fine:add(nn.SpatialMaxPooling(2, 2))
    branch_conv_fine:add(nn.SpatialDropout())
    branch_conv_fine:add(nn.View(64 * (1/4) * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]))
    branch_conv_fine:add(nn.Linear(64 * (1/4) * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3], 1024))
    branch_conv_fine:add(nn.PReLU())
    
    local branch_conv_coarse = nn.Sequential()
    branch_conv_coarse:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 32, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialMaxPooling(2, 2))
    branch_conv_coarse:add(nn.SpatialConvolution(32, 54, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialConvolution(54, 54, 5, 5, 1, 1, (5-1)/2))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.SpatialMaxPooling(2, 2))
    branch_conv_coarse:add(nn.SpatialDropout())
    branch_conv_coarse:add(nn.View(54 * (1/4) * (1/4) * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]))
    branch_conv_coarse:add(nn.Linear(54 * (1/4) * (1/4) * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3], 1024))
    branch_conv_coarse:add(nn.PReLU())
    branch_conv_coarse:add(nn.Dropout())
    branch_conv_coarse:add(nn.Linear(1024, 1024))
    branch_conv_coarse:add(nn.PReLU())

    local branch_dense = nn.Sequential()
    branch_dense:add(nn.View(INPUT_SZ))
    branch_dense:add(nn.Linear(INPUT_SZ, 1024))
    branch_dense:add(nn.PReLU())
    branch_dense:add(nn.Dropout())
    branch_dense:add(nn.Linear(1024, 1024))
    branch_dense:add(nn.PReLU())

    local concat = nn.ConcatTable()
    concat:add(branch_conv_fine)
    concat:add(branch_conv_coarse)
    concat:add(branch_dense)

    local model = nn.Sequential()
    model:add(concat)
    model:add(nn.JoinTable(2))
    model:add(nn.Linear(1024 + 1024 + 1024, 1024))
    model:add(nn.PReLU())
    model:add(nn.Dropout())
    model:add(nn.Linear(1024, 1))
    model:add(nn.Sigmoid())

    return model
end

-- Create base D network for 32x32 images.
-- Color and grayscale currently get the same network, which is suboptimal.
-- @param dimensions Image dimensions as table of {count channels, height, width}
-- @returns Sequential
function models.create_D32b(dimensions)
    local conv = nn.Sequential()
    
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    conv:add(nn.View(512 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(512 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 512))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 512))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    return conv
end

return models
