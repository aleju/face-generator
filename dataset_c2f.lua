require 'torch'
require 'image'
require 'paths'

local dataset = {}

-- load data from these directories
dataset.dirs = {}
-- load only images with these file extensions
dataset.fileExtension = ""

-- expected original height/width of images
dataset.originalScale = 64
-- desired height/width of images
dataset.fineScale = 32
dataset.coarseScale = 16
-- desired channels of images (1=grayscale, 3=color)
dataset.nbChannels = 3

-- cache for filepaths to all images
dataset.paths = nil

-- Set directories to load images from
-- @param dirs List of paths to directories
function dataset.setDirs(dirs)
  dataset.dirs = dirs
end

-- Set file extension that images to load must have
-- @param fileExtension the file extension of the images
function dataset.setFileExtension(fileExtension)
  dataset.fileExtension = fileExtension
end

function dataset.setFineScale(scale)
  dataset.fineScale = scale
end

function dataset.setCoarseScale(scale)
  dataset.coarseScale = scale
end

-- Set desired number of channels for the images (1=grayscale, 3=color)
-- @param nbChannels The number of channels
function dataset.setNbChannels(nbChannels)
  dataset.nbChannels = nbChannels
end

function dataset._toResult(fineImages)
    local N = fineImages:size(1)
    
    local coarseImages = torch.FloatTensor(N, dataset.nbChannels, dataset.fineScale, dataset.fineScale)
    for i=1,N do
        local tmp = image.scale(fineImages[i], dataset.coarseScale, dataset.coarseScale)
        coarseImages[i] = image.scale(tmp, dataset.fineScale, dataset.fineScale)
    end
    
    local diffImages = torch.FloatTensor(N, dataset.nbChannels, dataset.fineScale, dataset.fineScale)
    for i=1,N do
        diffImages[i] = torch.add(fineImages[i], -1, coarseImages[i])
    end


    local result = {}
    result.fine = fineImages
    result.coarse = coarseImages
    result.diff = diffImages
    
    function result:size()
        return N
    end
    
    function result:getCoarse(index, endIndex)
        if endIndex ~= nil then
            return result.coarse[{index, endIndex}]
        else
            return result.coarse[index]
        end
    end
    
    function result:getFine(index)
        if endIndex ~= nil then
            return result.fine[{index, endIndex}]
        else
            return result.fine[index]
        end
    end
    
    function result:getDiff(index)
        if endIndex ~= nil then
            return result.diff[{index, endIndex}]
        else
            return result.diff[index]
        end
    end
    

    setmetatable(result, {
        __index = function(self, index)
            local c = self.coarse[index]
            local f = self.fine[index]
            local d = self.diff[index]
            return {coarse = c, fine = f, diff = d}
        end,
        __len = function(self) return self.fine:size(1) end
    })
    
    return result
end

-- Load images from the dataset.
-- @param startAt Number of the first image.
-- @param count Count of the images to load.
-- @return Table of images. You can call :size() on that table to get the number of loaded images.
function dataset.loadImages(startAt, count)
    local endBefore = startAt + count
    
    if dataset.paths == nil then
        dataset.loadPaths()
    end

    local N = math.min(count, #dataset.paths)
    local data = torch.FloatTensor(N, dataset.nbChannels, dataset.fineScale, dataset.fineScale)
    for i=1,N do
        local img = image.load(dataset.paths[startAt + i], dataset.nbChannels, "float")
        data[i] = image.scale(img, dataset.fineScale, dataset.fineScale)
    end

    print(string.format('<dataset> loaded %d examples', N))

    return dataset._toResult(data)
end

-- Loads the paths of all images in the defined files
-- (with defined file extensions)
function dataset.loadPaths()
    local files = {}
    local dirs = dataset.dirs
    local ext = dataset.fileExtension

    for i=1, #dirs do
        local dir = dirs[i]
        -- Go over all files in directory. We use an iterator, paths.files().
        for file in paths.files(dir) do
            -- We only load files that match the extension
            if file:find(ext .. '$') then
                -- and insert the ones we care about in our table
                table.insert(files, paths.concat(dir,file))
            end
        end

        -- Check files
        if #files == 0 then
            error('given directory doesnt contain any files of type: ' .. ext)
        end
    end
    
    print(string.format("<dataset> Loaded %d filepaths", #files))
    
    dataset.paths = files
end

-- Loads a defined number of randomly selected images from
-- the cached paths (cached in loadPaths()).
-- @param count Number of random images.
-- @return List of Tensors
function dataset.loadRandomImages(count, startAt)
    if startAt == nil then
        startAt = 0
    end
    
    --local images = dataset.loadRandomImagesFromDirs(dataset.dirs, dataset.fileExtension, count)
    local images = dataset.loadRandomImagesFromPaths(count, startAt)
    local N = #images
    local data = torch.FloatTensor(N, dataset.nbChannels, dataset.fineScale, dataset.fineScale)
    for i=1, N do
        data[i] = image.scale(images[i], dataset.fineScale, dataset.fineScale)
    end

    --local ker = torch.ones(3)
    --local m = nn.SpatialSubtractiveNormalization(1, ker)
    --data = m:forward(data)

    print(string.format('<dataset> loaded %d random examples', N))

    return dataset._toResult(data)
end

-- Loads randomly selected images from the cached paths.
-- TODO: merge with loadRandomImages()
-- @param count Number of images to load
-- @param startAt Minimum allowed index of any returned image among the ordered list of paths
--                First image has index 0.
-- @returns List of Tensors
function dataset.loadRandomImagesFromPaths(count, startAt)
    if startAt == nil then
        startAt = 0
    end
    
    if dataset.paths == nil then
        dataset.loadPaths()
    end

    local shuffle = torch.randperm(#dataset.paths - startAt)
    
    local images = {}
    for i=1,math.min(shuffle:size(1), count) do
       -- load each image
       table.insert(images, image.load(dataset.paths[shuffle[i] + startAt], dataset.nbChannels, "float"))
    end
    
    return images
end

return dataset
