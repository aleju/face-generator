require 'torch'
require 'optim'
require 'image'
require 'pl'
require 'paths'
--image_utils = require 'utils.image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
ADVERSARIAL = require 'adversarial_c2f'
DATASET = require 'dataset_c2f'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models_c2f'

----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  --save             (default "logs")       subdirectory to save logs
  --saveFreq         (default 30)           save every saveFreq epochs
  --network          (default "")           reload pretrained network
  --noplot                                  plot while training
  --D_sgd_lr         (default 0.02)         D SGD learning rate
  --G_sgd_lr         (default 0.02)         G SGD learning rate
  --D_sgd_momentum   (default 0)            D SGD momentum
  --G_sgd_momentum   (default 0)            G SGD momentum
  --batchSize        (default 32)           batch size
  --N_epoch          (default 1000)         Number of examples per epoch (-1 means all)
  --G_L1             (default 0)            L1 penalty on the weights of G
  --G_L2             (default 0e-6)         L2 penalty on the weights of G
  --D_L1             (default 1e-7)         L1 penalty on the weights of D
  --D_L2             (default 0e-6)         L2 penalty on the weights of D
  --D_iterations     (default 1)            number of iterations to optimize D for
  --G_iterations     (default 1)            number of iterations to optimize G for
  --D_clamp          (default 1)            Clamp threshold for D's gradient (+/- N)
  --G_clamp          (default 5)            Clamp threshold for G's gradient (+/- N)
  --D_optmethod      (default "adam")       adam|adagrad|sgd
  --G_optmethod      (default "adam")       adam|adagrad|sgd
  --threads          (default 4)            number of threads
  --gpu              (default 0)            gpu to run on (default cpu)
  --noiseDim         (default 100)          dimensionality of noise vector
  --window           (default 3)            window id of sample image
  --coarseSize       (default 16)           coarse scale
  --fineSize         (default 32)           fine scale
  --grayscale                               grayscale mode on/off
  --seed             (default 1)            seed for the RNG
  --aws                                     run in AWS mode
]]

if OPT.fineSize ~= 32 then
    print("[Warning] Models are currently only optimized for fine size of 32.")
end

START_TIME = os.time()

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

-- fix seed
math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)

-- threads
torch.setnumthreads(OPT.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- possible output of disciminator
CLASSES = {"0", "1"}
Y_GENERATOR = 0
Y_NOT_GENERATOR = 1

-- axis of images: 3 channels, <scale> height, <scale> width
if OPT.grayscale then
    IMG_DIMENSIONS = {1, OPT.fineSize, OPT.fineSize}
    COND_DIM = {1, OPT.fineSize, OPT.fineSize}
else
    IMG_DIMENSIONS = {3, OPT.fineSize, OPT.fineSize}
    COND_DIM = {3, OPT.fineSize, OPT.fineSize}
end
-- size in values/pixels per input image (channels*height*width)
INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]
NOISE_DIM = {1, OPT.fineSize, OPT.fineSize}

----------------------------------------------------------------------
-- get/create dataset
----------------------------------------------------------------------
DATASET.nbChannels = IMG_DIMENSIONS[1]
DATASET.setFileExtension("jpg")
DATASET.setCoarseScale(OPT.coarseSize)
DATASET.setFineScale(OPT.fineSize)

if OPT.aws then
    DATASET.setDirs({"/mnt/datasets/out_aug_64x64"})
else
    DATASET.setDirs({"dataset/out_aug_64x64/faces"})
end
----------------------------------------------------------------------

-- run on gpu if chosen
print("<trainer> starting gpu support...")
require 'nn'
require 'cutorch'
require 'cunn'
if OPT.gpu then
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
end
torch.setdefaulttensortype('torch.FloatTensor')

if OPT.network ~= "" then
    print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
    local tmp = torch.load(OPT.network)
    MODEL_D = tmp.D
    MODEL_G = tmp.G
    OPTSTATE = tmp.optstate
    EPOCH = tmp.epoch
    
    if OPT.gpu ~= false then
        MODEL_D:float()
        MODEL_G:float()
    end
else
    MODEL_D = MODELS.create_D(IMG_DIMENSIONS, OPT.gpu ~= false)
    MODEL_G = MODELS.create_G(IMG_DIMENSIONS, OPT.noiseDim, OPT.gpu ~= false)
end
    
-- loss function: negative log-likelihood
CRITERION = nn.BCECriterion()

-- retrieve parameters and gradients
PARAMETERS_D, GRAD_PARAMETERS_D = MODEL_D:getParameters()
PARAMETERS_G, GRAD_PARAMETERS_G = MODEL_G:getParameters()

-- this matrix records the current confusion across classes
CONFUSION = optim.ConfusionMatrix(CLASSES)

print("Model D:")
print(MODEL_D)
print("Model G:")
print(MODEL_G)

-- count free parameters in D/G
local nparams = 0
local dModules = MODEL_D:listModules()
for i=1,#dModules do
  if dModules[i].weight ~= nil then
    nparams = nparams + dModules[i].weight:nElement()
  end
end
print('\nNumber of free parameters in D: ' .. nparams)


local nparams = 0
local gModules = MODEL_G:listModules()
for i=1,#gModules do
  if gModules[i].weight ~= nil then
    nparams = nparams + gModules[i].weight:nElement()
  end
end
print('Number of free parameters in G: ' .. nparams .. '\n')

-- Set optimizer state
if OPTSTATE == nil or OPT.rebuildOptstate == 1 then
    OPTSTATE = {
        adagrad = {
            D = { learningRate = 1e-3 },
            G = { learningRate = 1e-3 * 3 }
        },
        adam = {
            D = {},
            G = {}
        },
        rmsprop = {D = {}, G = {}},
        sgd = {
            D = {learningRate = OPT.D_sgd_lr, momentum = OPT.D_sgd_momentum},
            G = {learningRate = OPT.G_sgd_lr, momentum = OPT.G_sgd_momentum}
        }
    }
end

if EPOCH == nil then
    EPOCH = 1
end
PLOT_DATA = {}
VIS_NOISE_INPUTS = NN_UTILS.createNoiseInputs(100)

-- Get examples to plot
function getSamples(ds, N)
  local N = N or 8
  local noiseInputs = torch.Tensor(N, NOISE_DIM[1], NOISE_DIM[2], NOISE_DIM[3])
  local condInputs = torch.Tensor(N, COND_DIM[1], COND_DIM[2], COND_DIM[3])
  local gt_diff = torch.Tensor(N, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
  local gt = torch.Tensor(N, 3, OPT.fineSize, OPT.fineSize)

  -- Generate samples
  noiseInputs:uniform(-1, 1)
  for n = 1,N do
    local rand = math.random(ds:size())
    local example = ds[rand]
    condInputs[n] = example.coarse:clone()
    gt[n] = example.fine:clone()
    gt_diff[n] = example.diff:clone()
  end
  local samples = MODEL_G:forward({noiseInputs, condInputs})
  --local preds_D = MODEL_D:forward({samples, condInputs})

  local to_plot = {}
  for i=1,N do
    local refined = torch.add(condInputs[i]:float(), samples[i]:float())
    to_plot[#to_plot+1] = condInputs[i]:float()
    to_plot[#to_plot+1] = gt[i]:float()
    to_plot[#to_plot+1] = refined
    to_plot[#to_plot+1] = gt_diff[i]:float()
    to_plot[#to_plot+1] = samples[i]:float()
  end
  return to_plot
end

VAL_DATA = DATASET.loadImages(0, 500)

-- training loop
while true do
    print('Loading new training data...')
    TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch, 500)
    
    -- plot errors
    if not OPT.noplot then
        local to_plot = getSamples(VAL_DATA, 20)
        disp.image(to_plot, {win=OPT.window, width=2*10*IMG_DIMENSIONS[3], title="Coarse, GT, G img, GT diff, G diff (" .. OPT.save .. " epoch " .. EPOCH .. ")"})
    end
    
    -- train/test
    ADVERSARIAL.train(TRAIN_DATA)
    --adversarial.test(valData, nval)

    ADVERSARIAL.approxParzen(VAL_DATA, 200, OPT.batchSize)
end
