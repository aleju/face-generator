require 'torch'
require 'optim'
require 'pl'
require 'paths'
require 'image'

local adversarial = {}

-- training function
function adversarial.train(trainData)
    EPOCH = EPOCH or 1
    local N_epoch = OPT.N_epoch
    if N_epoch <= 0 then
        N_epoch = trainData:size()
    end
    local dataBatchSize = OPT.batchSize / 2
    local time = sys.clock()

    local inputs = torch.Tensor(OPT.batchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    local targets = torch.Tensor(OPT.batchSize)
    local noiseInputs = torch.Tensor(OPT.batchSize, NOISE_DIM[1], NOISE_DIM[2], NOISE_DIM[3])
    local condInputs = torch.Tensor(OPT.batchSize, COND_DIM[1], COND_DIM[2], COND_DIM[3])

    -- do one epoch
    local batchIdx = 0
    print(string.format("<trainer> Epoch #%d [batchSize = %d]", EPOCH, OPT.batchSize))
    
    for t = 1,N_epoch,dataBatchSize do 
        -- size of this batch, will usually be dataBatchSize but can be lower at the end
        local thisBatchSize = math.min(OPT.batchSize, N_epoch - t + 1)
        
        -- this script currently can't handle small sized batches
        if thisBatchSize < 4 then
            print(string.format("[INFO] skipping batch at t=%d, because its size is less than 4", t))
            break
        end
        
        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of discriminator
        local fevalD = function(x)
            collectgarbage()
            if x ~= parameters_D then -- get new parameters
                PARAMETERS_D:copy(x)
            end

            GRAD_PARAMETERS_D:zero() -- reset gradients

            --  forward pass
            local outputs = MODEL_D:forward({inputs, condInputs})
            local f = CRITERION:forward(outputs, targets)

            -- backward pass 
            local df_do = CRITERION:backward(outputs, targets)
            MODEL_D:backward({inputs, condInputs}, df_do)

            -- penalties (L1 and L2):
            if OPT.D_L1 ~= 0 or OPT.D_L2 ~= 0 then
                -- Loss:
                f = f + OPT.D_L1 * torch.norm(PARAMETERS_D, 1)
                f = f + OPT.D_L2 * torch.norm(PARAMETERS_D, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_D:add(torch.sign(PARAMETERS_D):mul(OPT.D_L1) + PARAMETERS_D:clone():mul(OPT.D_L2) )
            end
            
            -- update confusion (add 1 since targets are binary)
            for i = 1,thisBatchSize do
                local c
                if outputs[i][1] > 0.5 then c = 2 else c = 1 end
                CONFUSION:add(c, targets[i]+1)
            end

            -- Clamp D's gradients
            -- This helps a bit against D suddenly giving up (only outputting y=1 or y=0)
            if OPT.D_clamp ~= 0 then
                GRAD_PARAMETERS_D:clamp((-1)*OPT.D_clamp, OPT.D_clamp)
            end

            return f,GRAD_PARAMETERS_D
        end

        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of generator 
        local fevalG_on_D = function(x)
            collectgarbage()
            if x ~= PARAMETERS_G then -- get new parameters
                PARAMETERS_G:copy(x)
            end

            GRAD_PARAMETERS_G:zero() -- reset gradients

            -- forward pass
            local samples = MODEL_G:forward({noiseInputs, condInputs})
            local outputs = MODEL_D:forward({samples, condInputs})
            local f = CRITERION:forward(outputs, targets)

            --  backward pass
            local df_samples = CRITERION:backward(outputs, targets)
            MODEL_D:backward({samples, condInputs}, df_samples)
            local df_do = MODEL_D.gradInput[1]
            MODEL_G:backward({noiseInputs, condInputs}, df_do)

            -- penalties (L1 and L2):
            if OPT.G_L1 ~= 0 or OPT.G_L2 ~= 0 then
                -- Loss:
                f = f + OPT.G_L1 * torch.norm(PARAMETERS_G, 1)
                f = f + OPT.G_L2 * torch.norm(PARAMETERS_G, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_G:add(torch.sign(PARAMETERS_G):mul(OPT.G_L2) + PARAMETERS_G:clone():mul(OPT.G_L2))
            end
            
            if OPT.G_clamp ~= 0 then
                GRAD_PARAMETERS_G:clamp((-1)*OPT.G_clamp, OPT.G_clamp)
            end

            return f,GRAD_PARAMETERS_G
        end

        ----------------------------------------------------------------------
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        -- Get half a minibatch of real, half fake
        for k=1, OPT.D_iterations do
            -- (1.1) Real data 
            local inputIdx = 1
            local realDataSize = thisBatchSize / 2
            for i = 1, realDataSize do
                local randomIdx = math.random(trainData:size())
                local trainingExample = trainData[randomIdx]
                inputs[inputIdx] = trainingExample.diff:clone()
                condInputs[inputIdx] = trainingExample.coarse:clone()
                targets[inputIdx] = Y_NOT_GENERATOR
                inputIdx = inputIdx + 1
            end
            
            -- (1.2) Sampled data
            noiseInputs:uniform(-1, 1)
            for i = 1, realDataSize do
                local randomIdx = math.random(trainData:size())
                local trainingExample = trainData[randomIdx]
                condInputs[inputIdx] = trainingExample.coarse:clone()
                inputIdx = inputIdx + 1
            end
            inputIdx = inputIdx - realDataSize
            
            local generatedDiff = MODEL_G:forward({
                                    noiseInputs[{{realDataSize+1,2*realDataSize}}],
                                    condInputs[{{realDataSize+1,2*realDataSize}}]
                                  })
            for i = 1, realDataSize do
                inputs[inputIdx] = generatedDiff[i]:clone()
                targets[inputIdx] = Y_GENERATOR
                inputIdx = inputIdx + 1
            end

            if OPT.D_optmethod == "sgd" then
                optim.sgd(fevalD, PARAMETERS_D, OPTSTATE.sgd.D)
            elseif OPT.D_optmethod == "adagrad" then
                optim.adagrad(fevalD, PARAMETERS_D, OPTSTATE.adagrad.D)
            elseif OPT.D_optmethod == "adam" then
                optim.adam(fevalD, PARAMETERS_D, OPTSTATE.adam.D)
            else
                print("[Warning] Unknown optimizer method chosen for D.")
            end
        end -- end for K

        ----------------------------------------------------------------------
        -- (2) Update G network: maximize log(D(G(z)))
        for k=1, OPT.G_iterations do
            noiseInputs:uniform(-1, 1)
            targets:fill(Y_NOT_GENERATOR)
            for i=1, thisBatchSize do
                local randomIdx = math.random(trainData:size())
                local trainingExample = trainData[randomIdx]
                condInputs[i] = trainingExample.coarse:clone()
            end
            
            --optim.sgd(fevalG_on_D, parameters_G, OPTSTATE.sgd.G)
            --optim.adagrad(fevalG_on_D, parameters_G, ADAGRAD_STATE_G)
            if OPT.G_optmethod == "sgd" then
                optim.sgd(fevalG_on_D, PARAMETERS_G, OPTSTATE.sgd.G)
            elseif OPT.G_optmethod == "adagrad" then
                optim.adagrad(fevalG_on_D, PARAMETERS_G, OPTSTATE.adagrad.G)
            elseif OPT.G_optmethod == "adam" then
                optim.adam(fevalG_on_D, PARAMETERS_G, OPTSTATE.adam.G)
            else
                print("[Warning] Unknown optimizer method chosen for G.")
            end
        end

        batchIdx = batchIdx + 1
        -- display progress
        xlua.progress(t+thisBatchSize, N_epoch)
    end -- end for loop over dataset
    
    -- time taken
    time = sys.clock() - time
    print(string.format("<trainer> time required for this epoch = %d s", time))
    print(string.format("<trainer> time to learn 1 sample = %f ms", 1000 * time/N_epoch))

    -- print confusion matrix
    print("Confusion of D:")
    print(CONFUSION)
    local tV = CONFUSION.totalValid
    CONFUSION:zero()

    -- save/log current net
    if EPOCH % OPT.saveFreq == 0 then
        local filename = paths.concat(OPT.save, string.format('adversarial_c2f_%d_to_%d.net', OPT.coarseSize, OPT.fineSize))
        os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
        if paths.filep(filename) then
            os.execute(string.format("mv %s %s.old", filename, filename))
        end
        print(string.format("<trainer> saving network to %s", filename))
        
        NN_UTILS.prepareNetworkForSave(MODEL_G)
        NN_UTILS.prepareNetworkForSave(MODEL_D)
        torch.save(filename, {D = MODEL_D, G = MODEL_G, opt = OPT, epoch=EPOCH})
    end

    -- next epoch
    EPOCH = EPOCH + 1

    return tV
end

-- test function
--[[
function adversarial.test(dataset, N)
  local time = sys.clock()
  local N = N or dataset:size()

  local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local targets = torch.Tensor(opt.batchSize)
  local noise_inputs 
  if type(opt.noiseDim) == 'number' then
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)
  else
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  end
  local cond_inputs 
  if type(opt.condDim) == 'number' then
    cond_inputs = torch.Tensor(opt.batchSize, opt.condDim)
  else
    cond_inputs = torch.Tensor(opt.batchSize, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  end

  print('\n<trainer> on testing set:')
  for t = 1,N,opt.batchSize do
    -- display progress
    xlua.progress(t, N)

    ----------------------------------------------------------------------
    -- (1) Real data
    local targets = torch.ones(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      inputs[k] = sample[1]:clone()
      cond_inputs[k] = sample[3]:clone()
      k = k + 1
    end
    local preds = model_D:forward({inputs, cond_inputs}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end

    ----------------------------------------------------------------------
    -- (2) Generated data (don't need this really, since no 'validation' generations)
    noise_inputs:uniform(-1, 1)
    local c = 1
    for i = 1,opt.batchSize do
      sample = dataset[math.random(dataset:size())]
      cond_inputs[i] = sample[3]:clone()
    end
    local samples = model_G:forward({noise_inputs, cond_inputs})
    local targets = torch.zeros(opt.batchSize)
    local preds = model_D:forward({samples, cond_inputs}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()

  return cond_inputs
end
--]]

-- Unnormalized parzen window type estimate (used to track performance during training)
-- Really just a nearest neighbours of ground truth to multiple generations
function adversarial.approxParzen(ds, nsamples, nneighbors)
  best_dist = best_dist or 1e10
  print('<trainer> evaluating approximate parzen ')
  local noiseInputs = torch.Tensor(nneighbors, NOISE_DIM[1], NOISE_DIM[2], NOISE_DIM[3])
  local condInputs = torch.Tensor(nneighbors, COND_DIM[1], COND_DIM[2], COND_DIM[3])
  local distances = torch.Tensor(nsamples)
  for n = 1,nsamples do
    xlua.progress(n, nsamples)
    local example = ds[math.random(ds:size())]
    local condInput = example.coarse
    local fine = example.fine
    noiseInputs:uniform(-1, 1)
    for i = 1,nneighbors do
      condInputs[i] = condInput:clone() 
    end
    neighbors = MODEL_G:forward({noiseInputs, condInputs})
    neighbors:add(condInputs)
    -- compute distance
    local dist = 1e10
    for i = 1,nneighbors do
      dist = math.min(torch.dist(neighbors[i], fine), dist)
    end
    distances[n] = dist
  end
  print('average || x_' .. OPT.fineSize .. ' - G(x_' .. OPT.coarseSize .. ') || = ' .. distances:mean()) 

  -- save/log current net
  if distances:mean() < best_dist then 
    best_dist = distances:mean()

    local filename = paths.concat(OPT.save, string.format('adversarial_c2f_%d_to_%d.bestnet', OPT.coarseSize, OPT.fineSize))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D = MODEL_D, G = MODEL_G, opt = OPT})
  end
  return distances
end

return adversarial
