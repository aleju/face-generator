require 'torch'
require 'nn'
--require 'cunn'
require 'optim'
require 'pl'
require 'interruptable_optimizers'

local adversarial = {}
adversarial.accs = {}
function adversarial.mean(t)
    local sum = 0
    local count = 0

    for k,v in pairs(t) do
        if type(v) == 'number' then
            sum = sum + v
            count = count + 1
        end
    end

    return (sum / count)
end

-- training function
function adversarial.train(dataset, maxAccuracyD, accsInterval)
    EPOCH = EPOCH or 1
    --local N = N or dataset:size()
    local N = dataset:size()
    local dataBatchSize = OPT.batchSize / 2
    local time = sys.clock()
    local lastAccuracyD = 0.0
    local doTrainD = true
    local countTrainedD = 0
    local countNotTrainedD = 0
    local count_lr_increased_D = 0
    local count_lr_decreased_D = 0

    -- do one epoch
    print(string.format("<trainer> Epoch #%d [batchSize = %d]", EPOCH, OPT.batchSize))
    for t = 1,N,dataBatchSize do
        local thisBatchSize = math.min(OPT.batchSize, N - t + 1)
        local inputs = torch.Tensor(thisBatchSize, OPT.geometry[1], OPT.geometry[2], OPT.geometry[3])
        local targets = torch.Tensor(thisBatchSize)
        local noiseInputs = torch.Tensor(thisBatchSize, OPT.noiseDim)
        if thisBatchSize < 4 then
            print(string.format("[INFO] skipping batch at t=%d, because its size is less than 2", thisBatchSize))
            break
        end

        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of discriminator
        local fevalD = function(x)
            collectgarbage()
            local confusion_batch_D = optim.ConfusionMatrix(CLASSES)
            confusion_batch_D:zero()

            if x ~= PARAMETERS_D then -- get new parameters
                PARAMETERS_D:copy(x)
            end

            GRAD_PARAMETERS_D:zero() -- reset gradients

            --  forward pass
            local outputs = MODEL_D:forward(inputs)
            local f = CRITERION:forward(outputs, targets)

            -- backward pass 
            local df_do = CRITERION:backward(outputs, targets)
            MODEL_D:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if OPT.D_L1 ~= 0 or OPT.D_L2 ~= 0 then
                -- Loss:
                f = f + OPT.D_L1 * torch.norm(PARAMETERS_D ,1)
                f = f + OPT.D_L2 * torch.norm(PARAMETERS_D, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_D:add(torch.sign(PARAMETERS_D):mul(OPT.D_L1) + PARAMETERS_D:clone():mul(OPT.D_L2) )
            end

            -- update confusion (add 1 since targets are binary)
            for i = 1,thisBatchSize do
                local c
                if outputs[i][1] > 0.5 then c = 2 else c = 1 end
                CONFUSION:add(c, targets[i]+1)
                confusion_batch_D:add(c, targets[i]+1)
                --print("outputs[i][1]:" .. (outputs[i][1]) .. ", c: " .. c ..", targets[i]+1:" .. (targets[i]+1))
            end

            -- Optimize weights of D for this batch?
            confusion_batch_D:updateValids()
            local tV = confusion_batch_D.totalValid
            
            -- this keeps the accuracy of D at around 80%
            --[[
            if EPOCH > 1 then
                if tV > 0.95 then
                    OPTSTATE.adam.D.learningRate = OPTSTATE.adam.D.learningRate * 0.98
                    count_lr_decreased_D = count_lr_decreased_D + 1
                    --if EPOCH > 1 then
                    --    OPTSTATE.adam.G.learningRate = OPTSTATE.adam.G.learningRate * 1.001
                    --end
                elseif tV < 0.90 then
                    OPTSTATE.adam.D.learningRate = OPTSTATE.adam.D.learningRate * 1.02
                    count_lr_increased_D = count_lr_increased_D + 1
                    --if EPOCH > 1 then
                    --    OPTSTATE.adam.G.learningRate = OPTSTATE.adam.G.learningRate * 0.999
                    --end
                else
                    OPTSTATE.adam.D.learningRate = OPTSTATE.adam.D.learningRate * 1.001
                end
            end
            
            OPTSTATE.adam.D.learningRate = math.max(0.00001, OPTSTATE.adam.D.learningRate)
            OPTSTATE.adam.D.learningRate = math.min(0.001, OPTSTATE.adam.D.learningRate)
            --]]
            
            adversarial.accs[#adversarial.accs+1] = tV
            if #adversarial.accs > accsInterval then
                table.remove(adversarial.accs, 1)
            end
            local accAvg = adversarial.mean(adversarial.accs)
            doTrainD = (accAvg < maxAccuracyD)
            lastAccuracyD = tV
            if doTrainD then
                countTrainedD = countTrainedD + 1
                return f,GRAD_PARAMETERS_D
            else
                countNotTrainedD = countNotTrainedD + 1
                return false,false
            end
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
            local samples
            local samplesAE
            if MODEL_AE then
                samplesAE = MODEL_AE:forward(noiseInputs)
                samples = MODEL_G:forward(samplesAE)
            else
                samples = createImagesFromNoise(noiseInputs, false, true)
            end
            local outputs = MODEL_D:forward(samples)
            local f = CRITERION:forward(outputs, targets)

            --  backward pass
            local df_samples = CRITERION:backward(outputs, targets)
            MODEL_D:backward(samples, df_samples)
            local df_do = MODEL_D.modules[1].gradInput
            if MODEL_AE then
                MODEL_G:backward(samplesAE, df_do)
            else
                MODEL_G:backward(noiseInputs, df_do)
            end

            -- penalties (L1 and L2):
            if OPT.G_L1 ~= 0 or OPT.G_L2 ~= 0 then
                -- Loss:
                f = f + OPT.G_L1 * torch.norm(PARAMETERS_G, 1)
                f = f + OPT.G_L2 * torch.norm(PARAMETERS_G, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_G:add(torch.sign(PARAMETERS_G):mul(OPT.G_L2) + PARAMETERS_G:clone():mul(OPT.G_L2))
            end

            return f,GRAD_PARAMETERS_G
        end
        ------------------- end of eval functions ---------------------------

        ----------------------------------------------------------------------
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        -- Get half a minibatch of real, half fake
        for k=1, OPT.D_iterations do
            -- (1.1) Real data 
            local inputIdx = 1
            local realDataSize = thisBatchSize / 2
            for i = 1, realDataSize do
                local idx = math.random(dataset:size())
                local sample = dataset[idx]
                --inputs[k] = sample[1]:clone()
                inputs[inputIdx] = sample:clone()
                inputIdx = inputIdx + 1
            end
            targets[{{1,realDataSize}}]:fill(Y_NOT_GENERATOR)

            -- (1.2) Sampled data
            local samples = createImages(realDataSize, false)
            --noiseInputs:normal(0.0, 0.35)
            --local samples = MODEL_AE:forward(noiseInputs[{{dataBatchSize+1,opt.batchSize}}])
            --samples = MODEL_G:forward(samples)
            for i = 1, realDataSize do
                --print("t=" .. t .. ", k=" .. k .. ", i=" .. i .. ", inputIdx=" .. inputIdx .. ", #samples=" .. (samples:size(1) .. ", thisBatchSize=" .. thisBatchSize))
                samples[i]:clone()
                inputs[inputIdx] = samples[i]:clone()
                inputIdx = inputIdx + 1
            end
            targets[{{realDataSize+1,thisBatchSize}}]:fill(Y_GENERATOR)
            
            --interruptableSgd(fevalD, PARAMETERS_D, OPTSTATE.sgd.D)
            --optim.adagrad(fevalD, parameters_D, ADAGRAD_STATE_D)
            --interruptableAdagrad(fevalD, PARAMETERS_D, OPTSTATE.adagrad.D)
            interruptableAdam(fevalD, PARAMETERS_D, OPTSTATE.adam.D)
            --optim.rmsprop(fevalD, PARAMETERS_D, OPTSTATE.rmsprop.D)
        end -- end for K

        ----------------------------------------------------------------------
        -- (2) Update G network: maximize log(D(G(z)))
        for k=1, OPT.G_iterations do
            noiseInputs = createNoiseInputs(noiseInputs:size(1))
            targets:fill(Y_NOT_GENERATOR)
            
            --optim.sgd(fevalG_on_D, parameters_G, OPTSTATE.sgd.G)
            --optim.adagrad(fevalG_on_D, parameters_G, ADAGRAD_STATE_G)
            --interruptableAdagrad(fevalG_on_D, PARAMETERS_G, OPTSTATE.adagrad.G)
            interruptableAdam(fevalG_on_D, PARAMETERS_G, OPTSTATE.adam.G)
            --optim.rmsprop(fevalG_on_D, PARAMETERS_G, OPTSTATE.rmsprop.G)
        end

        -- display progress
        xlua.progress(t, dataset:size())
    end -- end for loop over dataset

    -- fill out progress bar completely,
    -- for some reason that doesn't happen in the previous loop
    -- probably because it progresses to t instead of t+dataBatchSize
    xlua.progress(dataset:size(), dataset:size())

    -- time taken
    time = sys.clock() - time
    print(string.format("<trainer> time required for this epoch = %d s", time))
    print(string.format("<trainer> time to learn 1 sample = %f ms", 1000 * time/dataset:size()))
    print(string.format("<trainer> trained D %d of %d times.", countTrainedD, countTrainedD + countNotTrainedD))
    --print(string.format("<trainer> adam learning rate D:%.5f | G:%.5f", OPTSTATE.adam.D.learningRate, OPTSTATE.adam.G.learningRate))
    print(string.format("<trainer> adam learning rate D increased:%d decreased:%d", count_lr_increased_D, count_lr_decreased_D))

    -- print confusion matrix
    print("Confusion of normal D:")
    print(CONFUSION)
    local tV = CONFUSION.totalValid
    trainLogger:add{['% mean class accuracy (train set)'] = tV * 100}
    CONFUSION:zero()

    -- save/log current net
    if EPOCH % OPT.saveFreq == 0 then
        local filename = paths.concat(OPT.save, 'adversarial.net')
        os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
        if paths.filep(filename) then
            os.execute(string.format("mv %s %s.old", filename, filename))
        end
        print(string.format("<trainer> saving network to %s", filename))
        torch.save(filename, {D = MODEL_D, G = MODEL_G, opt = OPT, optstate = OPTSTATE})
    end

    -- next epoch
    EPOCH = EPOCH + 1

    return tV
end

return adversarial
