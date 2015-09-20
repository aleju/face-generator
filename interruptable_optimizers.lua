-- The following optimizers are exactly identical to the optimizers in the optim package, 
-- except that the ones here can be stopped from within the called opfunc().
-- If opfunc() returns false instead of gradients, the optimizers will not perform a learning
-- step. This can be useful in the GAN architecture to prevent one side from outracing the other.

-- Adagrad
function interruptableAdagrad(opfunc, x, config, state)
    -- (0) get/update state
    if config == nil and state == nil then
      print('no state table, ADAGRAD initializing')
    end
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-3
    local lrd = config.learningRateDecay or 0
    state.evalCounter = state.evalCounter or 0
    local nevals = state.evalCounter

    -- (1) evaluate f(x) and df/dx
    local fx,dfdx = opfunc(x)
    
    -------------------
    -- this was changed
    if fx == false then
        return false
    end
    -------------------

    -- (3) learning rate decay (annealing)
    local clr = lr / (1 + nevals*lrd)
      
    -- (4) parameter update with single or individual learning rates
    if not state.paramVariance then
        state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx)
    end
    state.paramVariance:addcmul(1,dfdx,dfdx)
    state.paramStd:resizeAs(state.paramVariance):copy(state.paramVariance):sqrt()
    x:addcdiv(-clr, dfdx,state.paramStd:add(1e-10))

    -- (5) update evaluation counter
    state.evalCounter = state.evalCounter + 1

    -- return x*, f(x) before optimization
    return x,{fx}
end

-- Adam
function interruptableAdam(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    
    -------------------
    -- this was changed
    if fx == false then
        return false
    end
    -------------------

    -- Initialization
    state.t = state.t or 0
    -- Exponential moving average of gradient values
    state.m = state.m or x.new(dfdx:size()):zero()
    -- Exponential moving average of squared gradient values
    state.v = state.v or x.new(dfdx:size()):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.denom = state.denom or x.new(dfdx:size()):zero()

    state.t = state.t + 1
    
    -- Decay the first and second moment running average coefficient
    state.m:mul(beta1):add(1-beta1, dfdx)
    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

    state.denom:copy(state.v):sqrt():add(epsilon)

    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
    -- (2) update x
    x:addcdiv(-stepSize, state.m, state.denom)

    -- return x*, f(x) before optimization
    return x, {fx}
end

-- SGD
function interruptableSgd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   local wds = config.weightDecays
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)
   
   -------------------
   -- this was changed
   if fx == false then
      return false
   end
   -------------------

   -- (2) weight decay with single or individual parameters
   if wd ~= 0 then
      dfdx:add(wd, x)
   elseif wds then
      if not state.decayParameters then
         state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.decayParameters:copy(wds):cmul(x)
      dfdx:add(state.decayParameters)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
   
   -- (5) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr, state.deltaParameters)
   else
      x:add(-clr, dfdx)
   end

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
