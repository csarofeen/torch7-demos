-- Eugenio Culurciello
-- September 2016
-- RNN training test: ABBA sequence detector

require 'nn'
require 'nngraph'
require 'optim'
require 'RNN' -- https://raw.githubusercontent.com/karpathy/char-rnn/master/model/RNN.lua
-- local c = require 'trepl.colorize'

torch.setdefaulttensortype('torch.FloatTensor')
-- nngraph.setDebug(true)

local dictionary_size = 2 -- sequence of 2 symbols
local seq_length = 1000 -- sequence length
local nT = 4 -- RNN time steps

print('Creating Input...')
-- create a sequence of 2 numbers: {2, 1, 2, 2, 1, 1, 2, 2, 1 ..}
local s = torch.ceil(torch.rand(seq_length):add(0.5))
-- print('Inputs sequence:', s)
local y = torch.ones(1,seq_length)
for i = 4, seq_length do -- if you find sequence ...1001... then output is class '2', otherwise is '1'
   if (s[{i-3}]==2 and s[{i-2}]==1 and s[{i-1}]==1 and s[{i}]==2) then y[{1,{i}}] = 2 end
end
-- print('Desired output sequence:', y)
local x = torch.zeros(2,seq_length) -- create input with 1-hot encoding:
for i = 1, seq_length do
   if y[{1,{i}}] == 2 then x[{{},{i}}] = torch.ones(2) end
end
-- print('Input vector:', x)

-- model:
print('Creating Model...')
local rnn_size = 1
local rnn_layers = 1
local batch_size = 1
local RNNmodel = RNN(dictionary_size, rnn_size, rnn_layers, 0) -- input = 2 (classes), 1 layer, rnn_size=1, no dropout 
-- print('Test of RNN output:', RNNmodel:forward{ torch.Tensor(2), torch.Tensor(1) })

local params, grad_params 
params, grad_params = RNNmodel:getParameters()
print('Number of parameters in the model: ' .. params:nElement())
-- print(params, grad_params)

-- the initial state of the cell/hidden states
local init_state = {}
for L=1,rnn_layers do
  local h_init = torch.zeros(batch_size, rnn_size)
  table.insert(init_state, h_init:clone())
end

-- create clones of model to unroll in time:
local clones = {}
clones.rnn = {}
clones.criterion = {}
for i=1,nT do
   clones.rnn[i] = RNNmodel:clone('weight', 'gradWeights', 'bias')
   clones.criterion[i] = nn.ClassNLLCriterion()
end

function clone_list(tensor_list, zero_too)
  -- takes a list of tensors and returns a list of cloned tensors
  local out = {}
  for k,v in pairs(tensor_list) do
      out[k] = v:clone()
      if zero_too then out[k]:zero() end
  end
  return out
end


-- training function:
local init_state_global = clone_list(init_state)
local bo = 0 -- batch counter
function feval(p)
  if p ~= params then
    params:copy(p)
  end
  grad_params:zero()

  local predictions = {}
  local rnn_state = {[0]=init_state_global} -- initial state
  local loss = 0

  -- forward pass ---------------------------------------------------------------
  -- bo variable creates batches on the fly
  for t = 1, nT do
    clones.rnn[t]:training() -- make sure we are in correct training mode
    local  lst = clones.rnn[t]:forward{x[{{},{t+bo}}]:t(), unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    predictions[t] = lst[#lst]
    loss = loss + clones.criterion[t]:forward(predictions[t], y[{1,{t+bo}}])
  end
  loss = loss / nT 

  -- backward pass --------------------------------------------------------------
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local drnn_state = {[nT] = clone_list(init_state, true)} -- true also zeros the clones
  for t=nT,1,-1 do
    -- print(drnn_state)
    -- backprop through loss, and softmax/linear
    -- print(predictions[t], y[{1,{t+bo}}])
    local doutput_t = clones.criterion[t]:backward(predictions[t], y[{1,{t+bo}}])
    -- print('douptut', doutput_t)
    table.insert(drnn_state[t], doutput_t)
    local dlst = clones.rnn[t]:backward({x[{{},{t+bo}}]:t(), unpack(rnn_state[t-1])}, drnn_state[t])
    drnn_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k > 1 then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the 
        -- derivatives of the state, starting at index 2. I know...
        drnn_state[t-1][k-1] = v
      end
    end
  end
  -- transfer final state to initial state (BPTT)
  init_state_global = rnn_state[#rnn_state]

  -- create next batch:
  bo = bo + 1

  grad_params:div(nT)
  -- clip gradient element-wise
  grad_params:clamp(-5, 5)

  -- print(params,grad_params)

  return loss, grad_params
end


-- training:
print('Training...')
local losses = {}
local optim_state = {learningRate = 1e-1}
local iterations = seq_length-nT
for i = 1, iterations do
    local _, loss = optim.rmsprop(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % (seq_length/10) == 0 then
        print(string.format("Iteration %8d, loss = %4.4f, loss/seq_len = %4.4f, gradnorm = %4.4e", i, loss[1], loss[1] / nT, grad_params:norm()))
    end
end

