-- Eugenio Culurciello
-- September 2016
-- RNN training test: ABBA / 1221 sequence detector
-- based on: https://raw.githubusercontent.com/karpathy/char-rnn/master/model/RNN.lua

require 'nn'
require 'nngraph'
require 'optim'
dofile('RNN.lua')
local model_utils = require 'model_utils'

torch.setdefaulttensortype('torch.FloatTensor')
-- nngraph.setDebug(true)

local opt = {}
opt.dictionary_size = 2 -- sequence of 2 symbols
opt.train_size = 10000 -- train data size
opt.seq_length = 4 -- RNN time steps

print('Creating Input...')
-- create a sequence of 2 numbers: {2, 1, 2, 2, 1, 1, 2, 2, 1 ..}
local s = torch.Tensor(opt.train_size):random(2)
print('Inputs sequence:', s:view(1,-1))
local y = torch.ones(1, opt.train_size)
for i = 4, opt.train_size do -- if you find sequence ...1001... then output is class '2', otherwise is '1'
   if (s[{i-3}]==1 and s[{i-2}]==2 and s[{i-1}]==2 and s[{i}]==1) then y[{1,{i}}] = 2 end
   -- if (s[{i-1}]==1 and s[{i}]==2) then y[{1,{i}}] = 2 end -- detect one 1-2 sequence
end
print('Desired sequence:', y)
local x = torch.zeros(2, opt.train_size) -- create input with 1-hot encoding:
for i = 1, opt.train_size do
  if s[i] > 1 then x[{{},{i}}] = torch.Tensor{0,1} else x[{{},{i}}] = torch.Tensor{1,0} end
end
print('Input vector:', x)

-- model:
print('Creating Model...')
opt.rnn_size = 10
opt.rnn_layers = 1
opt.batch_size = 1

local protos = {} 
protos.rnn = RNN(opt.dictionary_size, opt.rnn_size, opt.rnn_layers, 0) -- input = 2 (classes), 1 layer, rnn_size=1, no dropout 
protos.criterion = nn.ClassNLLCriterion()
-- print('Test of RNN output:', RNNmodel:forward{ torch.Tensor(2), torch.Tensor(1) })

-- the initial state of the cell/hidden states
local init_state = {}
for L = 1, opt.rnn_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  table.insert(init_state, h_init:clone())
end

local params, grad_params 
-- get flattened parameters tensor
-- params, grad_params = model_utils.combine_all_parameters(protos.rnn)
params, grad_params = protos.rnn:getParameters()
print('Number of parameters in the model: ' .. params:nElement())
-- print(params, grad_params)

-- create clones of model to unroll in time:
print('Cloning RNN model:')
local clones = {}
clones.rnn = {}
clones.criterion = {}
for name, proto in pairs(protos) do
  print('cloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
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
local bo = 0 -- batch offset / counter
opt.grad_clip = 5
function feval(p)
  if p ~= params then
    params:copy(p)
  end
  grad_params:zero()

  -- bo variable creates batches on the fly
  
  -- forward pass ---------------------------------------------------------------
  local rnn_state = {[0]=init_state_global} -- initial state
  local predictions = {}
  local loss = 0
  for t = 1, opt.seq_length do
    clones.rnn[t]:training() -- make sure we are in correct training mode
    -- print( 'sample,target:', x[{{},{t+bo}}], y[{1,{t+bo}}] )
    local lst = clones.rnn[t]:forward{x[{{},{t+bo}}]:t(), unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    predictions[t] = lst[#lst]
    loss = loss + clones.criterion[t]:forward(predictions[t], y[{1,{t+bo}}])
  end
  loss = loss / opt.seq_length
  -- backward pass --------------------------------------------------------------
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
  for t = opt.seq_length, 1, -1 do
    -- print(drnn_state)
    -- backprop through loss, and softmax/linear
    -- print( 'prediction,target:', predictions[t], y[{1,{t+bo}}] )
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
  -- grad_params:div(opt.seq_length)
  -- clip gradient element-wise
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  -- print(params,grad_params)
  -- point to next batch:
  bo = bo + opt.seq_length
  return loss, grad_params
end


-- training:
opt.learning_rate = 2e-3
opt.decay_rate = 0.95
print('Training...')
local losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.train_size/opt.seq_length
for i = 1, iterations do
  local _, loss = optim.rmsprop(feval, params, optim_state)
  losses[#losses + 1] = loss[1]

  if i % (iterations/10) == 0 then
    print(string.format("Iteration %8d, loss = %4.4f, loss/seq_len = %4.4f, gradnorm = %4.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
  end
end


-- testing function:
bo = 0
function test()
  -- bo variable creates batches on the fly
  
  -- forward pass ---------------------------------------------------------------
  local rnn_state = {[0]=init_state} -- initial state
  local predictions = {}
  local loss = 0
  for t = 1, opt.seq_length do
    clones.rnn[t]:evaluate() -- make sure we are in correct testing mode
    local  lst = clones.rnn[t]:forward{x[{{},{t+bo}}]:t(), unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    predictions[t] = lst[#lst]
  end
  -- carry over lstm state
  rnn_state[0] = init_state -- here we want to test all steps
  -- print(predictions[opt.seq_length])
  -- print results:
  local max, idx
  max,idx = torch.max( predictions[opt.seq_length], 2)
  print('Prediction:', idx[1][1], 'Label:', y[{1,{opt.seq_length+bo}}][1])
  -- point to next batch:
  bo = bo + 1 -- here we want to test all steps
end

-- and test!
opt.test_samples = 50
for i = 1, opt.test_samples do
  test()
end
