-- Eugenio Culurciello
-- September 2016
-- RNN training test: ABBA sequence detector
-- based on: https://github.com/Element-Research/rnn/blob/master/examples/sequence-to-one.lua

require 'nn'
require 'rnn'

-- torch.setdefaulttensortype('torch.FloatTensor')
-- nngraph.setDebug(true)

-- SET UP MODEL: --------------------------------------------------------------

-- model hyper-parameters 
batchSize = 1
rho = 4 -- sequence length
hiddenSize = 10
nIndex = 2 -- input words
nClass = 2 -- output classes
lr = 0.1


-- build simple recurrent neural network
r = nn.Recurrent(
   hiddenSize, nn.Identity(), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
   :add(nn.LookupTable(nIndex, hiddenSize))
   :add(nn.SplitTable(1,2))
   :add(nn.Sequencer(r))
   :add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
   :add(nn.Linear(hiddenSize, nClass))
   :add(nn.LogSoftMax())

-- build criterion
criterion = nn.ClassNLLCriterion()

-- build input
-- create a sequence of 2 numbers: {2, 1, 2, 2, 1, 1, 2, 2, 1 ..}
ds = {}
ds.size = 1000
local target_seq = torch.LongTensor({1,2,2,1})
ds.input = torch.LongTensor(ds.size,rho):random(nClass)
ds.target = torch.LongTensor(ds.size):fill(1)
-- initialize targets:
local indices = torch.LongTensor(rho)
for i=1, ds.size do
  if torch.sum(torch.abs(torch.add(ds.input[i], -target_seq))) == 0 then ds.target[i] = 2 end
end

indices:resize(batchSize)

-- training
local inputs, targets = torch.LongTensor(), torch.LongTensor()
for iteration = 1, 10000 do
   -- 1. create a sequence of rho time-steps
   indices:random(1,ds.size) -- choose some random samples
   inputs:index(ds.input, 1,indices)
   targets:index(ds.target, 1,indices)
   -- print(inputs, targets)
   
   -- 2. forward sequence through rnn
   rnn:zeroGradParameters() 
   
   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
   
   if iteration%10 == 0 then print(string.format("Iteration %d ; NLL err = %f ", iteration, err)) end

   -- 3. backward sequence through rnn (i.e. backprop through time)
   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = rnn:backward(inputs, gradOutputs)
   
   -- 4. update
   rnn:updateParameters(lr)
end

-- testing:
for iteration = 1, 10 do
  print('\n\n\n ITERATION:', iteration)
   indices:random(1,ds.size) -- choose some random samples
   inputs:index(ds.input, 1,indices)
   targets:index(ds.target, 1,indices)
   
   local outputs = rnn:forward(inputs)
   max,idx = torch.max(outputs, 2)
   print('inputs:', inputs, 'targets:', targets, 'results:', idx)
end



