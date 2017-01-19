--------------------------------------------------------------------------------
-- Training and testing of RNN/LSTM models
-- for detection of sequence: 'abba'
--
-- Written by : Abhishek Chaurasia,
-- Reviewed by: Alfredo Canziani
--------------------------------------------------------------------------------

require 'nngraph'
require 'optim'

torch.manualSeed(6)
torch.setdefaulttensortype('torch.FloatTensor')

-- Colorizing print statement for test results
local green     = '\27[32m'        -- Green
local red       = '\27[31m'        -- Red
local rc        = '\27[0m'         -- Default
local redH      = '\27[41m'        -- Red highlight
local underline = '\27[4m'         -- Underline

print(red .. underline .. '\ne-Lab RNN Training Script\n' .. rc)
--------------------------------------------------------------------------------
-- Command line options
--------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:option('-d',         2,       '# of neurons in a layer')
cmd:option('-nHL',       1,       '# of hidden layers')
cmd:option('-T',         4,       'Unrolling steps')
cmd:option('-S',         4,       'Length of sequence')
cmd:option('-trainSize', 10000,   '# of input sequence')
cmd:option('-testSize',  150,     '# of input sequence')
cmd:option('-mode',     'RNN',    'RNN type [RNN|GRU|FW]')
cmd:option('-lr',        2e-2,    'Learning rate')
cmd:option('-lrd',       0.95,    'Learning rate decay')
cmd:option('-ds',       'abba',   'Data set [abba|randomSeq]')
cmd:text()
-- To get better detection; increase # of nHL or d or both

local opt = cmd:parse(arg or {})
--------------------------------------------------------------------------------
-- Hyperparameter definitions
local n   = 2
local d   = opt.d
local nHL = opt.nHL
local K   = 2
local T   = opt.T
local S   = ds == 'abba' and 4 or opt.S
local trainSize = opt.trainSize
local testSize  = opt.testSize
local mode = opt.mode
local lr   = opt.lr
local lrd  = opt.lrd
local optimState = {learningRate = lr, alpha = lrd}

-- Local packages
local data = opt.ds == 'abba' and require 'data' or require 'longData'
local network = require 'network'

--------------------------------------------------------------------------------
-- x : Inputs => Dimension : trainSize x n
-- y : Labels => Dimension : trainSize
local x, y = data.getData(trainSize, S)

--------------------------------------------------------------------------------
-- Get the model which is unrolled in time
local model, prototype = network.getModel(n, d, nHL, K, T, mode)

local criterion = nn.ClassNLLCriterion()

local w, dE_dw = model:getParameters()
print('# of parameters in the model: ' .. w:nElement())

-- Default intial state set to Zero
local h0 = {}
local h = {}
for l = 1, nHL do
   table.insert(h0, torch.zeros(d))
   table.insert(h, h0[#h0])
   if mode == 'FW' then -- Add the fast weights matrices A (A1, A2, ..., AnHL)
      table.insert(h0, torch.zeros(d, d))
      table.insert(h, h0[#h0])
   end
end

print(green .. 'Training ' .. mode .. ' model' .. rc)

-- Saving the graphs with input dimension information
model:forward({x[{ {1, T}, {} }], table.unpack(h)})
prototype:forward({x[1], table.unpack(h)})

if not paths.dirp('graphs') then paths.mkdir('graphs') end
graph.dot(model.fg, 'model', 'graphs/model')
graph.dot(prototype.fg, 'prototype', 'graphs/prototype')

-- Converts the output table into a Tensor that can be processed by the Criterion
local function table2Tensor(s)
   local p = s[1]:view(1, 2)
   for t = 2, T do p =  p:cat(s[t]:view(1, 2), 1) end
   return p
end

--------------------------------------------------------------------------------
-- Converts input tensor into table of dimension equal to first dimension of input tensor
-- and adds padding of zeros, which in this case are states
local function tensor2Table(inputTensor, padding)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   for l = 1, padding do outputTable[l + inputTensor:size(1)] = h0[l] end
   return outputTable
end

--------------------------------------------------------------------------------------
-- Training
--------------------------------------------------------------------------------------
local timer = torch.Timer()
local trainError = 0

timer:reset()
for itr = 1, trainSize - T, T do
   local xSeq = x:narrow(1, itr, T)
   local ySeq = y:narrow(1, itr, T)

   local feval = function()
      --------------------------------------------------------------------------------
      -- Forward Pass
      --------------------------------------------------------------------------------
      model:training()       -- Model in training mode

      -- Input to the model is table of tables
      -- {x_seq, h1, h2, ..., h_nHL}
      local states = model:forward({xSeq, table.unpack(h)})
      -- States is the output returned from the selected models
      -- Contains predictions + tables of hidden layers
      -- {{y}, {h}}

      -- Store predictions
      local prediction = table2Tensor(states)

      local err = criterion:forward(prediction, ySeq)
      --------------------------------------------------------------------------------
      -- Backward Pass
      --------------------------------------------------------------------------------
      local dE_dh = criterion:backward(prediction, ySeq)

      -- convert dE_dh into table and assign Zero for states
      local m = mode == 'FW' and 2 or 1
      local dE_dhTable = tensor2Table(dE_dh, m*nHL)

      model:zeroGradParameters()
      model:backward({xSeq, table.unpack(h)}, dE_dhTable)

      -- Store final output states
      for l = 1, nHL do h[l] = states[l + T] end
      if mode == 'FW' then for l = nHL+1, 2*nHL do h[l] = states[l + T] end end

      return err, dE_dw
   end

   local err
   w, err = optim.rmsprop(feval, w, optimState)
   trainError = trainError + err[1]

   if ((itr - 1) / T) % math.floor((trainSize - T - 1)/100) == 0 then
      print(string.format("Iteration %8d, Training Error/seq_len = %4.4f, gradnorm = %4.4e",
                           itr, err[1] / T, dE_dw:norm()))
   end
end
trainError = trainError/trainSize
local trainTime = timer:time().real

-- Set model into evaluate mode
model:evaluate()
prototype:evaluate()

-- torch.save('prototype.net', prototype)
--------------------------------------------------------------------------------
-- Testing
--------------------------------------------------------------------------------
x, y = data.getData(testSize, S)

for l = 1, nHL do h[l] = h0[l] end  -- Reset the states
if mode == 'FW' then for l = nHL+1, 2*nHL do h[l] = h0[l] end end

local seqBuffer = {}                   -- Buffer to store previous input characters
local nPopTP = 0
local nPopFP = 0
local nPopFN = 0
local pointer = S
local style = rc

-- get the style and update count
local function getStyle(nPop, style, prevStyle)
   if nPop > 0 then
      nPop = nPop - 1
      io.write(style)
   else
      style = prevStyle
   end
   return nPop, style
end

local function test(t)
   local states = prototype:forward({x[t], table.unpack(h)})

   -- Final output states which will be used for next input sequence
   for l = 1, nHL do h[l] = states[l] end
   if mode == 'FW' then for l = nHL+1, 2*nHL do h[l] = states[l] end end

   -- Prediction which is of size 2
   local predIdx = nHL + 1
   if mode == 'FW' then predIdx = 2 * nHL + 1 end
   local prediction = states[predIdx]

   -- Mapping vector into character based on encoding used in data.lua
   local mappedCharacter = x[t][1] == 1 and 'a' or 'b'

   if t < S then                        -- Queue to store past 4 sequences
      seqBuffer[t] = mappedCharacter
   else

      -- Class 1: Sequence is NOT abba
      -- Class 2: Sequence IS abba
      local max, idx = torch.max(prediction, 1) -- Get the prediction mapped to class
      if idx[1] == y[t] then
         -- Change style to green when sequence is detected
         if y[t] == 2 then nPopTP = S end
      else
         -- In case of false prediction
         if y[t] == 2 then nPopFN = S else nPopFP = S end
      end

      local popLocation = pointer % S + 1

      -- When whole correct/incorrect sequence has been displayed with the given style;
      -- reset the style
      style = rc
      io.write(style)
      nPopTP, style = getStyle(nPopTP, green, style)
      nPopFP, style = getStyle(nPopFP, redH, style)
      nPopFN, style = getStyle(nPopFN, underline, style)

      -- Display the sequence with style
      io.write(style .. seqBuffer[popLocation])

      -- Previous element of the queue is replaced with the current element
      seqBuffer[pointer] = mappedCharacter
      -- Increament/Reset the pointer of queue
      pointer = popLocation
   end
end

print("\nNotation for output:")
print("====================")
print("+ " .. green     .. "True Positive" .. rc)
print("+ " .. rc        .. "True Negative" .. rc)
print("+ " .. redH      .. "False Positive" .. rc)
print("+ " .. underline .. "False Negative" .. rc)

timer:reset()
for i = 1, testSize do
   test(i)
end
local testTime = timer:time().real
print('\n')
print("Training data size: " .. trainSize)
print("Test     data size: " .. testSize)
print(string.format("\nTotal train time: %4.0f ms %s||%s Avg. train time: %3.0f us",
                   (trainTime*1000), red, rc, (trainTime*10^6/trainSize)))
print(string.format("Total test  time: %4.0f ms %s||%s Avg. test  time: %3.0f us",
                   (testTime*1000), red, rc, (testTime*10^6/testSize)))

