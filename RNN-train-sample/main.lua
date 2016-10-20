--------------------------------------------------------------------------------
-- Training and testing of RNN model
-- for detection of sequence: 'abba'
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

require 'nngraph'
require 'optim'

torch.manualSeed(6)

local data = require 'data'
local rnn = require 'RNN'
torch.setdefaulttensortype('torch.FloatTensor')

-- Hyperparameter definitions
local n = 2                -- Sequence of 2 values
local d = 2                -- # of neurons in a layer
local nHL = 1              -- # of hidden layers
local K = 2                -- # of classes
local T = 4                -- Length of sequence
local trainSize = 10000    -- # of input sequence
local testSize = 150       -- # of input sequence
-- To get better detection; increase # of nHL or d or both

local lr = 2e-2
local lrd = 0.95
local optimState = {learningRate = lr, alpha = lrd}

-- Colorizing print statement for test results
local truePositive  = '\27[32m'
local trueNegative  = '\27[0m'
local falsePositive = '\27[41m'
local falseNegative = '\27[4m'

-- x : Inputs => Dimension : trainSize x n
-- y : Labels => Dimension : trainSize
local x, y = data.getData(trainSize, T)

-- Get the model which is unrolled in time
local model, prototype = rnn.getModel(n, d, nHL, K, T)

local criterion = nn.ClassNLLCriterion()

local w, dE_dw = model:getParameters()
print('# of parameters in the model: ' .. w:nElement())

-- Default intial state set to Zero
local h0 = {}
local h = {}
for l = 1, nHL do
   h0[l] = torch.zeros(d)
   h[l] = h0[l]:clone()
end

-- Saving the graphs with input dimension information
model:forward({x[{ {1, 4}, {} }], table.unpack(h)})
prototype:forward({x[1], table.unpack(h)})

graph.dot(model.fg, 'Whole model', 'Whole model')
graph.dot(prototype.fg, 'RNN model', 'RNN model')

-- Converts the output table into a Tensor that can be processed by the Criterion
local function table2Tensor(s)
   local p = s[1]:view(1, 2)
   for t = 2, T do
      p =  p:cat(s[t]:view(1, 2), 1)
   end
   return p
end

-- Converts input tensor into table of dimension equal to first dimension of input tensor
-- and adds padding of zeros, which in this case are states
local function tensor2Table(inputTensor, padding)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do
      outputTable[t] = inputTensor[t]
   end

   for l = 1, padding do
      outputTable[l + inputTensor:size(1)] = h0[l]:clone()
   end
   return outputTable
end

--------------------------------------------------------------------------------------
-- Training
--------------------------------------------------------------------------------------
local trainError = 0

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
      -- States is the output returned from the RNN model
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
      local dE_dhTable = tensor2Table(dE_dh, nHL)

      model:zeroGradParameters()
      model:backward({xSeq, table.unpack(h)}, dE_dhTable)

      -- Store final output states
      for l = 1, nHL do
         h[l] = states[l + T]
      end

      return err, dE_dw
   end

   local err
   w, err = optim.rmsprop(feval, w, optimState)
   trainError = trainError + err[1]

   if itr % (trainSize/(10*T)) == 1 then
      print(string.format("Iteration %8d, Training Error/seq_len = %4.4f, gradnorm = %4.4e",
                           itr, err[1] / T, dE_dw:norm()))
   end
end
trainError = trainError/trainSize

-- Set model into evaluate mode
model:evaluate()
prototype:evaluate()

-- torch.save('model.net', model)
-- torch.save('prototype.net', prototype)
--------------------------------------------------------------------------------
-- Testing
--------------------------------------------------------------------------------

local x, y = data.getData(trainSize, T)

for l = 1, nHL do
   h[l] = h0[l]:clone()                -- Reset the states
end

local seqBuffer = {}                   -- Buffer to store previous input characters
local nPopTP = 0
local nPopFP = 0
local nPopFN = 0
local pointer = 4
local style = trueNegative

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
   for l = 1, nHL do
      h[l] = states[l]
   end
   -- Prediction which is of size 2
   local prediction = states[nHL + 1]

   local mappedCharacter = 'a'
   -- Mapping vector into character based on encoding used in data.lua
   if x[t][1] == 0 then
      mappedCharacter = 'b'
   end

   if t < T then                        -- Queue to store past 4 sequences
      seqBuffer[t] = mappedCharacter
   else

      -- Class 1: Sequence is NOT abba
      -- Class 2: Sequence IS abba
      local max, idx = torch.max(prediction, 1) -- Get the prediction mapped to class
      if idx[1] == y[t] then
         -- Change style to green when sequence is detected
         if y[t] == 2 then
            nPopTP = T
         end
      else
         -- In case of false prediction
         if y[t] == 2 then
            nPopFN = T
         else
            nPopFP = T
         end
      end

      local popLocation = pointer + 1
      if popLocation == T + 1 then      -- Next pointer will be 1
         popLocation = 1
      end

      -- When whole correct/incorrect sequence has been displayed with the given style;
      -- reset the style
      style = trueNegative
      io.write(style)
      nPopTP, style = getStyle(nPopTP, truePositive, style)
      nPopFP, style = getStyle(nPopFP, falsePositive, style)
      nPopFN, style = getStyle(nPopFN, falseNegative, style)

      -- Display the sequence with style
      io.write(style .. seqBuffer[popLocation])

      -- Previous element of the queue is replaced with the current element
      seqBuffer[pointer] = mappedCharacter
      -- Increament/Reset the pointer of queue
      if pointer == T then
         pointer = 1
      else
         pointer = pointer + 1
      end
   end
end

print("\nNotation for output:")
print("====================")
print("+ " .. truePositive ..  "True Positive" .. trueNegative)
print("+ " .. trueNegative ..  "True Negative" .. trueNegative)
print("+ " .. falsePositive .. "False Positive" .. trueNegative)
print("+ " .. falseNegative .. "False Negative" .. trueNegative)
print("\n")

for i = 1, testSize do
   test(i)
end
print('\n')
