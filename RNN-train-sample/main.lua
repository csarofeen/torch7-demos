--------------------------------------------------------------------------------
-- Training and testing of RNN/LSTM models
-- for detection of sequence: 'abba'
--
-- Written by : Abhishek Chaurasia,
-- Reviewed by: Alfredo Canziani
--------------------------------------------------------------------------------


require 'optim'
require 'cunn'
require 'cutorch'
require 'cudnn'

--torch.manualSeed(6) remove determinism, would need to use cuda based seed

torch.setdefaulttensortype('torch.CudaTensor')

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
--cmd:option('-mode',     'RNN',    'RNN type [RNN|GRU|FW]') tanh based RNN was the only one I found to work This is hard coded below
cmd:option('-lr',        2e-2,    'Learning rate')
cmd:option('-lrd',       0.95,    'Learning rate decay')
cmd:option('-ds',       'randomSeq',   'Data set [abba|randomSeq]')
cmd:text()
-- To get better detection; increase # of nHL or d or both

local opt = cmd:parse(arg or {})
--------------------------------------------------------------------------------
-- Hyperparameter definitions
-- n   : # of inputs
-- d   : # of neurons in hidden layer
-- nHL : # of hidden layers
-- K   : # of output neurons
-- S   : Sequence length
-- T   : Unrolling step length; how many steps you want the model to examine concurrently

local n   = 2       --input feature size
local d   = opt.d
local nHL = opt.nHL
local K   = 2       --output feature size
local T   = opt.T
local S   = ds == 'abba' and 4 or opt.S
local trainSize = opt.trainSize
local testSize  = opt.testSize
local mode = "cudnn.RNNTanh"
local lr   = opt.lr
local lrd  = opt.lrd
local optimState = {learningRate = lr, alpha = lrd}

-- Local packages
local data = opt.ds == 'abba' and require 'data' or require 'longData'

--------------------------------------------------------------------------------
-- x : Inputs => Dimension : trainSize x n
-- y : Labels => Dimension : trainSize
local x, y = data.getData(trainSize, S)

--------------------------------------------------------------------------------

model = nn.Sequential()
model:add(nn.View(1, -1, 2) )
model:add(cudnn.RNNTanh(n, d, nHL, true, 0, true))
model:add(nn.View(-1, d))
model:add(nn.Linear(d, K):cuda())
model:add(nn.LogSoftMax():cuda())

local criterion = nn.ClassNLLCriterion()
criterion:cuda()

local w, dE_dw = model:getParameters()
print('# of parameters in the model: ' .. w:nElement())

print(green .. 'Training ' .. mode .. ' model' .. rc)

model:forward(x[{ {1, T}, {} }])

--------------------------------------------------------------------------------------
-- Training
--------------------------------------------------------------------------------------
local timer = torch.Timer()
local trainError = 0

timer:reset()
--loop through sequence 1 at a time
for itr = 1, trainSize - T, T do
   local xSeq = x:narrow(1, itr, T)
   local ySeq = y:narrow(1, itr, T)

   local feval = function()
      model:training()
      model:zeroGradParameters()
      prediction = model:forward(xSeq)
      local err = criterion:forward(prediction, ySeq)
      local dE_dh = criterion:backward(prediction, ySeq)
      model:backward(xSeq, criterion.gradInput)
      return criterion.output, dE_dw
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

--------------------------------------------------------------------------------
-- Testing
--------------------------------------------------------------------------------
x, y = data.getData(testSize, S)

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
   -- local states = prototype:forward({x[t], table.unpack(h)})
   prediction = model:forward(x[t])

   local mappedCharacter = x[t][1] == 1 and 'a' or 'b'


   if t < S then                        -- Queue to store past 4 sequences
      seqBuffer[t] = mappedCharacter
   else

      -- Class 1: Sequence is NOT abba
      -- Class 2: Sequence IS abba
      local max, idx = torch.max(prediction, 2) -- Get the prediction mapped to class
      idx = idx[1][1] --idx comes out a little different now.

      if idx == y[t] then
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
