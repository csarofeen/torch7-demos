require 'nn'
require 'nngraph'
require 'optim'

local data = require 'data.lua'
local rnn = require 'RNN.lua'
torch.setdefaulttensortype('torch.FloatTensor')

-- Hyperparameter definitions
local batchSize = 1        -- # # of batches
local dictionarySize = 2   -- Sequence of 2 values
local L = 1                -- # of layers
local M = 10               -- # of neurons in a layer
local seqLength = 4        -- Length of sequence
local trainSize = 10000    -- # of input sequence
local gradClip = 5         -- clipping of gradParams

local lr = 2e-3
local lrd = 0.95

-- Colorizing print statement for testing results
local pf = function(...) print(string.format(...)) end
local r = sys.COLORS.red
local g = sys.COLORS.green
local nc = sys.COLORS.none

-- x : Inputs => Dimension : dictionarySize x trainSize
-- y : Labels => Dimension : 1 x trainSize
local x, y = data.getData(trainSize, seqLength)

local prototype = {}       -- Name space
prototype.model = rnn.getModel(dictionarySize, M, L)
prototype.criterion = nn.ClassNLLCriterion()

local params, gradParams = prototype.model:getParameters()
print('# of parameters in the model: ' .. params:nElement())

-- Clones of model unrolled in time:
local clones = {}
clones.model = {}
clones.criterion = {}
for name, proto in pairs(prototype) do
   for i = 1, seqLength do
      clones[name][i] = proto:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end
end

local function cloneTable(tensor_list)
   -- takes a table of tensors and returns a table of cloned tensors
   local out = {}
   for k,v in pairs(tensor_list) do
      out[k] = v:clone():zero()
   end
   return out
end

-- Default intial state set to Zero
local hDefault = {}
for j = 1, L do
   table.insert(hDefault, torch.zeros(batchSize, M))
end

local h0 = cloneTable(hDefault)
local bo = 0               -- Skip sequence based on size of sequence

local function feval()
   gradParams:zero()
   --------------------------------------------------------------------------------
   -- Forward Pass
   --------------------------------------------------------------------------------

   local h = {}               -- Tables of states of each layer for each sequence
   h[0] = h0
   local yHat = {}            -- Table with predictions for each sequence
   local loss = 0
   for t = 1, seqLength do
      clones.model[t]:training()       -- Model in training mode

      -- Input to the model is table of tables
      -- {x, h1, h2, ..., hL}
      local states = clones.model[t]:forward({x[{ {}, {t+bo} }]:t(), unpack(h[t-1])})
      -- States is the output returned from the RNN model
      -- Its a table containing tables of hidden layers and final prediction
      -- {h1, h2, ..., hL, y}

      h[t] = {}
      for i = 1, L do
         table.insert(h[t], states[i])
      end

      yHat[t] = states[L+1]
      loss = loss + clones.criterion[t]:forward(yHat[t], y[1][t+bo])
   end
   loss = loss / seqLength

   --------------------------------------------------------------------------------
   -- Backward Pass
   --------------------------------------------------------------------------------

   local delta = {}
   delta[seqLength] = cloneTable(hDefault)
   for t = seqLength, 1, -1 do
      local deltaL = clones.criterion[t]:backward(yHat[t], y[1][t+bo])
      -- print(deltaL)
      -- print(yHat[t])
      -- print(y[1][t+bo])
      -- io.read()
      table.insert(delta[t], deltaL)
      local dStates = clones.model[t]:backward({x[{ {}, {t+bo} }]:t(), unpack(h[t-1])}, delta[t])
      delta[t-1] = {}
      for k = 2, L+1 do
         delta[t-1][k-1] = dStates[k]:clone()
      end
   end

   h0 = h[seqLength]
   gradParams:clamp(-gradClip, gradClip)
   bo = bo + seqLength
   return loss, gradParams
end


--------------------------------------------------------------------------------
-- Training
--------------------------------------------------------------------------------
local losses = {}
local optimState = {learningRate = lr, alpha = lrd}
local iterations = trainSize/seqLength
for i = 1, iterations do
   local _, loss = optim.rmsprop(feval, params, optimState)
   losses[#losses + 1] = loss[1]

   if i % (iterations/10) == 0 then
      print(string.format("Iteration %8d, loss = %4.4f, loss/seq_len = %4.4f, gradnorm = %4.4e",
                           i, loss[1], loss[1] / seqLength, gradParams:norm()))
   end
end

-- Set model into evaluate mode
for i = 1, seqLength do
   clones.model[i]:evaluate()
end

--------------------------------------------------------------------------------
-- Testing
--------------------------------------------------------------------------------
bo = 0
function test()
   local h = {}
   h[0] = hDefault
   local yHat = {}
   local loss = 0

   for t= 1, seqLength do
      local states = clones.model[t]:forward({x[{ {}, {t+bo} }]:t(), unpack(h[t-1])})
      h[t] = {}
      for i = 1, L do
         table.insert(h[t], states[i])
      end
      yHat[t] = states[L+1]
   end
   h[0] = hDefault

   -- print("Sequence: ", x[1][1+bo], x[1][2+bo], x[1][3+bo], x[1][4+bo])
   local check = 0
   local color = nc
   local mappedCharacter = 'a'
   if x[1][4+bo] == 0 then
      mappedCharacter = 'b'
   else
      mappedCharacter = 'a'
   end

   if x[1][1+bo] == 1 and x[1][2+bo] == 0 and x[1][3+bo] == 0 and x[1][4+bo] == 1 then
      check = 1
   else
      check = 0
   end

   local max, idx = torch.max(yHat[seqLength], 2)
   if idx[1][1] == y[1][seqLength+bo] then
      if check == 1 then
         color = g
      end
   else
      color = r
   end
   io.write(color .. mappedCharacter .. nc)
   bo = bo + 1
end

print('\n')
for i = 1, 150 do
   test()
end
print('\n')
