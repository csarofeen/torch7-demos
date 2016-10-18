--------------------------------------------------------------------------------
-- A simple RNN block
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local RNN = {}

--[[
                    +-----------+
                    |           |
                    |   +----+  |
                    V   |    |--+
              +--->(+)->| h1 |
              |x1       |    |------+
              |         +----+      |
              |                     |
              |                     |
              |                     |
              |     +-----------+   |
              |     |           |   |
      +-----+ |     |   +----+  |   |     +---+
      |   1 | |     V   |    |--+   +---->|   |
      | x:2 +-+--->(+)->| h2 |   +------->| y |
      |   3 | |x2       |    |---+  +---->|   |
      +-----+ |         +----+      |     +---+
              |                     |
              |                     |
              |                     |
              |     +-----------+   |
              |     |           |   |
              |     |   +----+  |   |
              |     V   |    |--+   |
              +--->(+)->| h3 |      |
               x3       |    |------+
                        +----+

--]]

-- N : # of inputs
-- M : # of neurons in hidden layer
-- nHL : # of hidden layers

-- Returns a simple RNN model
local function getPrototype(N, M, nHL, K)
   local inputs = {}
   table.insert(inputs, nn.Identity()())       -- input X
   for j = 1, nHL do
      table.insert(inputs, nn.Identity()())    -- previous states h[j]
   end

   local x, n
   local outputs = {}
   for j = 1, nHL do
      if j == 1 then
         x = inputs[j]:annotate{name = 'Input',
                       graphAttributes = {
                       style = 'filled',
                       fillcolor = 'moccasin'}}
         n = N
      else
         x = outputs[j-1]
         n = M
      end

      local hPrev = inputs[j+1]:annotate{name = 'Previous state',
                                graphAttributes = {
                                style = 'filled',
                                fillcolor = 'lightpink'}}

      -- Concat input with previous state
      local nextH = ({x, hPrev} - nn.JoinTable(1) - nn.Linear(n+M, M))
                    :annotate{name = 'Hidden layer: ' .. tostring(j),
                     graphAttributes = {
                     style = 'filled',
                     fillcolor = 'skyblue'}}

      table.insert(outputs, nextH)
   end

   local logsoft = (outputs[#outputs] - nn.Linear(M, K) - nn.LogSoftMax())
                   :annotate{name = 'Prediction',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}
   table.insert(outputs, logsoft)

   -- Output is table with {h, prediction}
   return nn.gModule(inputs, outputs)
end

-- Links all the RNN models, given the # of sequences
function RNN.getModel(N, M, nHL, K, seq)
   local prototype = getPrototype(N, M, nHL, K)

   local clones = {}
   for i = 1, seq do
      clones[i] = prototype:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end

   local inputSequence = nn.Identity()()        -- Input sequence
   local H0 = {}                                -- Initial states of hidden layers
   local H = {}                                 -- Intermediate states
   local outputs = {}

   -- Linking initial states to intermediate states
   for l = 1, nHL do
      H0[l] = nn.Identity()()
      H[l] = H0[l]
             :annotate{name = 'Previous state layer ' .. tostring(l),
              graphAttributes = {
              style = 'filled',
              fillcolor = 'lightpink'}}
   end

   local splitInput = inputSequence - nn.SplitTable(1)

   for i = 1, seq do
      local x = (splitInput - nn.SelectTable(i))
                :annotate{name = 'Input ' .. tostring(i),
                 graphAttributes = {
                 style = 'filled',
                 fillcolor = 'moccasin'}}

      local tempStates = ({x, table.unpack(H)} - clones[i])
                         :annotate{name = 'RNN Model ' .. tostring(i),
                          graphAttributes = {
                          style = 'filled',
                          fillcolor = 'skyblue'}}

      outputs[i] = (tempStates - nn.SelectTable(nHL + 1))  -- Prediction
                   :annotate{name = 'Prediction',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}

      if i < seq then
         for l = 1, nHL do                         -- State values passed to next sequence
            H[l] = (tempStates - nn.SelectTable(l))
                   :annotate{name = 'Previous state layer ' .. tostring(l),
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'lightpink'}}
         end
      else
         for l = 1, nHL do                         -- State values passed to next sequence
            outputs[seq + l] = (tempStates - nn.SelectTable(l))
                               :annotate{name = 'State of last sequence, layer ' .. tostring(l),
                                graphAttributes = {
                                style = 'filled',
                                fillcolor = 'lightpink'}}
         end
      end
   end

   -- Output is table of {Predictions, Hidden states of last sequence}
   local g = nn.gModule({inputSequence, table.unpack(H0)}, outputs)

   return g, clones[1]
end

return RNN
