--------------------------------------------------------------------------------
-- A simple building block for RNN/LSTMs
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local network = {}
nngraph.setDebug(true)
-- n   : # of inputs
-- d   : # of neurons in hidden layer
-- nHL : # of hidden layers
-- K   : # of output neurons

-- Links all the prototypes, given the # of sequences
function network.getModel(n, d, nHL, K, T, mode)
   local prototype
   if mode == 'RNN' then
      local RNN = require 'RNN'
      prototype = RNN.getPrototype(n, d, nHL, K)
   elseif mode == 'GRU' then
      local GRU = require 'GRU'
      prototype = GRU.getPrototype(n, d, nHL, K)
   else
      print("Invalid model type. Available options: (RNN/GRU)")
   end

   local clones = {}
   for i = 1, T do
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
             :annotate{name = 'h^('..l..')[0]',
              graphAttributes = {
              style = 'filled',
              fillcolor = 'lightpink'}}
   end

   if mode == 'GRU' then
      H0[nHL + 1] = nn.Identity()()
      H[nHL + 1] = H0[nHL + 1]
             :annotate{name = 'Ones'}
   end

   local splitInput = inputSequence - nn.SplitTable(1)

   for i = 1, T do
      local x = (splitInput - nn.SelectTable(i))
                :annotate{name = 'x['..i..']',
                 graphAttributes = {
                 style = 'filled',
                 fillcolor = 'moccasin'}}

      local tempStates = ({x, table.unpack(H)} - clones[i])
                         :annotate{name = mode .. '['..i..']',
                          graphAttributes = {
                          style = 'filled',
                          fillcolor = 'skyblue'}}

      outputs[i] = (tempStates - nn.SelectTable(nHL + 1))  -- Prediction
                   :annotate{name = 'y\'['..i..']',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}

      if i < T then
         for l = 1, nHL do                         -- State values passed to next sequence
            H[l] = (tempStates - nn.SelectTable(l))
                   :annotate{name = 'h^('..l..')['..i..']',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'lightpink'}}
            if mode == 'GRU' then
               H[nHL + 1] = H0[nHL + 1]
                            :annotate{name = 'Ones'}
            end
         end
      else
         for l = 1, nHL do                         -- State values passed to next sequence
            outputs[T + l] = (tempStates - nn.SelectTable(l))
                               :annotate{name = 'h^('..l..')['..i..']',
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

return network
