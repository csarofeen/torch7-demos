--------------------------------------------------------------------------------
-- Return rnn block
--------------------------------------------------------------------------------

local RNN = {}

--[[
                  +-----------+
                  |           |
                  |   +----+  |
                  V   |    |--+
            +--->(+)->| h1 |
            |         |    |------+
            |         +----+      |
            |                     |
            |                     |
            |                     |
            |     +-----------+   |
            |     |           |   |
      +---+ |     |   +----+  |   |     +---+
      |   | |     V   |    |--+   +---->|   |
      | x +-+--->(+)->| h2 |   +------->| y |
      |   | |         |    |---+  +---->|   |
      +---+ |         +----+      |     +---+
            |                     |
            |                     |
            |                     |
            |     +-----------+   |
            |     |           |   |
            |     |   +----+  |   |
            |     V   |    |--+   |
            +--->(+)->| h3 |      |
                      |    |------+
                      +----+

--]]

-- N : # of inputs
-- M : # of neurons in hidden layer
-- L : # of hidden layers

function RNN.getModel(N, M, L)
   local inputs = {}
   table.insert(inputs, nn.Identity()())       -- input X
   for j = 1, L do
      table.insert(inputs, nn.Identity()())    -- previous states h[j]
   end

   local x, n
   local outputs = {}
   for j = 1, L do
      if j == 1 then
         x = inputs[j]
         n = N
      else
         x = outputs[j-1]
         n = M
      end

      local nextH = {x, inputs[j+1]} - nn.JoinTable(2) - nn.Linear(n+M, M)

      table.insert(outputs, nextH)
   end

   local zL_1 = outputs[#outputs]

   local proj = zL_1 - nn.Linear(M, N)
   local logsoft = nn.LogSoftMax()(proj)
   table.insert(outputs, logsoft)

   return nn.gModule(inputs, outputs)
end
return RNN
