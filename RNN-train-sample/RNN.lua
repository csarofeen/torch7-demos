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

      local i2h = nn.Linear(n, M)(x)
      local h2h = nn.Linear(M, M)(inputs[j+1])
      local nextH = nn.Tanh()(nn.CAddTable()({i2h, h2h}))

      table.insert(outputs, nextH)
   end

   local zL_1 = outputs[#outputs]

   local proj = nn.Linear(M, N)(zL_1)
   local logsoft = nn.LogSoftMax()(proj)
   table.insert(outputs, logsoft)

   return nn.gModule(inputs, outputs)
end
return RNN
