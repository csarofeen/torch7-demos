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
      local nextH = ({x, hPrev} - nn.JoinTable(2) - nn.Linear(n+M, M))
                    :annotate{name = 'Hidden layer: ' .. tostring(j),
                     graphAttributes = {
                     style = 'filled',
                     fillcolor = 'skyblue'}}

      table.insert(outputs, nextH)
   end

   local logsoft = (outputs[#outputs] - nn.Linear(M, N) - nn.LogSoftMax())
                   :annotate{name = 'Prediction',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'springgreen'}}
   table.insert(outputs, logsoft)

   return nn.gModule(inputs, outputs)
end
return RNN
