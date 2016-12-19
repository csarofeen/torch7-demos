--------------------------------------------------------------------------------
-- Simple GRU block
--
-- Notations used from: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local GRU = {}

function GRU.getPrototype(n, d, nHL, K)
   local inputs = {}
   inputs[1] = nn.Identity()()               -- input X
   for j = 1, nHL + 1 do
      table.insert(inputs, nn.Identity()())  -- previous states h[j] + tensor of ones
   end

   local ONES = inputs[nHL + 2]

   local x, nIn
   local outputs = {}
   for j = 1, nHL do
      if j == 1 then
         x = inputs[j]
         nIn = n
      else
         x = outputs[j-1]
         nIn = d
      end

      local hPrev = inputs[j+1]

      local z = {x, hPrev} - nn.JoinTable(1)
                           - nn.Linear(nIn + d, d)
                           - nn.Sigmoid()

      local r = {x, hPrev} - nn.JoinTable(1)
                           - nn.Linear(nIn + d, d)
                           - nn.Sigmoid()

      local prodRH = {r, hPrev} - nn.CMulTable()
      local hTilda = {x, prodRH} - nn.JoinTable(1)
                                 - nn.Linear(nIn + d, d)
                                 - nn.Tanh()

      local subZ = {ONES, z} - nn.CSubTable()
      local prodZH = {subZ, hPrev} - nn.CMulTable()
      local prodZhTilda = {z, hTilda} - nn.CMulTable()

      local nextH = {prodZH, prodZhTilda} - nn.CAddTable()

      table.insert(outputs, nextH)
   end

   local logsoft = (outputs[#outputs] - nn.Linear(d, K) - nn.LogSoftMax())
                   :annotate{name = 'y\'[t]',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}
   table.insert(outputs, logsoft)

   -- Output is table with {h, prediction}
   return nn.gModule(inputs, outputs)
end

return GRU
