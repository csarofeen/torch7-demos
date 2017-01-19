local data = {}
local key
local debug = false

function data.getData(trainSize, seqLength)
   -- Get key
   if not key then
      key = torch.Tensor(seqLength):random(2)
      io.write('Key generated: ' .. sys.COLORS.blue)
      for _, k in ipairs(key:totable()) do
         io.write(k == 1 and 'a' or 'b')
      end
      io.write(sys.COLORS.none .. '\n')
   end

   -- Generate random sequence of 1s and 2s
   local s = torch.Tensor(trainSize):random(2)

   -- Labels for training
   local y = torch.ones(trainSize)

   -- Injects key for ~1/3 of cases
   local injections = math.floor(trainSize / seqLength / 3)
   assert(injections > 0,
          'trainSize should be at least 3 times longer than seqLength')
   for inj = 1, injections do
      local idx = math.random(trainSize - seqLength + 1)
      s:narrow(1, idx, seqLength):copy(key)
   end

   for i = 1, trainSize - seqLength + 1 do
      if s:narrow(1, i, seqLength):eq(key):prod() == 1 then
         y[i+seqLength-1] = 2
      end
   end

   if debug then
      -- Debugging
      local none = sys.COLORS.none
      local red = sys.COLORS.red
      for i = 1, trainSize do
         local modifier
         if i > trainSize - seqLength + 1 or y[i+seqLength-1] == 1 then
            io.write(none)
         else
            io.write(red)
         end
         io.write(s[i] == 1 and 'a' or 'b')
      end
      io.write('\n')
      for _, k in ipairs(y:totable()) do
         io.write(k == 1 and ' ' or '|')
      end
      io.write('\n')
   end

   -- Mapping of input from R trainSize to trainSize x 2
   -- a => 1 => <1, 0>
   -- b => 2 => <0, 1>

   local x = torch.zeros(trainSize, 2)
   for i = 1, trainSize do
      if s[i] == 1 then
         x[i][1] = 1
      else
         x[i][2] = 1
      end
   end
   return x, y
end

function data.getKey()
   assert(key ~= nil, 'Key not yet generated')
   return key
end

function data.toggleDebug()
   debug = not debug
   print('Debug is ' .. (debug and 'ON' or 'OFF'))
end

return data
