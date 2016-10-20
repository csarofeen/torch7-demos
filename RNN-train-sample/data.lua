local data = {}

function data.getData(trainSize, seqLength)
   -- Generate random sequence of 1s and 2s
   local s = torch.Tensor(trainSize):random(2)

   -- Labels for training
   local y = torch.ones(trainSize)

   for i = seqLength, trainSize do
      if s[i-3] == 1 and s[i-2] == 2 and s[i-1] == 2 and s[i] == 1 then
         y[i] = 2
      end
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

return data
