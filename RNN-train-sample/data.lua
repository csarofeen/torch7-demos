local data = {}

function data.getData(trainSize, seqLength)
   -- Generate random sequence of 1s and 2s
   local s = torch.Tensor(trainSize):random(2)

   -- Labels for training
   local y = torch.ones(1, trainSize)

   for i = seqLength, trainSize do
      if s[i-3] == 1 and s[i-2] == 2 and s[i-1] == 2 and s[i] == 1 then
         y[1][i] = 2
      end
   end

   -- Mapping of input from R 1xtrainSize to 2xtrainSize
   -- 1 => <1, 0>
   -- 2 => <0, 1>

   local x = torch.zeros(2, trainSize)
   for i = 1, trainSize do
      if s[i] == 1 then
         x[{{}, {i}}] = torch.Tensor({1, 0})
      else
         x[{{}, {i}}] = torch.Tensor({0, 0})
      end
   end
   return x, y
end

return data
