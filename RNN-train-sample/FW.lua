--------------------------------------------------------------------------------
-- Simple FW block
--
-- Notations used from: Using Fast Weights to Attend to the Recent Past
-- URL: https://arxiv.org/abs/1610.06258
--
-- Written by: Alfredo Canziani, Jan 17
--------------------------------------------------------------------------------

local FW = {}

local function normalise(d)
   local input = nn.Identity()()
   local mean = input - nn.Mean() - nn.Replicate(d)
   local centred = {input, mean} - nn.CSubTable()
   local std = centred - nn.Square() - nn.Mean() - nn.Sqrt()
   - nn.Replicate(d) - nn.AddConstant(1e-5)
   local output = {centred, std} - nn.CDivTable()
   return nn.gModule({input},{output})
end

function FW.getPrototype(n, d, nHL, K, opt)

   local opt = opt or {}
   local l = opt.lambda or 0.9
   local e = opt.eta or 0.5
   local S = opt.S or 1
   local LayerNorm = opt.LayerNorm or true

   local inputs = {}
   inputs[1] = nn.Identity()()               -- input X
   for j = 1, 2 * nHL do
      table.insert(inputs, nn.Identity()())  -- previous states h[j] and A
   end

   local x, nIn
   local outputs = {}
   for j = 1, nHL do
      if j == 1 then
         x = inputs[j]:annotate{name = 'x[t]',
            graphAttributes = { style = 'filled', fillcolor = 'moccasin'}
         }

         nIn = n
      else
         x = outputs[2 * j - 3]
         nIn = d
      end

      local hPrev = inputs[2*j]:annotate{name = 'h^('..j..')[t-1]',
         graphAttributes = { style = 'filled', fillcolor = 'lightpink'}
      }

      local A = inputs[2*j + 1]:annotate{name = 'A^('..j..')[t-1]',
         graphAttributes = { style = 'filled', fillcolor = 'plum'}
      }


      local hPrevVect = hPrev - nn.View(-1, 1)
      local lambda_A = (A - nn.MulConstant(l)):annotate{name = 'lambda_A'}
      local eta_hhT = ({
         hPrevVect - nn.Identity(), hPrevVect - nn.Identity()
      } - nn.MM(false, true) - nn.MulConstant(e)):annotate{name = 'eta_hhT'}
      A = {lambda_A, eta_hhT} - nn.CAddTable()
      A:annotate{name = 'A^('..j..')[t]',
         graphAttributes = { style = 'filled', fillcolor = 'paleturquoise'}
      }

      local dot = {x, hPrev} - nn.JoinTable(1)
                             - nn.Linear(nIn + d, d)
      dot:annotate{name = 'dot'}

      local hs = (dot - nn.ReLU()):annotate{name = 'h_' .. 0,
         graphAttributes = { style = 'filled', fillcolor = 'tan'}
      }
      for s = 1, S do
         hs = {dot, {A, hs - nn.View(-1, 1)} - nn.MM()} - nn.CAddTable()
         if LayerNorm then
            hs = (hs - normalise(d)):annotate{name = 'LayerNorm',
               graphAttributes = { style = 'filled', fillcolor = 'lavender'}
            }
            end
         hs = (hs - nn.ReLU()):annotate{name = 'h_' .. s,
            graphAttributes = { style = 'filled', fillcolor = 'tan'}
         }
      end

      hs:annotate{name = 'h^('..j..')[t]',
         graphAttributes = { style = 'filled', fillcolor = 'skyblue'}
      }


      table.insert(outputs, hs)
      table.insert(outputs, A)
   end

   local logsoft = (outputs[#outputs - 1] - nn.Linear(d, K) - nn.LogSoftMax())
                   :annotate{name = 'y\'[t]', graphAttributes = {
                    style = 'filled', fillcolor = 'seagreen1'}}
   table.insert(outputs, logsoft)

   -- Output is table with {h, prediction}
   return nn.gModule(inputs, outputs)
end

return FW
