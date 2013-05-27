--------------------------------------------------------------------------------
-- Sparse stereo correspondence with Torch7
--------------------------------------------------------------------------------
-- Alfredo Canziani, May 2013
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'camera'
require 'nnx'
require 'edgeDetector'
require 'pl'

-- Parsing the command line ----------------------------------------------------
print '==> Processing options'
opt = lapp [[
--bgTh                  (default .3)      Background filtering [0, 2.5]
--kSize                 (default 5)       Edge kernel size {3,5}
--showInterestPoints                      Show interest points
]]

-- Parameters ------------------------------------------------------------------
width  = 160 --800
height = 120 --600
fps = 30
corrWindowSize = 9  -- Correlation Window Size, MUST BE AN ODD NUMBER!!!
-- dir = "demo_test"
local neighborhood = image.gaussian1D(25)
local normalisation = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3)
-- sys.execute(string.format('mkdir -p %s',dir))

---------------------------------------------------------------------------------
-- In this program camera1 is supposed to serve as the RIGHT camera,
-- whereas camera2 shall match the LEFT camera. The LEFT and RIGHT cameras
-- provide respectively the RIGHT- and LEFT-shifted images

-- iCameraX[c]: {i}mage from {Camera} {X} [{c}olour version; greyscale otherwise]
---------------------------------------------------------------------------------
camera1 = image.Camera{idx=1,width=width,height=height,fps=fps}
camera2 = image.Camera{idx=2,width=width,height=height,fps=fps}
---------------------------------------------------------------------------------

-- Some reference code ----------------------------------------------------------
-- module = nn.SpatialConvolutionMM(1,64,9,9)
-- print(module)
-- print(#module.weight)
-- module.bias = torch.zeros(module.bias)
-- img = image.rgb2y(image.lena())
-- module(img)
---------------------------------------------------------------------------------

-- f = 1
-- maxEdge = 0
-- minEdge = math.huge

while true do
   sys.tic()
   iCameraR = image.rgb2y(camera1:forward())
   iCameraL = image.rgb2y(camera2:forward())
   edgesR = image.scale(edgeDetector(iCameraL,opt.kSize),width,height)[1]
   edgesL = image.scale(edgeDetector(iCameraR,opt.kSize),width,height)[1]

   -- if we'd like to see the interest points
   if opt.showInterestPoints then
      a = torch.Tensor(3,height,width)
      b = torch.Tensor(3,height,width)
      a[{ 1,{},{} }] = iCameraR
      a[{ 2,{},{} }] = iCameraR
      a[{ 3,{},{} }] = iCameraR
      b[{ 1,{},{} }] = iCameraL
      b[{ 2,{},{} }] = iCameraL
      b[{ 3,{},{} }] = iCameraL
   end

   for i = 1, height, height/8 do
      for j = 1, width, width/8 do
         -- print('i = ' .. i .. ', j = ' .. j)
         max1,y = torch.max(edgesL[{ {i, i-1+height/8}, {j, j-1+width/8} }],1)
         max,x = torch.max(max1,2)
         x = x[1][1]
         y = y[1][x]

         -- adding constrains to <x> and <y>
         x = (x < math.ceil(corrWindowSize/2)) and math.ceil(corrWindowSize/2) or (x > width/8  - math.ceil(corrWindowSize/2)) and width/8  - math.ceil(corrWindowSize/2) or x
         y = (y < math.ceil(corrWindowSize/2)) and math.ceil(corrWindowSize/2) or (x > height/8 - math.ceil(corrWindowSize/2)) and height/8 - math.ceil(corrWindowSize/2) or x

         -- if we'd like to see the interest points
         if opt.showInterestPoints then
            if max[1][1] > opt.bgTh then
               a[{ {},y+i,x+j }] = 0
               a[{ 1,y+i,x+j }] = 1
            else
               a[{ {},y+i,x+j }] = 0
               a[{ 2,y+i,x+j }] = 1
            end
         end

         -- if max[1][1] > maxEdge then
         --    maxEdge = max[1][1]
         -- end
         -- min = (edgesL[{ {i, i+height/8}, {j, j+width/8} }]):abs():min()
         -- if min < minEdge then
         --    minEdge = min
         -- end
      end
   end

   -- edgesL = edgesL:cmul(edgesL:gt(2):float())
   -- if we'd like to see the interest points
   if opt.showInterestPoints then
   win = image.display{win=win,image={a,b}, legend="FPS: ".. 1/sys.toc(), min=0, max=1, zoom=4}
   end
   win = image.display{win=win,image={iCameraR}, min=0, max=1, legend=string.format('Right camera, FPS: %g', 1/sys.toc()), zoom=4}
   win3 = image.display{win=win3,image={normalisation(iCameraR)}, legend=string.format('Right camera SCN, FPS: %g', 1/sys.toc()), zoom=4}
   -- win = image.display{win=win,image={iCameraR,iCameraL}, legend="FPS: ".. 1/sys.toc(), min=0, max=1,nrow=2}
   -- win2 = image.display{win=win2,image={edgesL,edgesR}, legend="FPS: ".. 1/sys.toc(), min = -12, max = 12,  zoom=4}
   -- image.savePNG(string.format("%s/frame_1_%05d.png",dir,f),a1)
   -- if f == 100 then
   --    f = 1
   --    io.write('Threashold bgTh: ')
   --    opt.bgTh = tonumber(io.read())
   -- end
   -- f = f + 1
   -- print("FPS: ".. 1/sys.toc()) 
end
