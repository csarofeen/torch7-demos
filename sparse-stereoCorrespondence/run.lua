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
--bgTh    (default .3)     Background filtering [0, 2.5]
--kSize   (default 5)       Edge kernel size {3,5}
]]

-- Parameters ------------------------------------------------------------------
width  = 160 --320 --1600
height = 120 --240 --896
fps = 30
-- dir = "demo_test"
local neighborhood = image.gaussian1D(21)
local normalisation = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3)


-- sys.execute(string.format('mkdir -p %s',dir))

camera1 = image.Camera{idx=1,width=width,height=height,fps=fps}
camera2 = image.Camera{idx=2,width=width,height=height,fps=fps}

---------------------------------------------------------------------------------
-- In this program camera1 is supposed to serve as the RIGHT camera,
-- whereas camera2 shall match the LEFT camera. The LEFT and RIGHT cameras
-- provide respectively the RIGHT- and LEFT-shifted images

-- iCameraX[c]: {i}mage from {Camera} {X} [{c}olour version; greyscale otherwise]
---------------------------------------------------------------------------------

iCameraRc = camera1:forward() -- acquiring image from the RIGHT camera
iCameraLc = camera2:forward() -- acquiring image from the LEFT camera

-- converting in B&W

iCameraR = image.rgb2y(iCameraRc)
iCameraL = image.rgb2y(iCameraLc)

-- f = 1
-- maxEdge = 0
-- minEdge = math.huge

while true do
   sys.tic()
   iCameraR = image.rgb2y(camera1:forward())
   iCameraL = image.rgb2y(camera2:forward())
   edgesR = image.scale(edgeDetector(iCameraL,opt.kSize),width,height)[1]
   edgesL = image.scale(edgeDetector(iCameraR,opt.kSize),width,height)[1]
   a = torch.Tensor(3,height,width)
   b = torch.Tensor(3,height,width)
   a[{ 1,{},{} }] = iCameraR
   a[{ 2,{},{} }] = iCameraR
   a[{ 3,{},{} }] = iCameraR
   b[{ 1,{},{} }] = iCameraL
   b[{ 2,{},{} }] = iCameraL
   b[{ 3,{},{} }] = iCameraL
   for i = 1, height-height/8, height/8 do
      for j = 1, width-width/8, width/8 do
         -- print('i = ' .. i .. ', j = ' .. j)
         max1,y = torch.max(edgesL[{ {i, i+height/8}, {j, j+width/8} }],1)
         max,x = torch.max(max1,2)
         x = x[1][1]
         y = y[1][x]
         if max[1][1] > opt.bgTh then
            a[{ {},y+i,x+j }] = 0
            a[{ 1,y+i,x+j }] = 1
         else
            a[{ {},y+i,x+j }] = 0
            a[{ 2,y+i,x+j }] = 1
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
   win = image.display{win=win,image={a,b}, legend="FPS: ".. 1/sys.toc(), min=0, max=1, zoom=4}
   -- win = image.display{win=win,image={iCameraR,iCameraL}, legend="FPS: ".. 1/sys.toc(), min=0, max=1,nrow=2}
   win2 = image.display{win=win2,image={edgesL,edgesR}, legend="FPS: ".. 1/sys.toc(), min = -12, max = 12,  zoom=4}
   -- image.savePNG(string.format("%s/frame_1_%05d.png",dir,f),a1)
   -- if f == 100 then
   --    f = 1
   --    io.write('Threashold bgTh: ')
   --    opt.bgTh = tonumber(io.read())
   -- end
   -- f = f + 1
   -- print("FPS: ".. 1/sys.toc()) 
end
