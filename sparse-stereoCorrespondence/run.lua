--------------------------------------------------------------------------------
-- Sparse stereo correspondence with Torch7
--------------------------------------------------------------------------------
-- Alfredo Canziani, May 2013
--------------------------------------------------------------------------------

require 'camera'
require 'nnx'
-- Computing the edges of the LEFT image (RIGHT camera)
require 'edgeDetector'

width  = 320 --1600
height = 240 --896
fps = 30
-- dir = "demo_test"
local neighborhood = image.gaussian1D(21)
local normalisation = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3)


-- sys.execute(string.format('mkdir -p %s',dir))

camera1 = image.Camera{idx=0,width=width,height=height,fps=fps}
camera2 = image.Camera{idx=1,width=width,height=height,fps=fps}

---------------------------------------------------------------------------------
-- In this program camera1 is supposed to serve as the RIGHT camera,
-- whereas camera2 shall match the LEFT camera. The LEFT and RIGHT cameras
-- provide respectively the RIGHT- and LEFT-shifted images

-- iCameraX[c]: {i}mage from {Camera} {X} [{c}olour version; greyscale otherwise]
---------------------------------------------------------------------------------

iCameraRc = camera1:forward() -- acquiring image from the RIGHT camera
iCameraLc = camera2:forward() -- acquiring image from the LEFT camera

-- converting in B&W

iCameraR = image.rgb2y(iCameraRc):float()
iCameraL = image.rgb2y(iCameraLc):float()

-- f = 1
opt = {}
opt.kSize = 5

while true do
   sys.tic()
   iCameraR = image.rgb2y(camera1:forward()):float()
   iCameraL = image.rgb2y(camera2:forward()):float()
   edgesR = edgeDetector(iCameraR:double(),opt.kSize):float()[1]
   edgesL = edgeDetector(iCameraL:double(),opt.kSize):float()[1]
   edgesL = edgesL:cmul(edgesL:gt(2):float())
   win = image.display{win=win,image={iCameraR,iCameraL}, legend="FPS: ".. 1/sys.toc(), min=0, max=1,nrow=2}
   win2 = image.display{win=win2,image={edgesR,edgesL}, legend="FPS: ".. 1/sys.toc(), min=0, max=1,nrow=2}
   -- image.savePNG(string.format("%s/frame_1_%05d.png",dir,f),a1)
   -- f = f + 1
   -- print("FPS: ".. 1/sys.toc()) 
end
