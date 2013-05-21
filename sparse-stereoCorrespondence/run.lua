--------------------------------------------------------------------------------
-- Sparse stereo correspondence with Torch7
--------------------------------------------------------------------------------
-- Alfredo Canziani, May 2013
--------------------------------------------------------------------------------

require 'camera'
width  = 320 --1600
height = 240 --896
fps = 30
-- dir = "demo_test"

-- sys.execute(string.format('mkdir -p %s',dir))

camera1 = image.Camera{idx=1,width=width,height=height,fps=fps}
camera2 = image.Camera{idx=2,width=width,height=height,fps=fps}

a1 = camera1:forward()
a2 = camera2:forward()
-- f = 1

while true do
   sys.tic()
   a1 = camera1:forward()
   win = image.display{win=win,image={a1,a2}, legend="FPS: ".. 1/sys.toc(), min=0, max=1}
   -- image.savePNG(string.format("%s/frame_1_%05d.png",dir,f),a1)
   -- f = f + 1
   -- print("FPS: ".. 1/sys.toc()) 
end
