--------------------------------------------------------------------------------
-- temporal difference demo
-- E. Culurciello, May 2015
--
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'pl'
require 'nn'
require 'sys'
require 'paths'
require 'image'
require 'qtwidget'
local frame = assert(require('frame'))

-- Local definitions -----------------------------------------------------------
local pf = function(...) print(string.format(...)) end
local Cr = sys.COLORS.red
local Cb = sys.COLORS.blue
local Cg = sys.COLORS.green
local Cn = sys.COLORS.none
local THIS = sys.COLORS.blue .. 'THIS' .. Cn

-- Title definition -----------------------------------------------------------
title = [[
Driving video parser
]]

-- Options ---------------------------------------------------------------------
opt = lapp(title .. [[
--nt             (default 8)                  Number of threads for multiprocessing
-z, --zoom       (default 1)                  Zoom ouput window
--vp             (default localhost)          file path or IP address of video stream
--batch          (default 1)                  Batch size of images for batch processing
--fps            (default 30)                 Frames per second (camera setting)
--loglevel       (default 1)                  Logging level from 0 (no logging) to 10
]])

pf(Cb..title..Cn)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.nt)
print('Number of threads used:', torch.getnumthreads())

-- global objects:
source = {} -- source object
source.ldirn = paths.dirname(opt.vp) -- base directory of the source video
source.fps = opt.fps
-- local src = torch.FloatTensor(3, source.h, source.w)

-- init application packages
frame:init(opt, source)

-- profiling timers
local timer = torch.Timer()
local t_loop = 1 -- init to 1s


-- display
local display = {} -- main display object
local zoom = opt.zoom
local loop_idx = 0
local side = math.min(source.h, source.w)
local z = side / 512 -- zoom
local win_w = zoom*source.w
local win_h = zoom*source.h

-- offset in display (we process scaled and display)
local offsd = zoom
if opt.spatial == 1 then
   offsd = zoom*source.h/eye
end

if not win then
   win = qtwidget.newwindow(win_w, win_h, 'Driving Video Parser')
else
   win:resize(win_w, win_h)
end

-- Set font size to a visible dimension
win:setfontsize(zoom*20*z)


display.forward = function(output, img, fps)
   win:gbegin()
   win:showpage()

   -- display frame
   image.display{image = img, win = win, zoom = zoom}

   win:gend()
end

-- set screen grab function
display.screen = function()
   return win:image()
end

-- set continue function
display.continue = function()
   return win:valid()
end

-- set close function
display.close = function()
   if win:valid() then
      win:close()
   end
end


-- process
process = {}
function process.forward(src)
   process.p1 = process.p1 or src -- previous frame
   process.tdi = process.tdi or torch.Tensor(src):zero()

   process.td = torch.abs(process.p1 - src) -- temporal diff images

   process.p1 = src:clone() --copy frame as previous frame
   
   return process.td
end


-- create main functions
local main = function()
   status, err = pcall(function()
      while display.continue() do
         timer:reset()

         src = frame.forward(src)
         if not src then
            break
         end

         local img = process.forward(src)
         
         display.forward(result, img, (1/t_loop))

         t_loop = timer:time().real

         collectgarbage()
      end
   end)
   if status then
      print('process done!')
   else
      print('Error ' .. err)
   end
   display.close()
end

-- execute main loop
main()
