local frame = {}

local pf = function(...) print(string.format(...)) end
local Cr = sys.COLORS.red
local Cb = sys.COLORS.blue
local Cg = sys.COLORS.green
local Cn = sys.COLORS.none
local THIS = sys.COLORS.blue .. 'THIS' .. Cn

local function prep_libvideo_decoder_video(opt, source)
   local cam = assert(require('libvideo_decoder'))
   cam.loglevel(opt.loglevel)

   local status = false
   status, source.h, source.w, source.length, source.fps = cam.init(opt.vp);
   if not status then
      error("No video")
   else
      if opt.loglevel > 0 then
         pf(Cb..'video statistics: %s fps, %dx%d (%s frames)'..Cn,
            (source.fps and tostring(source.fps) or 'unknown'),
            source.h,
            source.w,
            (source.length and tostring(source.length) or 'unknown'))
      end
   end

   -- video library only handles byte tensors
   local img_tmp = torch.FloatTensor(opt.batch, 3, source.h, source.w)

   -- set frame forward function
   frame.forward = function(img)
      local n = opt.batch
      for i=1,opt.batch do
         if not cam.frame_rgb(img_tmp[i]) then
            if i == 1 then
               return false
            end
            n = i-1
            break
         end
      end
      if n == opt.batch then
         img = img_tmp:clone()
      else
         img = img_tmp:narrow(1,1,n):clone()
      end

      return img
   end

   source.cam = cam
end


function frame:init(opt, source)
      prep_libvideo_decoder_video(opt, source)
end

return frame
