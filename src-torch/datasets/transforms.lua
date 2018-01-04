--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}

function M.Compose(transforms)
	return function(input,target)

      for _, transform in ipairs(transforms) do
			local targetOut = nil
         input, targetOut = transform(input,target)
         if targetOut ~= nil then target = targetOut end
      end
      return input, target
   end
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end

-- Scales the smaller edge to size
function M.Scale(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input,target)
      local w, h = input:size(3), input:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return input
      end
      if w < h then
			if target then
				return image.scale(input, size, h/w * size, interpolation), image.scale(target, size, h/w * size, 'simple')
         else
				return image.scale(input, size, h/w * size, interpolation)
         end
      else
			if target then
	         return image.scale(input, w/h * size, size, interpolation), image.scale(target, w/h * size, size, 'simple')
	      else
				return image.scale(input, w/h * size, size, interpolation)
	      end
      end
   end
end

function M.ScaleDim(size, dim, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input,target)

      local w, h = input:size(3), input:size(2)
      if dim == 'w' then
			if target then
	         return image.scale(input, size, h, interpolation), image.scale(target, size, h, 'simple')
	      else
				return image.scale(input, size, h, interpolation)
	      end
      elseif dim == 'h' then
			if target then
	         return image.scale(input, w, size, interpolation), image.scale(target, w, size, 'simple')
	      else
				return image.scale(input, w, size, interpolation)
			end
      else
         print('wrong dimension requested')
      end
   end
end

-- Crop to centered rectangle
function M.CenterCrop(size)
   return function(input, target)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      if target then
		   return image.crop(input, w1, h1, w1 + size, h1 + size), -- center patch
					 image.crop(target, w1, h1, w1 + size, h1 + size)
		else
		   return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
		end
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(size, padding)
   padding = padding or 0

   return function(input,target)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
         if target then
	         local temp = target.new(3, target:size(2) + 2*padding, target:size(3) + 2*padding)
	         temp:zero()
	            :narrow(2, padding+1, target:size(2))
	            :narrow(3, padding+1, target:size(3))
	            :copy(target)
	         target = temp
	      end
      end

      local w, h = input:size(3), input:size(2)
      if w == size and h == size then
         return input, target
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local out = image.crop(input, x1, y1, x1 + size, y1 + size)
      assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
      local targetOut
      if target then
	      targetOut = image.crop(target, x1, y1, x1 + size, y1 + size)
	   end
      return out, targetOut
   end
end

-- Four corner patches and center crop from image and its horizontal reflection
-- TODO: Extend to target
function M.TenCrop(size)
   local centerCrop = M.CenterCrop(size)

   return function(input)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1)
   end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
   return function(input,target)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end
      if target then
	      return image.scale(input, targetW, targetH, 'bicubic'), image.scale(target, targetW, targetH, 'simple')
	   else
			return image.scale(input, targetW, targetH, 'bicubic')
	   end
   end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCrop(size)
   local scale = M.Scale(size)
   local crop = M.CenterCrop(size)

   return function(input)
      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(0.08, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            local targetOut = image.crop(target, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic'), image.scale(targetOut, size, size, 'simple')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input,target))
   end
end

function M.HorizontalFlip(prob)
   return function(input, target)
      if torch.uniform() < prob then
         input = image.hflip(input)
         target = image.hflip(target)
      end
      return input, target
   end
end

function M.Rotation(deg)
   return function(input, target)
      if deg ~= 0 then
         input = image.rotate(input, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
         if target then
				target = image.rotate(target, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
			end
      end
      return input, target
   end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.OneGrayscaleImagePerChannel()
   local gs

   return function(input,target)
      local nbImages, H, W =  input:size(4), input:size(2), input:size(3)
      local gsTensor = torch.Tensor(nbImages, H, W)
      for i, img in ipairs(input:split(1,4))  do
		  img = img:squeeze()
        gs = gs or img.new()
        grayscale(gs, img)
        gsTensor[i] = gs[1]
      end

      return gsTensor
   end
end

function M.getImageROI()

   return function(input, target)
      local videoROI = input:narrow(4, 1, 1):squeeze()
      local bgModel = input:narrow(4, 2, 1):squeeze()
      local img = input:narrow(4, 3, 1):squeeze()

      local _, maxIndex = videoROI:max(1)
      local frameROI = maxIndex:eq(1):float():expandAs(videoROI)
      img:cmul(img, frameROI)
      bgModel:cmul(bgModel, frameROI)
      
      return torch.cat({bgModel, img},4):squeeze()
   end
end


function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Contrast(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.RandomOrder(ts)
   return function(input)
      local img = input.img or input
      local order = torch.randperm(#ts)
      for i=1,#ts do
         img = ts[order[i]](img)
      end
      return img
   end
end

-- using the 1x1 conv approach instead should vetter use resources
-- when using 1x1 conv approach this function shouldnt be used
function SlidingWindow(opt)
   local patchSize = opt.patchSize
   local padding = (patchSize-1)/2
   local stride = 1
   return function(input)
      -- padding image
      -- Dont forget to change back the first dim size to 2 !!!!
      local temp = input.new(1, input:size(2) + 2*padding, input:size(3) + 2*padding)
      temp:zero()
         :narrow(2, padding+1, input:size(2))
         :narrow(3, padding+1, input:size(3))
         :copy(input)
      local output = {}
      for i = 0,input:size(2)-1,stride do
         for k = 0,input:size(3)-1,stride do
            -- crop tiny patchSize patchs through the whole image
            table.insert(output, image.crop(temp, k, i, k+patchSize, i+patchSize))
         end
      end
      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end
      return input.cat(output, 1)
   end
end

function M.ColorJitter(opt)
   local brightness = opt.brightness or 0
   local contrast = opt.contrast or 0
   local saturation = opt.saturation or 0

   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, M.Brightness(brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, M.Contrast(contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, M.Saturation(saturation))
   end

   if #ts == 0 then
      return function(input) return input end
   end

   return M.RandomOrder(ts)
end

return M
