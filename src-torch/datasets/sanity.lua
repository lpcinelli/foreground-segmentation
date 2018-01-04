--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local CDNetDataset = torch.class('resnet.CDNetDataset', M)

function CDNetDataset:__init(imageInfo, opt, split)
   -- self.imageInfo = torch.load(imageInfo[split].processedImgs)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   self.bgDir = paths.concat(opt.data, 'background')
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function CDNetDataset:get(i)
   local imagePath = ffi.string(self.imageInfo.imagePath[i]:data())
   -- local gtPath = ffi.string(self.imageInfo.gtPath[i]:data())
   -- local bgModelPath = ffi.string(self.imageInfo.bgModelPath[i]:data())
   -- local ROIPath = ffi.string(self.imageInfo.ROIPath[i]:data())

   -- local img = self:_loadImage(paths.concat(self.dir, imagePath), 3)
   -- local gtImage = self:_loadImage(paths.concat(self.dir, gtPath), 1)
   -- local bgModel = self:_loadImage(paths.concat(self.bgDir, bgModelPath), 3)
   -- local videoROI = self:_loadImage(paths.concat(self.bgDir, ROIPath), 3)

   -- gtImage = gtImage:eq(1):float()

   -- return {
				-- input = torch.cat({videoROI:expandAs(img), bgModel:expandAs(img), img}, 4),
				-- target = gtImage,
			 -- }
   return torch.load(imagePath)
end

function CDNetDataset:_loadImage(path, channels)
   channels = channels == nil and 3 or channels
   local ok, input = pcall(function()
      return image.load(path, channels, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, channels, 'float')
   end

   return input
end

function CDNetDataset:size()
   -- return #self.imageInfo
   return self.imageInfo.imagePath:size(1)
   end


-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function CDNetDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         -- t.getImageROI(),
         -- t.OneGrayscaleImagePerChannel(),
         -- t.ScaleDim(256,'w'),
         -- t.ScaleDim(192,'h'),  -- previously 192
         -- t.Lighting(0.1, pca.eigval, pca.eigvec),
         -- t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      return t.Compose{
         -- t.getImageROI(),
         -- t.OneGrayscaleImagePerChannel(),
         -- t.ScaleDim(256,'w'),
         -- t.ScaleDim(192,'h'),
         -- t.ColorNormalize(meanstd),
         -- Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CDNetDataset
