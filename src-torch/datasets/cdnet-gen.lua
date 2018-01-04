local lfs = require 'lfs'
local sys = require 'sys'
local ffi = require 'ffi'
local image = require 'image'
local t = require 'datasets/transforms'
local URL = 'http://wordpress-jodoin.dmi.usherb.ca/static/dataset/dataset2014.zip'

local M = {}

function preprocess(input, target)
   local  transform = t.Compose{
      t.getImageROI(),
      t.OneGrayscaleImagePerChannel(),
      t.ScaleDim(256,'w'), -- 256
      t.ScaleDim(192,'h'), -- 192
      -- t.Lighting(0.1, pca.eigval, pca.eigvec),
      -- t.ColorNormalize(meanstd),
      -- t.HorizontalFlip(0.5),
      }
   return transform(input,target)
end

-- Check whether dir is empty or net
local function isemptydir(directory)
	for filename in lfs.dir(directory) do
		if filename ~= '.' and filename ~= '..' then
			return false
		end
	end
	return true
end

-- Convert the generated list to a tensor for faster loading
local function list2tensor(imagePaths, maxLength)
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end
   return imagePath
end

local function bgMedianModel(path)
   local nbElem = 150
   local modelDir =  paths.concat(path, 'model')
   local referenceDir =  paths.concat(path, 'reference')

   -- Is there a model already?
   if not paths.dirp(modelDir) then
      paths.mkdir(modelDir)

   elseif not isemptydir(modelDir) then

      return image.load(modelDir .. '/staticModel.jpg', 3, 'float')
   end
   -- paths.mkdir(modelDir)
   local refImgs

   for img in lfs.dir(referenceDir) do

      if img ~= "." and img ~= ".." then
         if refImgs == nil then
            refImgs = image.load(referenceDir .. '/' .. img, 3, 'float')
         else
            refImgs = torch.cat(refImgs, image.load(referenceDir .. '/' .. img, 3, 'float'), 4)
         end
      end
   end
   bgModel = torch.median(refImgs, 4):squeeze()
   bgModelpath = modelDir .. '/staticModel.jpg'
   image.save(bgModelpath,bgModel)
   return bgModel
end

local function findImages(dir, dstDir)
   -- local imagePath = torch.CharTensor()


   local datasetDir = paths.dirname(dir)
   local split = paths.basename(dir)
   ----------------------------------------------------------------------

   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- list of desired video situations
   local videoTypeList={
                           'PTZ',
                           -- 'badWeather',
                           -- 'baseline',
                           -- 'cameraJitter',
                           -- 'dynamicBackground',
                           -- 'intermittentObjectMotion',
                           -- 'lowFramerate',
                           -- 'nightVideos',
                           -- 'shadow',
                           -- 'thermal',
                           -- 'turbulence'
                        }
   -- list of undesireddisregarded videos
   local list_of_videos = {}

   local maxLength = -1
   -- local inputPaths = {}
   -- local gtPaths = {}
   -- local bgModelPaths = {}
   -- local ROIPaths = {}
   -- local processedImgs = {}
   local imgPaths = {}
   local f,g

   for videoType=1,#videoTypeList do -- get specific type of video
      local dir_path = dir .. '/' .. videoTypeList[videoType]
      for video in lfs.dir(dir_path) do -- get the list of the videos of that type
         if video~="." and video~=".." then

            print(datasetDir .. '/background/' .. videoTypeList[videoType] .. '/' .. video)
            local bgModel = bgMedianModel(datasetDir .. '/background/' .. videoTypeList[videoType] .. '/' .. video)
            local bgModelPath = videoTypeList[videoType] .. '/' .. video .. '/model/' .. 'staticModel.jpg'

	    local ROIPath = videoTypeList[videoType] .. '/' .. video .. '/ROI/' .. 'ROI.jpg'
	    local ROIPathComplete = datasetDir .. '/background/' .. ROIPath
	    local ROIImg = image.load(ROIPathComplete, 3, 'float')

            -- find all input frames for the current video
            f = io.popen('find -L ' .. dir_path .. '/' .. video .. '/input' .. findOptions)
            g = io.popen('find -L ' .. dir_path .. '/' .. video .. '/groundtruth/' .. findOptions)
            local gtLine = g:read('*line')
            local gtExtension = gtLine:match('%.[A-z]+') -- get file extension type
            -- paths.mkdir(dstDir, split .. '/' .. videoTypeList[videoType] .. '/' .. video)
            -- Generate a list of all the images and groundtruths
            while true do
               local inputLine = f:read('*line')
               if not inputLine then break end

               local inputFilename = paths.basename(inputLine)
               local frame = inputFilename:match('%d+') -- get frame number

               -- local inputPath = videoTypeList[videoType] .. '/' .. video .. '/input/' .. inputFilename

               local gtFilename = 'gt' .. frame .. gtExtension
               local gtPath = datasetDir .. '/' .. split .. '/' ..  videoTypeList[videoType] .. '/' .. video .. '/groundtruth/' .. gtFilename
	       local imgPath = paths.concat(dstDir, split .. '/' .. videoTypeList[videoType] .. '/' .. video)
               paths.mkdir(imgPath)
               imgPath = paths.concat(imgPath, frame)
               table.insert(imgPaths, imgPath)
               -- table.insert(inputPaths, inputPath)
               -- table.insert(gtPaths, gtPath)
               -- table.insert(bgModelPaths, bgModelPath)
               -- table.insert(ROIPaths, ROIPath)

               -- maxLength = math.max(maxLength, #inputPath + 1, #gtPath + 1)
               maxLength = math.max(maxLength, #imgPath + 1)

	       local inputImg = image.load(inputLine, 3, 'float')
	       local gtImg = image.load(gtPath, 1, 'float')
	       local bgModelImg = torch.FloatTensor(bgModel:size()):copy(bgModel)
   	       gtImg = gtImg:eq(1):float()

	       local input, target = preprocess(torch.cat({ROIImg:expandAs(inputImg), bgModelImg:expandAs(inputImg), inputImg}, 4), gtImg)
               torch.save(imgPath, {input = input, target = target})
            end
         end
      end
   end

   f:close()
   g:close()

   -- local inputPathTensor = list2tensor(inputPaths, maxLength)
   -- local gtPathTensor = list2tensor(gtPaths, maxLength)
   -- local bgModelPathTensor = list2tensor(bgModelPaths, maxLength)
   -- local ROIPathTensor = list2tensor(ROIPaths, maxLength)
   local imgPathTensor = list2tensor(imgPaths, maxLength)

   -- return inputPathTensor, gtPathTensor, bgModelPathTensor, ROIPathTensor
   return imgPathTensor 
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local gtPath = torch.CharTensor()  -- path to each groundtruth in dataset
   local bgModelPath = torch.CharTensor()  -- path to each bg model in dataset

   local trainDir = paths.concat(opt.data, 'train')
   local valDir = paths.concat(opt.data, 'val')
   local bgModelDir = paths.concat(opt.data, 'background')
   local dstDir = paths.concat(opt.gen, opt.dataset)
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)
   assert(paths.dirp(bgModelDir), 'background directory not found: ' .. bgModelDir)

   print(" | finding all validation images")
   local valImages = findImages(valDir, dstDir)
   torch.save(paths.concat(opt.gen, opt.dataset .. 'Val.t7'), valImages)
   -- local valImagePath, valGtPath, valBgModelPath, valROIPath = findImages(valDir)

   print(" | finding all training images")
   local trainImages = findImages(trainDir, dstDir)
   torch.save(paths.concat(opt.gen, opt.dataset .. 'Train.t7'), trainImages)
   -- local trainImagePath, trainGtPath, trainBgModelPath, trainROIPath = findImages(trainDir)

   local info = {
      basedir = opt.data,
      train = {
	 imagePath = trainImages,
	 -- processedImgs = paths.concat(opt.gen, opt.dataset .. 'Train.t7'),
         -- imagePath = trainImagePath,
         -- gtPath = trainGtPath,
         -- bgModelPath = trainBgModelPath,
         -- ROIPath = trainROIPath,

      },
      val = {
	 imagePath = valImages,
	 -- processedImgs = paths.concat(opt.gen, opt.dataset .. 'Val.t7'),
         -- imagePath = valImagePath,
         -- gtPath = valGtPath,
         -- bgModelPath = valBgModelPath,
         -- ROIPath = valROIPath,
      },
   }

   print(" | saving image tensor files  to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
