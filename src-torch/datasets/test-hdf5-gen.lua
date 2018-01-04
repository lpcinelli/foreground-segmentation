local lfs = require 'lfs'
local sys = require 'sys'
local ffi = require 'ffi'
local image = require 'image'
require 'hdf5'
-- local hdf5 = require 'hdf5'

local URL = 'http://wordpress-jodoin.dmi.usherb.ca/static/dataset/dataset2014.zip'

local M = {}

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

local function saveHdf5(hdf5File, dir, imgPath)
	-- local name = imgPath:gsub('%.[A-z]+','')
	local name = imgPath
	hdf5File:write(paths.concat(dir, name), image.load(paths.concat(dir, imgPath)))
end

local function bgMedianModel(path, hdf5Dataset)
   local nbElem = 150
   local modelDir =  paths.concat(path, 'model')
   local referenceDir =  paths.concat(path, 'reference')
   local bgSplitDir = paths.dirname(paths.dirname(path))

   -- Is there a model already?
   if not paths.dirp(modelDir) then
      paths.mkdir(modelDir)
   -- elseif not isemptydir(modelDir) then
   --    return
   end

   paths.mkdir(modelDir)
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
   bgModelPath = modelDir .. '/staticModel.jpg'
   image.save(bgModelPath,bgModel)
   saveHdf5(hdf5Dataset, bgSplitDir, bgModelPath)

   return
end

local function findImages(dir, hdf5Dataset, hdf5FilePath)
   local imagePath = torch.CharTensor()
   local datasetDir = paths.dirname(dir)
   ----------------------------------------------------------------------

   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- list of desired video situations
   local videoTypeList={
                           -- 'PTZ',
                           -- 'badWeather',
                           'baseline',
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
   local inputPaths = {}
   local gtPaths = {}
   local bgModelPaths = {}
   local ROIPaths = {}
   local f,g

   for videoType=1,#videoTypeList do -- get specific type of video
      local dir_path = dir .. '/' .. videoTypeList[videoType]
      for video in lfs.dir(dir_path) do -- get the list of the videos of that type
         if video~="." and video~=".." then

         	local bgModelPath
         	local ROIPath
         	if paths.basename(dir) == 'train' then
	            print(datasetDir .. '/background/' .. videoTypeList[videoType] .. '/' .. video)
	            bgMedianModel(datasetDir .. '/background/' .. videoTypeList[videoType] .. '/' .. video, hdf5Dataset)

	            bgModelPath = videoTypeList[videoType] .. '/' .. video .. '/model/' .. 'staticModel'
	            ROIPath = videoTypeList[videoType] .. '/' .. video .. '/ROI/' .. 'ROI'

	            saveHdf5(hdf5Dataset, datasetDir..'/background/', ROIPath .. '.jpg')
            end

          --   print(hdf5FilePath)
          --   if not paths.filep(hdf5FilePath) then
          --   	print('saved ROI')
	         --    saveHdf5(hdf5Dataset, datasetDir..'/background/', ROIPath .. '.jpg')
	         -- end

            -- find all input frames for the current video
            f = io.popen('find -L ' .. dir_path .. '/' .. video .. '/input' .. findOptions)
            g = io.popen('find -L ' .. dir_path .. '/' .. video .. '/groundtruth/' .. findOptions)
            -- Generate a list of all the images and groundtruths
            while true do
               local line = f:read('*line')
               if not line then break end

               local inputFilename = paths.basename(line)
               local frame = inputFilename:match('%d+') -- get frame number
               local inputPath = videoTypeList[videoType] .. '/' .. video .. '/input/' .. inputFilename
               local inputExtension = inputPath:match('%.[A-z]+') -- get file extension type
               inputPath = inputPath:gsub('%.[A-z]+','')

               local gtExtension = g:read('*line'):match('%.[A-z]+') -- get file extension type
               local gtFilename = 'gt' .. frame --.. gtExtension
               local gtPath = videoTypeList[videoType] .. '/' .. video .. '/groundtruth/' .. gtFilename


               saveHdf5(hdf5Dataset, dir, inputPath .. inputExtension)
               saveHdf5(hdf5Dataset, dir, gtPath .. gtExtension)
               table.insert(inputPaths, inputPath)
               table.insert(gtPaths, gtPath)
	            table.insert(ROIPaths, ROIPath)
	            table.insert(bgModelPaths, bgModelPath)

               maxLength = math.max(maxLength, #inputPath + 1, #gtPath + 1)
            end

         end
      end
   end

   f:close()
   g:close()

   local inputPathTensor = list2tensor(inputPaths, maxLength)
   local gtPathTensor = list2tensor(gtPaths, maxLength)
   local bgModelPathTensor = list2tensor(bgModelPaths, maxLength)
   local ROIPathTensor = list2tensor(ROIPaths, maxLength)

   return inputPathTensor, gtPathTensor, bgModelPathTensor, ROIPathTensor
end


function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local gtPath = torch.CharTensor()  -- path to each groundtruth in dataset
   local bgModelPath = torch.CharTensor()  -- path to each bg model in dataset

   local trainDir = paths.concat(opt.data, 'train')
   local valDir = paths.concat(opt.data, 'val')
   local bgModelDir = paths.concat(opt.data, 'background')
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)
   assert(paths.dirp(bgModelDir), 'background directory not found: ' .. bgModelDir)

   local hdf5Dataset = hdf5.open(paths.concat(opt.data, opt.dataset .. '.h5'), 'w')

   print(" | finding all validation images")
   local valImagePath, valGtPath, valBgModelPath, valROIPath = findImages(valDir, hdf5Dataset, paths.concat(opt.data, opt.dataset .. '.h5'))

   print(" | finding all training images")
   local trainImagePath, trainGtPath, trainBgModelPath, trainROIPath = findImages(trainDir, hdf5Dataset, paths.concat(opt.data, opt.dataset .. '.h5'))

   hdf5Dataset:close()

   local info = {
      basedir = opt.data,
      train = {
         imagePath = trainImagePath,
         gtPath = trainGtPath,
         bgModelPath = trainBgModelPath,
         ROIPath = trainROIPath,

      },
      val = {
         imagePath = valImagePath,
         gtPath = valGtPath,
         bgModelPath = valBgModelPath,
         ROIPath = valROIPath,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
