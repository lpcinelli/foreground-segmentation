local nn = require 'nn'
require 'cunn'

 local Convolution = cudnn.SpatialConvolution
--local Convolution = nn.SpatialDilatedConvolution
local Deconvolution = cudnn.SpatialFullConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = cudnn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt, pretrainedParams)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels
   local pretrainedModel

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride, dilation)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      -- dilation = dilation == nil and 1 or dilation 
      if dilation > 1 then stride = 1 end
      if useConv then
         -- 1x1 convolution
	 -- if stride < 1 then
	 --    return nn:Sequential():add(Convolution(nInputPlane, nOutputPlane,1,1,1,1))
	 --       :add(nn.SpatialUpSamplingBilinear(math.floor((1/stride) + 0.5)))
         -- else
            return nn.Sequential()
               :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, 0, 0, dilation, dilation))
            :add(SBatchNorm(nOutputPlane))
         -- end
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialDilatedMaxPooling(1, 1, stride, stride, 0, 0, dilation, dilation))
            -- :add(Avg(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride, dilation)
      local nInputPlane = iChannels
      iChannels = n

      dilation = dilation == nil and 1 or dilation
      if dilation > 1 then stride = 1 end
      
      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3, stride, stride, dilation, dilation, dilation, dilation))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1, dilation, dilation, dilation,dilation))
      s:add(SBatchNorm(n))
      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride, dilation)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride, dilation)
      local nInputPlane = iChannels
      iChannels = n * 4

      dilation = dilation == nil and 1 or dilation 
      if dilation > 1 then stride = 1 end
      
      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3, stride, stride, dilation, dilation, dilation, dilation))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0,1,1))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride, dilation)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   local function basicblock2(n, stride, dilation)
      local nInputPlane = iChannels
      iChannels = n

      dilation = dilation == nil and 1 or dilation
      if dilation > 1 then stride = 1 end
      
      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3, stride, stride, dilation, dilation, dilation, dilation))
      s:add(SBatchNorm(n))
      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride, dilation)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   local function deconvblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n
      local intermediate = math.sqrt(n*nInputPlane)
  
      local s = nn.Sequential()
      s:add(Deconvolution(nInputPlane,intermediate,3,3,stride,stride,1,1))
         s:add(SBatchNorm(intermediate))
         s:add(ReLU(true))
      s:add(Deconvolution(intermediate,n,3,3,stride,stride,1,1))
         s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, 1/(stride*stride), 1)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride, dilation)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1, dilation))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 64, def[1]))
      model:add(layer(block, 128, def[2], 2))
      model:add(layer(block, 256, def[3], 1, 2))
      model:add(layer(block, 512, def[4], 1, 2))
      -- model:add(Avg(7, 7, 1, 1))
      model:add(Convolution(nFeatures,1,1,1,1,1,0,0)) -- full convolution
      -- model:add(nn.View(nFeatures):setNumInputDims(3))
      -- model:add(nn.Linear(nFeatures, 1000))
      model:add(nn.SpatialSamplingBilinear(8))
      model:add(nn.Sigmoid())
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(layer(basicblock, 16, n))
      model:add(layer(basicblock, 32, n, 2))
      model:add(layer(basicblock, 64, n, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 10))
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-100 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-100')

      -- The ResNet CIFAR-100 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(layer(basicblock, 16, n))
      model:add(layer(basicblock, 32, n, 2))
      model:add(layer(basicblock, 64, n, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 100))

   elseif opt.dataset == 'cdnet' or opt.dataset == 'sanity' then
      -- ###########################################################################################
      -- 		IMAGENET-BASED MODEL
      -- ###########################################################################################
      if opt.base ==  'imagenet' then
      	 -- Configurations for ResNet:
         --  num. residual blocks, num features, residual block function
         local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
         }

         assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
         local def, nFeatures, block = table.unpack(cfg[depth])
         iChannels = 64
         print(' | ResNet-' .. depth .. ' ImageNet-like')

         -- #############################################################################
   	 -- FULL DECONV VERSION

         model:add(Convolution(2,64,7,7,2,2,3,3))
         model:add(SBatchNorm(64))
         model:add(ReLU(true))
         pooling_module = nn.SpatialMaxPooling(2,2,2,2,1,1)
         model:add(pooling_module)
 
         model:add(layer(block, 64, def[1]))
         model:add(layer(block, 128, def[2], 2))
         model:add(layer(block, 256, def[3], 2))
         model:add(layer(block, 512, def[4], 2))

	 -- -- fully-connected
         -- model:add(Convolution(nFeatures,2048,1,1))

         -- model:add(deconvblock(128, 2))
      
	 -- local s = nn.Sequential()
         -- s:add(Deconvolution(128,64,3,3,2,2,1,1))
         -- s:add(SBatchNorm(64))
         -- s:add(ReLU(true))
         -- s:add(nn.SpatialMaxUnpooling(pooling_module))
         -- s:add(Deconvolution(64,64,7,7,2,2,3,3,1,1))
         -- s:add(SBatchNorm(64))

         -- model:add(nn.Sequential()
         --    :add(nn.ConcatTable()
         --       :add(s)
         --       :add(shortcut(128, 64, 1/8, 1)))
         --    :add(nn.CAddTable(true))
         --    :add(ReLU(true)))

	 -- model:add(Deconvolution(2048,512,3,3,2,2,1,1))
         -- model:add(SBatchNorm(512))
	 model:add(Deconvolution(512,256,3,3,2,2,1,1))
         model:add(SBatchNorm(256))
         -- model:add(ReLU(true))
         iChannels = 256

         -- model:add(Deconvolution(512,256,3,3,2,2,1,1))
         -- model:add(SBatchNorm(256))
         model:add(Deconvolution(256,128,3,3,2,2,1,1))
         model:add(SBatchNorm(128))
         -- model:add(ReLU(true))
         iChannels = 128

         -- model:add(Deconvolution(256,64,3,3,2,2,1,1))
         model:add(Deconvolution(128,64,3,3,2,2,1,1))
         model:add(SBatchNorm(64))
         -- model:add(ReLU(true))
         iChannels = 64

         model:add(nn.SpatialMaxUnpooling(pooling_module))
         -- model:add(Deconvolution(64,64,7,7,2,2,3,3,1,1))
         -- model:add(SBatchNorm(64))
         model:add(Deconvolution(64,32,7,7,2,2,3,3,1,1))
         model:add(SBatchNorm(32))
         model:add(ReLU(true))

	 -- model:add(Convolution(64,1,1,1,1,1,0,0))
	 model:add(Convolution(32,1,1,1,1,1,0,0))
         model:add(nn.Sigmoid())

         -- #############################################################################
   	 -- DILATED VERSION - 8 STRIDE
	 
         -- model:add(Convolution(2,3,1,1,1,1))
         -- model:add(Convolution(2,64,7,7,2,2,3,3))
         -- model:add(SBatchNorm(64))
         -- model:add(ReLU(true))
         -- model:add(Max(3,3,2,2,1,1))
         -- model:add(layer(block, 64, def[1]))
         -- model:add(layer(block, 128, def[2], 2, 1))
         -- model:add(layer(block, 256, def[3], 1, 2))
         -- model:add(layer(block, 512, def[4], 1, 4))
         -- model:add(Convolution(nFeatures,1,1,1,1,1,0,0)) -- full convolution
         -- -- model:add(Convolution(nFeatures,1,3,3,1,1,6,6,6,6)) -- full convolution
         -- model:add(nn.SpatialUpSamplingBilinear(8))
         -- model:add(nn.Sigmoid())

         -- #############################################################################
   	 -- UPSAMPLE ONLY VERSION
      
         -- model:add(Convolution(2,64,7,7,2,2,3,3))
         -- model:add(SBatchNorm(64))
         -- model:add(ReLU(true))
         -- model:add(Max(3,3,2,2,1,1))
         -- model:add(layer(block, 64, def[1]))
         -- model:add(layer(block, 128, def[2], 2))
         -- model:add(layer(block, 256, def[3], 2))
         -- model:add(layer(block, 512, def[4], 2))
         -- model:add(Convolution(nFeatures,1,1,1,1,1,0,0)) -- full convolution
         -- -- model:add(Convolution(nFeatures,1,3,3,1,1,6,6,6,6)) -- full convolution
         -- model:add(nn.SpatialUpSamplingBilinear(32))
         -- model:add(nn.Sigmoid())

      -- ###########################################################################################
      -- 		CIFAR-10-BASED MODEL
      -- ###########################################################################################
      elseif opt.base ==  'cifar10' then
         assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
         local n = (depth - 2) / 6
         local padSize = {math.ceil((opt.patchSize-1)/2), math.floor((opt.patchSize-1)/2)}
         local finalFeatMapSize = math.ceil(opt.patchSize/4)
         iChannels = 16
         print(' | ResNet-' .. depth .. ' CDNET2014')
   
         -- The ResNet CDNET2014 model
         -- #############################################################################
   	-- DECONV VERSION
   
         -- model:add(Convolution(2,16,3,3,1,1,1,1)) -- greyscale image (2 channel only: ref + target)
         -- model:add(SBatchNorm(16))
         -- model:add(ReLU(true))
         -- model:add(layer(basicblock, 16, n))
         -- model:add(layer(basicblock, 32, n, 2))
         -- model:add(layer(basicblock, 64, n, 2))
         -- ---------------------------------------------------------
         -- -- Use 1x1 conv rather than FC to improve sliding window performance
         -- model:add(Convolution(64,128,1,1,1,1,0,0))
         -- model:add(SBatchNorm(128))
         -- model:add(ReLU(true))
         -- ---------------------------------------------------------
         -- -- should I do deconv here to get the original image size back? Yes!
         -- -- simple bilinear upsampling could be done but recover exact original size is difficult due to strided conv
         -- -- should it be done before or after sigmoid()?
         -- -- the average pooling layer is "compensated" in the reconstruction approach through the initial padding do the input image
         -- model:add(nn.SpatialFullConvolution(128,64,3,3,2,2,1,1,1,1))
         -- model:add(SBatchNorm(64))
         -- model:add(ReLU(true))
         -- model:add(nn.SpatialFullConvolution(64,1,3,3,2,2,1,1,1,1))
         -- -------------------------------------------------------
         -- -- output should be the prob. of being FG
         -- -- nn.Sigmoid() acts element-wise
         -- model:add(nn.Sigmoid())
   
         -- #############################################################################
   	-- DECODER VERSION
   
        -- model:add(Convolution(2,16,3,3,1,1,1,1))
        -- model:add(SBatchNorm(16))
        -- model:add(ReLU(true))
        -- model:add(layer(basicblock, 16, n))
        -- model:add(layer(basicblock, 32, n, 2))
        -- model:add(layer(basicblock, 64, n, 2))
         ---------------------------------------------------------
        -- -- the SegNet uses unpooling to upsample but I dont have the pooling indices cuz I do strided conv to downsample
	-- model:add(nn.SpatialUpSamplingBilinear(2))
        -- model:add(layer(basicblock2, 32, n))
        -- -- model:add(Convolution(64,32,3,3,1,1,1,1,1,1))
        -- -- model:add(SBatchNorm(32))
        -- -- model:add(ReLU(true))
         
	-- model:add(nn.SpatialUpSamplingBilinear(2))
        -- model:add(layer(basicblock2, 16, n))
        -- -- model:add(Convolution(32,16,3,3,1,1,1,1,1,1))
        -- -- model:add(SBatchNorm(16))
        -- -- model:add(ReLU(true))
         ---------------------------------------------------------
	-- -- At the end FC that outputs nb of classes (FG only)
	-- model:add(Convolution(16,1,1,1,1,1,0,0))
        -- model:add(nn.Sigmoid())
   
         -- #############################################################################
   	-- UPSAMPLE ONLY VERSION
   
         model:add(Convolution(2,16,3,3,1,1,1,1)) -- greyscale image (2 channel only: ref + target)
         model:add(SBatchNorm(16))
         model:add(ReLU(true))
         model:add(layer(basicblock, 16, n))
         model:add(layer(basicblock, 32, n, 2))
         model:add(layer(basicblock, 64, n, 2))
         model:add(Convolution(64,1,1,1,1,1,0,0))
         model:add(nn.SpatialUpSamplingBilinear(4))
         model:add(nn.Sigmoid())
    
         -- #############################################################################
   	-- FULL DILATED VERSION
   	-- too big for gpu mem, use dilation = 2 instead
   
         -- model:add(Convolution(2,16,3,3,1,1,1,1)) -- greyscale image (2 channel only: ref + target)
         -- model:add(SBatchNorm(16))
         -- model:add(ReLU(true))
         -- model:add(layer(basicblock, 16, n))
         -- model:add(layer(basicblock, 32, n, 2, 1))
         -- model:add(layer(basicblock, 64, n, 2, 1))
         -- model:add(Convolution(64,1,1,1,1,1,0,0))
         -- model:add(nn.SpatialUpSamplingBilinear(4))
         -- model:add(nn.Sigmoid())

      else
         error('invalid base: ' .. opt.base)
      end -- end if base

   else
      error('invalid dataset: ' .. opt.dataset)
   end -- end if dataset

   local function ConvInit(name)
      local modules = model:findModules(name)
      for k,v in pairs(modules) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 and k < #modules then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
	 -- end
      end
   end

   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   local function UpSampleInit(name)
	for k,v in pairs(model:findModules(name)) do
           local n = v.kW*v.kH*v.nOutputPlane
           v.weight:normal(0,math.sqrt(2/n))
	   -- initialize with "homogeneuos interpolation filter of stride 2"
	   local factor = (v.weight:size(3)+1)/2
	   local kernel1d = torch.cat(	torch.linspace(1,factor,factor),
	                        torch.linspace(factor-1,1,factor-1))/factor
           kernel1d:resize(2*factor-1,1)
	   local kernel2d = kernel1d*kernel1d:t()
	   for i=1,v.weight:size(2) do
	      v.weight[{{i},{i},{},{}}] = kernel2d
	   end
           if cudnn.version >= 4000 then
              v.bias = nil
              v.gradBias = nil
           else
              v.bias:zero()
           end
	end
   end

   local function ReusePretrainedWeights(pretrainedLayer, newLayer)
      local status, err = pcall(function()
         local pretrainedModules = pretrainedModel:findModules(pretrainedLayer)
         local newModules = model:findModules(newLayer)
	 -- if #pretrainedModules == 0 then
            -- pretrainedModules = pretrainedModel:findModules('nn.SpatialConvolution')
         -- end
         for i=1,#pretrainedModules do
            if i> #newModules then break end
	    local pretrainedSize = torch.Tensor(pretrainedModules[i].weight:size():totable())
	    local newSize = torch.Tensor(newModules[i].weight:size():totable())
            -- 	torch.typename(pretrainedModules[i]) == 'cudnn.SpatialConvolution') then
           	-- print(newModules[i].weight:size())
            	-- torch.typename(pretrainedModules[i]) == 'cudnn.SpatialConvolution') and
           	-- if  pretrainedModules[i].weight:size() == newModules[i].weight:size() then
           	if torch.all(pretrainedSize:eq(newSize)) then
		   -- if newModules[i].bias and pretrainedModules[i].bias then
		      -- print('copying bias')
		      -- newModules[i].bias:copy(pretrainedModules[i].bias)
	   	   print('copying item ' .. tostring(i))
	   	   print(pretrainedModules[i])
	   	   print(newModules[i])
                   newModules[i].weight:copy(pretrainedModules[i].weight)
		   -- print('copying bias')
                   newModules[i].bias:copy(pretrainedModules[i].bias)
	   	   -- print('\n')
               -- newModules(i).gradWeight:copy(pretrainedModules[i].gradWeight)
               -- newModules(i).gradBias:copy(pretrainedModules[i].gradBias)
               end
         end
      end)
      if not status then 
	 print('error during weight loading')
         print(err) 
      end
      return
   end



   UpSampleInit('cudnn.SpatialFullConvolution')
   UpSampleInit('nn.SpatialFullConvolution')
   
   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   -- ConvInit('cudnn.SpatialFullConvolution')
   -- ConvInit('nn.SpatialFullConvolution')
   ConvInit('cudnn.SpatialDilatedConvolution')
   ConvInit('nn.SpatialDilatedConvolution')
   
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
 
   if pretrainedParams then
      pretrainedModel = torch.load(pretrainedParams)
      ReusePretrainedWeights('cudnn.SpatialConvolution', 'nn.SpatialDilatedConvolution')
      ReusePretrainedWeights('nn.SpatialConvolution', 'nn.SpatialDilatedConvolution')
      ReusePretrainedWeights('nn.SpatialBatchNormalization','nn.SpatialBatchNormalization')
   end

   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   model:cuda()

   if nn.SpatialBatchNormalization.cudnn == 'deterministic' then
         local newModules = model:findModules(newLayer)
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   -- all architctures but cdnet begin with conv
   -- cudnn conv first layer should ignore gradInput
   if opt.dataset ~= 'cdnet' and opt.dataset ~= 'sanity' then
      model:get(1).gradInput = nil
   end

   return model
end

return createModel
