local nn = require 'nn'
require 'cunn'

 local Convolution = cudnn.SpatialConvolution
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
            -- :add(nn.SpatialDilatedMaxPooling(1, 1, stride, stride, 0, 0, dilation, dilation))
            :add(Avg(1, 1, stride, stride))
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

   local function branch(insert, inPlane, outPlane)
      -- iChannels = outPlane
      
      local block = nn.Sequential()
      -- local max_pooling = nn.SpatialMaxPooling(2,2,2,2)
      -- block:add(max_pooling)
      block:add(insert)
      -- block:add(nn.SpatialMaxUnpooling(max_pooling))
      -- block:add(nn.SpatialUpSamplingNearest(2))
      block:add(Deconvolution(inPlane,outPlane, 3,3, 2,2, 1,1))
      block:add(SBatchNorm(outPlane))

      local parallel = nn.ConcatTable(2)
      parallel:add(nn.Identity())
      parallel:add(block)
      
      local model = nn.Sequential()
      model:add(parallel)
      model:add(nn.JoinTable(2))
      
      return model
   end

   local function transitionUp(block2Upsample, n)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(block2Upsample)
      s:add(Deconvolution(nInputPlane, n, 3,3, 2,2, 1,1))
      
      local l = nn.Sequential()
              :add(nn.ConcatTable(2)
                 :add(s)
                 :add(nn.Identity()))
              :add(nn.JoinTable(2))
      return l
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
   	 -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
      [18]  = {{2, 2, 2, 2}, 512, basicblock},
      [35]  = {{3, 4, 6, 3}, 512, basicblock},
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

      firstLayer = nn.Sequential()
      
      model:add(Convolution(2,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      pooling_module = nn.SpatialMaxPooling(2,2,2,2,1,1)
      model:add(pooling_module)

      local levels = {}
      table.insert(levels, layer(block, 64, def[1]))
      table.insert(levels, layer(block, 128, def[2], 2))
      table.insert(levels, layer(block, 256, def[3], 2))
      table.insert(levels, layer(block, 512, def[4], 2))
     
      local block0 = table.remove(levels)
      local block1 = nn:Sequential():add(table.remove(levels))
                                    :add(branch(block0,512,256))
      local block2 = nn:Sequential():add(table.remove(levels))
                                    :add(branch(block1,512,128))
      local block3 = nn:Sequential():add(table.remove(levels))
                                    :add(branch(block2,256,64))

      model:add(block3)

      model:add(Convolution(128,64, 3,3, 1,1, 1,1))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      
      model:add(nn.SpatialMaxUnpooling(pooling_module))
      model:add(Deconvolution(64,32, 7,7, 2,2, 3,3, 1,1))
      -- model:add(Deconvolution(64,32, 7,7, 2,2, 3,3))
      model:add(SBatchNorm(32))
      model:add(ReLU(true))

      model:add(Convolution(32,1,1,1,1,1,0,0))
      model:add(nn.Sigmoid())
      
      -- while next(levels) ~= nil do
         -- model:insert(table.remove(levels),1)
         -- model:insert(transitionUp(model,2*iChannels),1)
         -- -- net:add(transitionUp(table.remove(levels)))
	 
	 -- end


   else
      error('invalid dataset: ' .. opt.dataset)
   end

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
