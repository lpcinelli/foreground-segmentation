------------------------------
-- library
------------------------------

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Deconvolution = cudnn.SpatialFullConvolution
local ReLU = cudnn.ReLU
local MaxPool = cudnn.SpatialMaxPooling
local BN = nn.SpatialBatchNormalization

------------------------------
-- function
------------------------------
local function createModel(opt)
   -- local depth = opt.depth
   local iChannels

   local function branch(insert, inPlanes, outPlanes)
   
   	local block = nn.Sequential()
   	block:add(MaxPool(2,2,2,2))
   	block:add(insert)
     --	block:add(nn.SpatialUpSamplingNearest(2))
   	block:add(Deconvolution(inPlanes, outPlanes, 2, 2, 2, 2))
   
   	local parallel = nn.ConcatTable(2)
   	parallel:add(nn.Identity())
   	parallel:add(block)
   
   	local model = nn.Sequential()
   	model:add(parallel)
   	model:add(nn.JoinTable(2))
   
   	return model
   end
   
   local function conv(n_input, n_middle, n_output, filtsize, out_bn)
   
   	local model = nn.Sequential()
   
   	model:add(cudnn.SpatialConvolution(n_input, n_middle, filtsize, filtsize, 1, 1, 1, 1))
   	model:add(BN(n_middle))
   	model:add(ReLU(true))
   
   	model:add(cudnn.SpatialConvolution(n_middle, n_output, filtsize, filtsize, 1, 1, 1, 1))
   	if out_bn == true then
   		model:add(BN(n_output))
   	        -- model:add(ReLU(true))
   	end
   
   	return model
   
   end
   
   -- number of input channels
   local num_input = 2

   -- number of output
   local num_output = num_class or 1
   
   -- filter size
   local filtsize = 3
   
   local block0 = conv(512, 1024, 1024, filtsize, true)
   
   local block1 = nn.Sequential()
   block1:add(conv(256, 512, 512, filtsize, true))
   block1:add(branch(block0, 1024, 512))
   block1:add(conv(512*2, 512, 512, filtsize, true))
   
   local block2 = nn.Sequential()
   block2:add(conv(128, 256, 256, filtsize, true))
   block2:add(branch(block1, 512, 256))
   block2:add(conv(256*2, 256, 256, filtsize, true))
   
   local block3 = nn.Sequential()
   block3:add(conv(64, 128, 128, filtsize, true))
   block3:add(branch(block2, 256, 128))
   block3:add(conv(128*2, 128, 128, filtsize, true))
   
   local model = nn.Sequential()
   model:add(conv(num_input, 64, 64, filtsize, true))
   model:add(branch(block3, 128, 64))
   model:add(conv(64*2, 64, 32, filtsize, true))
   
   model:add(cudnn.SpatialConvolution(32, num_output, 1, 1, 1, 1))	
   model:add(nn.Sigmoid())
   
   ------------------------------------------------------------------
   
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
      end
   end

   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   local function DeconvInit(name)
        for k,v in pairs(model:findModules(name)) do
           local n = v.kW*v.kH*v.nOutputPlane
           v.weight:normal(0,math.sqrt(2/n))
           -- initialize with "homogeneuos interpolation filter of stride 2"
           local factor = (v.weight:size(3)+1)/2
           local kernel1d = torch.cat(  torch.linspace(1,factor,factor),
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

   -- ConvInit('cudnn.SpatialConvolution')
   -- ConvInit('cudnn.SpatialFullConvolution')
   -- -- DeconvInit('cudnn.SpatialFullConvolution')
   -- BNInit('nn.SpatialBatchNormalization')

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
