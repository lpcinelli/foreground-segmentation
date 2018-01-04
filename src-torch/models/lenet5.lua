local nn = require 'nn'
require 'cunn'
require 'cudnn'
require 'dpnn'

local Convolution = cudnn.SpatialConvolution
-- local Convolution = nn.SpatialConvolution
local Deconvolution = cudnn.SpatialFullConvolution
local DilatedConv = nn.SpatialDilatedConvolution
local Avg = cudnn.SpatialAveragePooling
local DilatedMax = nn.SpatialDilatedMaxPooling
local Max = nn.SpatialMaxPooling
local Unpool = nn.SpatialMaxUnpooling
local ReLU = cudnn.ReLU
local SBatchNorm = nn.SpatialBatchNormalization
local Crop = nn.SpatialUniformCrop

local function createModel(opt)

   local model = nn.Sequential()

   if opt.dataset == 'cdnet' or opt.dataset == 'sanity' or opt.dataset == 'test-hdf5' then
		-- -- DECODER VERSION 
		-- -- conv subnet
      		-- -- -- model:add(nn.SpatialZeroPadding(7,7,7,7))
		-- model:add(Convolution(2,6,5,5,1,1,2,2))
		-- model:add(SBatchNorm(6))
		-- model:add(ReLU(true))
		-- local pool1 = Max(3,3,3,3)
		-- model:add(pool1)

		-- model:add(Convolution(6,16,5,5,1,1,2,2))
		-- model:add(SBatchNorm(16))
		-- model:add(ReLU(true))
		-- local pool2 = Max(3,3,3,3)
		-- model:add(pool2)

		-- -- -- fully connected 1
		-- -- model:add(Convolution(16,120,3,3,1,1,1,1))
		-- -- model:add(SBatchNorm(120))
		-- -- model:add(ReLU(true))

		-- -- -- deconv subnet
		-- -- model:add(Convolution(120,16,3,3,1,1,1,1))
		-- -- model:add(SBatchNorm(16))
		-- -- model:add(ReLU(true))

		-- model:add(Unpool(pool2))
		-- model:add(Convolution(16,6,5,5,1,1,2,2))
		-- -- model:add(SBatchNorm(6))
		-- model:add(ReLU(true))

		-- model:add(Unpool(pool1))
		-- model:add(Convolution(6,6,5,5,1,1,2,2))
		-- -- model:add(SBatchNorm(6))
		-- model:add(ReLU(true))

		-- model:add(Convolution(6,1,1,1,1,1,0,0))
		-- model:add(nn.Sigmoid())

 -- #############################################################################
		-- DECONV VERSION
		model:add(Convolution(2,6,5,5,1,1,2,2))
		model:add(ReLU(true))
		model:add(SBatchNorm(6))
		model:add(Max(3,3,3,3))

		model:add(Convolution(6,16,5,5,1,1,2,2))
		model:add(SBatchNorm(16))
		model:add(ReLU(true))
		model:add(Max(3,3,3,3))

		-- -- fully connected 1
		model:add(Convolution(16,120,3,3,1,1,1,1))
		model:add(SBatchNorm(120))
		model:add(ReLU(true))

		model:add(Deconvolution(120,16,3,3,3,3,0,0,1,1))
		-- model:add(SBatchNorm(16))
		-- model:add(ReLU(true))
		
		model:add(Deconvolution(16,6,5,5,3,3,1,1,1,0))
		-- model:add(SBatchNorm(6))
		-- model:add(ReLU(true))

		-- testar usando kernel 5x5
		model:add(Convolution(6,1,1,1,1,1,0,0))
		model:add(nn.Sigmoid())

 -- #############################################################################
		-- UPSAMPLE ONLY VERSION

      		-- model:add(nn.SpatialZeroPadding(7,7,7,7))
		-- model:add(Convolution(2,6,5,5,1,1,2,2))
		-- -- model:add(SBatchNorm(6))
		-- model:add(ReLU(true))
		-- model:add(Max(3,3,2,2))

		-- model:add(Convolution(6,16,5,5,1,1,2,2))
		-- -- model:add(SBatchNorm(16))
		-- model:add(ReLU(true))
		-- model:add(Max(3,3,2,2))

		-- -- fully connected 1 
		-- model:add(Convolution(16,120,3,3,1,1,0,0))
		-- -- model:add(SBatchNorm(120))
		-- model:add(ReLU(true))

		-- -- fully connected 2
		-- model:add(Convolution(120,1,1,1,1,1,0,0))
		-- model:add(nn.SpatialUpSamplingBilinear(4))
		-- model:add(nn.Sigmoid())

 -- #############################################################################
		-- FULL DILATED VERSION

      		-- model:add(nn.SpatialZeroPadding(21,21,21,21))
      		-- -- model:add(nn.SpatialZeroPadding(7,7,7,7))
		-- model:add(Convolution(2,6,5,5,1,1,0,0))
		-- model:add(SBatchNorm(6))
		-- model:add(ReLU(true))
		-- model:add(DilatedMax(3,3,1,1,0,0,1,1))
 
		-- model:add(DilatedConv(6,16,5,5,1,1,0,0,3,3)) -- Dilated 3
		-- -- model:add(Convolution(6,16,5,5,1,1,0,0)) -- Dilated 3
		-- model:add(SBatchNorm(16))
		-- model:add(ReLU(true))
		-- model:add(DilatedMax(3,3,1,1,0,0,3,3)) -- Dilated 3
		
		-- model:add(DilatedConv(16,120,3,3,1,1,0,0,9,9)) -- Dilated 3x3
		-- -- model:add(Convolution(16,120,3,3,1,1,0,0)) -- Dilated 3
		-- model:add(SBatchNorm(120))
		-- model:add(ReLU(true))
		
		-- model:add(Convolution(120,1,1,1,1,1,0,0)) -- fully connected
		-- model:add(nn.Sigmoid())

 -- #############################################################################
		-- ORIGINAL

      		-- model:add(nn.SpatialZeroPadding(13,13,13,13))
      		-- model:add(Crop(27,27))
		-- model:add(Convolution(2,6, 5,5, 1,1, 2,2))
		-- -- model:add(SBatchNorm(6))
		-- model:add(ReLU(true))
		-- model:add(Max(3,3,3,3))
 
		-- model:add(Convolution(6,16, 5,5, 1,1, 2,2))
		-- -- model:add(SBatchNorm(16))
		-- model:add(ReLU(true))
		-- model:add(Max(3,3,3,3))
		
		-- model:add(Convolution(16,120, 3,3, 1,1, 0,0)) -- 1st fully connected
		-- -- model:add(SBatchNorm(120))
		-- model:add(ReLU(true))
		
		-- model:add(Convolution(120,1, 1,1, 1,1, 0,0)) -- 2nd fully connected
		-- model:add(nn.Sigmoid())


   else
      error('invalid dataset: ' .. opt.dataset)
   end

   -- box-muller transform for normmally distributed pseudo-random numbers
   local function gaussian(mu, sigma2)
      local epsilon = 1e-320
      local u1, u2 = 0.0,0.0
      while u1 < epsilon do
         u1, u2 = math.random(), math.random()
      end
      local z0 = math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
      return z0*math.sqrt(sigma2) + mu
   end

   local function ConvInit(name)
      local modules = model:findModules(name)
      for k,v in pairs(modules) do
         local n = v.kW*v.kH*v.nOutputPlane
         -- v.weight:apply(function(x)
         --         local i = math.huge
		 -- while i > 0.2 do
         --            i = gaussian(0, 0.01)
         --         end
         --         return i
         --         end)
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 and k < #modules then
            v.bias = nil
            v.gradBias = nil
         else
            -- v.bias:fill(0.1)
            v.bias:fill(0)
         end
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
	   local kernel1d = torch.cat(torch.linspace(1,factor,factor),
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

local function replaceLayer(original, new)
   original_nodes, container_nodes = model:findModules(original)
   for i = 1, #threshold_nodes do
     -- Search the container for the current threshold node
      for j = 1, #(container_nodes[i].modules) do
         if container_nodes[i].modules[j] == threshold_nodes[i] then
            -- Replace with a new instance
            container_nodes[i].modules[j] = new
         end
      end
   end
end


   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   -- all architctures but cdnet begin with conv
   -- cudnn conv first layer should ignore gradInput
   -- model:get(1).gradInput = nil

   return model
end

return createModel
