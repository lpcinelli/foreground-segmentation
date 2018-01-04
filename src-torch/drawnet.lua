require 'nn';
require 'cunn';
require 'cudnn';
local generateGraph = require 'optnet.graphgen'

local modelPath = 'checkpoints/lenet5/dilation/ch-1/BN-layer/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Wed-Jan-25-00:11:07-2017/model_60.t7'
local modelname = 'lenet5-dilation-2'

local M = {}

-- visual properties of the generated graph
-- follows graphviz attributes
local graphOpts = {
  displayProps =  {shape='ellipse',fontsize=14, style='solid'},
  -- nodeData = function(oldData, tensor)
    --return oldData .. '\n' .. 'Size: '.. tensor:numel()
    -- local text_sz = ''
    -- for i = 1,tensor:dim() do
    --   if i == 1 then
    --     text_sz = text_sz .. '' .. tensor:size(i)
    --   else
    --     text_sz = text_sz .. ', ' .. tensor:size(i)
    --   end
    -- end
    -- return oldData
    -- return oldData .. '\n' .. 'Size: {'.. text_sz .. '}\n' .. 'Mem size: ' .. tensor:numel()
  -- end
}

local function copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   local input = torch.CudaTensor()
   --print('type of input: ' .. torch.type(input))
   --print(sample:size())
   input:resize(sample:size()):copy(sample)

   return input
end

function M.DrawModel(model, input, name)
  -- model: A network architecture
  -- input: The input for the given network architecture
  -- name:  The model name (string).
  --        The files, '<name>.dot' and '<name>.svg' will be generated.
  local input_
  if torch.type(input) == 'table' then
      input_ = {}
      --print('table: ', #input)
      for i = 1,#input do
        input_[i] = copyInputs(input[i])
        --print(torch.type(input_[i]))
      end
  else
    input_ = copyInputs(input)
    --print(torch.type(input_))
  end

  g = generateGraph(model, input_, graphOpts)
  graph.dot(g, name, name)

  --print(torch.type(g))
  --print(g)
  --print(#g.nodes)
  --print(g.nodes[#g.nodes]:label())
  --print(g:leaves())

  return g
end

-- local model = torch.load(modelPath)
-- model:cuda():evaluate()
-- local input = torch.CudaTensor(2,256,192):normal(0,0.1)
-- drawModel(model, input, modelname)

return M
