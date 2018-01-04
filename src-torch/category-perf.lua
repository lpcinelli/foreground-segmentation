require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

function copyInputs(sample, opt)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   inputImg = inputImg or (opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   targetImg = targetImg or (opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())

   inputImg:resize(sample.input:size()):copy(sample.input)
   targetImg:resize(sample.target:size()):copy(sample.target)
end

-- new computerScore function used for segmentation task
function computeScore(output, target, threshold)
   -- threshold segmentation mask: is FG if prog >= 0.5
   threshold = threshold == nil and 0.5 or threshold
   output[output:ge(threshold)] = 1
   output[output:lt(threshold)] = 0

   local targetArea = target:sum() -- TP + FN
   local outputArea = output:sum() -- TP + FP
   local intersection = torch.cmul(output,target):sum() -- TP
   local trueNeg = (output + target):eq(0):float():sum() -- TN

   local function treatNaN(a)
      if outputArea == 0 and targetArea == 0 then
      	a = 1
      elseif outputArea == 0 or targetArea == 0 then
      	a = 0
      end
      return a
   end

   local function precision() -- TP/(TP+FP)
      return treatNaN(intersection/outputArea)
   end
   local function recall() -- TP/(TP+FN)
      return treatNaN(intersection/targetArea)
   end
   local function f1Score(a, b)
      return treatNaN(2*torch.cdiv(torch.cmul(a, b), (a+b)))
   end
   local function f1Direct() -- 2*TP/(2*TP+FN+FP)
      return treatNaN(2*intersection/(outputArea + targetArea))
   end
   local function IoU() -- TP/(TP+FN+FP)
   	local IoUPerImage = torch.cdiv(	(output + target):eq(2):float():sum(3):sum(4), -- overlap regions have 1+1 values (output+target)
   					(output + target):ge(1):float():sum(3):sum(4)) -- union region is either 1 or 2 (output+target)
	local nanPos = IoUPerImage:ne(IoUPerImage)
	IoUPerImage[nanPos] = 0
	return IoUPerImage:sum()/(IoUPerImage:size(1) - nanPos:sum())
   end
   local function accuracy() -- (TP+TN)/(TP+TN+FN+FP)
   	return (intersection + trueNeg)/(target:view(-1):size(1))
   end
   local function specifity() -- TN/(TN+FP)
   	return trueNeg/(trueNeg + (outputArea - intersection))
   end
   local function falsePosRate() -- FP/(FP+TN)
   	return (outputArea - intersection)/((outputArea - intersection) + trueNeg)
   end
   local function falseNegRate() -- FN/(TP+FN)
   	return (targetArea - intersection)/targetArea
   end
   local function classifErr() -- (FN+FP)/(TP+TN+FN+FP)
   	return 1 - accuracy()
   end

   local recallVal = recall()
   local precisionVal = precision()
   local IoUVal = IoU()
   local f1 = f1Direct()
   local fnrVal = falseNegRate()

   -- specifity()
   -- falseNegRate()
   -- falsePosRate()
   -- classifErr()

   return f1, precisionVal, recallVal, fnrVal, IoUVal
end

function parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Threshold finetuning script')
   cmd:text('Options:')
   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',       '',      'Datase name')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-nThreads',        1,     'Number of data loading threads')
   cmd:option('-batchSize',       32,    'Mini-batch size (1 = pure stochastic)')
   cmd:option('-model',        'none',   'Path to model')
   cmd:option('-trials',        100,      'Nb of trials')
   cmd:text()

   local opt = cmd:parse(arg or {})

   if not paths.filep(opt.model) then
      cmd:error('error: unable to find model: ' .. opt.model .. '\n')
   end

   if not paths.dirp(opt.data) then
      cmd:error('error: unable to find path to dataset: ' .. opt.data .. '\n')
   end

   return opt
end

local opt = parse(arg)
-- torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)
opt.manualSeed = 0
opt.gen = 'gen/'

-- Load model
local model = torch.load(opt.model)
local criterion = nn.BCECriterion():cuda()

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- Logger
dirPath = paths.dirname(opt.model)
local logger = optim.Logger(paths.concat(dirPath,'threshold-tuning.log'))
logger:setNames{"Thres", "F1", "Prec", "Rec", "Fnr", "IoU"}
-- local dummyLogger = optim.Logger(nil)

-- The trainer handles the training loop and evaluation on validation set
-- local trainer = Trainer(model, criterion, opt, nil, dummyLogger)

model:cuda()
model:evaluate()

local thresholdList = torch.FloatTensor(opt.trials):random(20,50)/100
local stats = torch.FloatTensor(opt.trials,5):zero()
-- for trial = 1, opt.trials do

local size = valLoader:size()
local N = 0

for n, sample in valLoader:run() do

   print(string.format('Iter %d/%d', n, size))

   -- Copy input and target to the GPU
   copyInputs(sample, opt)

   local output = model:forward(inputImg):float()
   local batchSize = output:size(1)

   for i=1,thresholdList:size(1) do
      local f1, precision, recall, fnr, IoU = computeScore(output, sample.target, thresholdList[i])

      stats[i][1] = stats[i][1] + f1*batchSize
      stats[i][2] = stats[i][2] + precision*batchSize
      stats[i][3] = stats[i][3] + recall*batchSize
      stats[i][4] = stats[i][4] + fnr*batchSize
      stats[i][5] = stats[i][5] + IoU*batchSize
   end

   N = N + batchSize
end 

stats = stats/N

-- Update logger
for i=1,thresholdList:size(1) do
   local F1   =  stats[i][1]      
   local Prec =  stats[i][2]
   local Rec  =  stats[i][3]
   local Fnr  =  stats[i][4]
   local IoU  =  stats[i][5]
   logger:add{thresholdList[i], F1, Prec, Rec, Fnr, IoU}
end

local val, ind = torch.max(stats[{ {}, {1} }], 1)
ind = ind:squeeze()

logger:add{}
logger:add{thresholdList[ind], stats[ind][1], stats[ind][2], stats[ind][3], stats[ind][4], stats[ind][5]}
print(string.format(' * Finished:: Thres %.2f  F1 %.3f  Prec %.3f  Rec %.3f  Fnr %.3f  IoU %.3f',
                       thresholdList[ind], stats[ind][1], stats[ind][2], stats[ind][3], stats[ind][4], stats[ind][5]))

