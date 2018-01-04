--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local image = require 'image'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState, logger1)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
      -- Adam parameters
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 1e-8,
      -- Rmsprop parameter
      alpha = opt.alpha,
   }
   print('alpha = ', self.optimState.alpha)
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   self.logger = logger1
   self.logger:setNames{'Epoch', 'Iter', 'Loss', 'Update Scale'}
end

function Trainer:train(epoch, dataloader, threshold)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local timeCounter = 0.0
   local printFreq = 50
   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local lossSum = 0.0
   -- local f1Sum, lossSum, precisionSum, recallSum, accuracySum, IoUSum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
   -- local specificitySum, fnRateSum, fpRateSum, classiErrSum = 0.0, 0.0, 0.0, 0.0
   local TPSum, TNSum, FPSum, FNSum = 0.0, 0.0, 0.0, 0.0
   local f1, precision, recall, acc, IoU, specificity, fnRate, fpRate 
   local N = 0
   local stats = torch.Tensor(trainSize, 1)

   -- print(string.format('=> Training epoch #%d  lr=%.3e', epoch, self.optimState.learningRate))

   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      -- print(model)
      -- print('\n\ninput:')
      -- print(self.input:size())
      -- print('\n\noutput:')
      -- print(output:size())
      -- print('\ntarget:')
      -- print(self.target:size())
      local batchSize = output:size(1)
      -- local crop = self.model:findModules('nn.SpatialUniformCrop')[1]
      -- local targetPatchCenter = self:getCenter(crop.coord)
      
      -- local loss = self.criterion:forward(self.model.output, self.targetPatchCenter)
      local loss = self.criterion:forward(self.model.output, self.target)
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      -- self.criterion:backward(self.model.output, self.targetPatchCenter)
      self.model:backward(self.input, self.criterion.gradInput)

      local paramScale = torch.norm(self.params)

      if self.opt.optimizer == 'adam' then
         optim.adam(feval, self.params, self.optimState)
      elseif self.opt.optimizer == 'rmsprop' then
         optim.rmsprop(feval, self.params, self.optimState)
      else
         optim.sgd(feval, self.params, self.optimState)
      end

      paramScale = torch.abs(paramScale - torch.norm(self.params))/paramScale

      local basicCounting, metrics = table.unpack(self:computeScore(output, sample.target, 1, nil, nil, threshold))
      
      TPSum = TPSum + basicCounting[1]
      TNSum = TNSum + basicCounting[2]
      FNSum = FNSum + basicCounting[3]
      FPSum = FPSum + basicCounting[4]

      f1, precision, recall, acc, IoU, specificity, fnRate, fpRate = self:computeMetrics(TPSum, TNSum, FNSum, FPSum)
      local f1Batch, precisionBatch, recallBatch, accBatch, IoUBatch, specificityBatch, fnRateBatch, fpRateBatch = table.unpack(metrics)
      -- local f1, precision, recall, acc, IoU, specificity, fnRate, fpRate = self:computeScore(output, sample.target, 1, n, epoch, threshold)
      -- local f1, precision, recall, acc, IoU, specificity, fnRate, fpRate = self:computeScore(output, targetPatchCenter, 1, n, epoch)

      -- f1Sum = f1Sum + f1*batchSize
      lossSum = lossSum + loss*batchSize
      -- precisionSum = precisionSum + precision*batchSize
      -- recallSum = recallSum + recall*batchSize
      -- accuracySum = accuracySum + acc*batchSize
      -- IoUSum = IoUSum + IoU*batchSize
      -- specificitySum = specificitySum + specificity*batchSize
      -- fnRateSum = fnRateSum + fnRate*batchSize
      -- fpRateSum = fpRateSum + fpRate*batchSize
      -- classiErrSum = classiErrSum + classiErr*batchSize
      N = N + batchSize
      -- timeCounter  = timeCounter + timer:time().real

      if n%printFreq == 0 then
         print((' | Epoch: [%d][%d/%d]    Time %.3f Loss %1.4f (%1.4f)  Prec %.3f (%.3f)  \z
      					  Rec %.3f  (%.3f)  IoU %.3f  (%.3f)  FNR %.3f  (%.3f)  FPR %.3f  (%.3f)  \z
      			  		  f1 %.3f (%.3f)  '):format(epoch, n, trainSize, timer:time().real,
      					  loss, lossSum/N, precisionBatch, precision, recallBatch, recall,
      					  IoUBatch, IoU, fnRateBatch, fnRate, fpRateBatch, fpRate, f1Batch, f1))
         -- timer:reset()
         -- dataTimer:reset()
         -- print(timeCounter/1000)
         -- timeCounter = 0.0
      end

         timer:reset()
         dataTimer:reset()

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())
      -- timer:reset()
      -- dataTimer:reset()
      self.logger:add{tostring(epoch), tostring(n), loss, paramScale}
      -- stats[n] = torch.Tensor({f1, f1Sum/N, precision, precisionSum/N, recall, recallSum/N, loss, lossSum/N})
      stats[n] = torch.Tensor({loss})
   end
   return {{ TP=TPSum, TN=TNSum, FN=FNSum, FP=FPSum}, {f1=f1, acc=acc, rec=recall, prec=precision, iou=IoU, loss=lossSum/N, spec=specificity, fnr=fnRate, fpr=fpRate}}
end

function Trainer:test(epoch, dataloader, threshold)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local lossSum = 0.0
   -- local f1, lossSum, precision, recall, acc, IoU = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
   -- local specificity, fnRate, fpRate = 0.0, 0.0, 0.0
   local TPSum, TNSum, FPSum, FNSum = 0.0, 0.0, 0.0, 0.0
   local f1, precision, recall, acc, IoU, specificity, fnRate, fpRate
   local timeCounter = 0.0
   local N = 0
   local printFreq = 50
   -- local patchSize = opt.patchSize

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      -- image.save(string.format(self.opt.save .. '/imgs/%d.jpg',n), output[1])

      -- local crop = self.model:findModules('nn.SpatialUniformCrop')[1]
      -- local targetPatchCenter = self:getCenter(crop.coord)

      -- local loss = self.criterion:forward(self.model.output, self.targetPatchCenter)
      local loss = self.criterion:forward(self.model.output, self.target)

      local basicCounting, metrics = table.unpack(self:computeScore(output, sample.target, 1, nil, nil, threshold))
      -- local f1, precision, recall, acc, IoU, specificity, fnRate, fpRate = table.unpack(metrics)
      -- local basicCounting, metrics = self:computeScore(output, targetPatchCenter, 1)
      -- local f1, precision, recall, acc, IoU, specificity, fnRate, fpRate = self:computeScore(output, sample.target, 1, nil, nil, threshold)

      TPSum = TPSum + basicCounting[1]
      TNSum = TNSum + basicCounting[2]
      FNSum = FNSum + basicCounting[3]
      FPSum = FPSum + basicCounting[4]
      
      f1, precision, recall, acc, IoU, specificity, fnRate, fpRate = self:computeMetrics(TPSum, TNSum, FNSum, FPSum)
      local f1Batch, precisionBatch, recallBatch, accBatch, IoUBatch, specificityBatch, fnRateBatch, fpRateBatch = table.unpack(metrics)
      -- print(('TPSum %d  TNSum %d  FNSum %d  FPSum %d' ):format(TPSum, TNSum, FNSum, FPSum))
      
      -- f1Sum = f1Sum + f1*batchSize
      lossSum = lossSum + loss*batchSize
      -- precisionSum = precisionSum + precision*batchSize
      -- recallSum = recallSum + recall*batchSize
      -- accuracySum = accuracySum + acc*batchSize
      -- IoUSum = IoUSum + IoU*batchSize
      -- specificitySum = specificitySum + specificity*batchSize
      -- fnRateSum = fnRateSum + fnRate*batchSize
      -- fpRateSum = fpRateSum + fpRate*batchSize
      -- classiErrSum = classiErrSum + classiErr*batchSize
      -- top5Sum = top5Sum + top5*batchSize
      N = N + batchSize
      -- timeCounter  = timeCounter + timer:time().real
      if n % printFreq == 0 then
         -- print(('Time %.5f  Data %.5f'):format(timeCounter/1000, dataTime))
         print((' | Validation: [%d][%d/%d]    Time %.3f Loss %1.4f (%1.4f)  Prec %.3f (%.3f)  \z
      					  Rec %.3f  (%.3f)  IoU %.3f  (%.3f)  FNR %.3f  (%.3f)  FPR %.3f  (%.3f)  \z
      			  		  f1 %.3f (%.3f)  '):format(epoch, n, size, timer:time().real,
      					  loss, lossSum/N, precisionBatch, precision, recallBatch, recall,
      					  IoUBatch, IoU, fnRateBatch, fnRate, fpRateBatch, fpRate, f1Batch, f1))
         -- print((' | Validation: [%d][%d/%d] Time %.3f  loss %.3f   prec %.3f  rec %.3f  FNR %.3f  FPR %.3f  \z
   					-- f1 %.3f'):format(epoch, n, size,
					-- timer:time().real/printFreq,
   					-- loss, precisionBatch, recallBatch,
   					-- fnRateBatch, fpRateBatch, f1Batch))
         -- print((' | Per batch: [%d][%d/%d]  Prec %.3f  Rec %.3f  FNR %.3f  FPR %.3f  \z
   					-- f1 %.3f'):format(epoch, n, size,
   					-- precisionSum/N, recallSum/N,
   					-- fnRateSum/N, fpRateSum/N, f1Sum/N))
         -- print((' | Test1: [%d][%d/%d]    Time %.6f  Data %.6f  Loss %1.4f (%1.4f)  Prec %.3f (%.3f)  \z
   					-- Rec %.3f  (%.3f)  IoU %.3f  (%.3f)  FNR %.3f  (%.3f)  FPR %.3f  (%.3f)  \z
   					-- f1 %.3f (%.3f)  '):format(epoch, n, size, timer:time().real, dataTime,
   					-- loss, lossSum/N, precision, precisionSum/N, recall, recallSum/N,
   					-- IoU, IoUSum/N, fnRate, fnRateSum/N, fpRate, fpRateSum/N, f1, f1Sum/N))
         -- print((' | Whole dataset: [%d][%d/%d]  Prec %.3f  Rec %.3f  FNR %.3f  FPR %.3f  \z
   					-- f1 %.3f'):format(epoch, n, size,
   					-- (TPSum/(TPSum+FPSum)),  (TPSum/(TPSum+FNSum)), (FNSum/(FNSum+TPSum)),
   					-- (FPSum/(FPSum+TNSum)), (2*TPSum/(2*TPSum+FNSum+FPSum))))
         -- timer:reset()
         -- dataTimer:reset()
         -- print(timeCounter/1000)
         -- timeCounter = 0.0
      end	

      timer:reset()
      dataTimer:reset()
      -- self.testLogger:add{tostring(epoch), tostring(n), f1, f1Sum/N, precision, precisionSum/N, recall, recallSum/N, loss, lossSum/N }
      -- stats[n] = torch.Tensor({f1, f1Sum/N, precision, precisionSum/N, recall, recallSum/N, loss, lossSum/N})
   end
   self.model:training()
   -- print(('TPSum %d  TNSum %d  FNSum %d  FPSum %d' ):format(TPSum, TNSum, FNSum, FPSum))
   return {{ TP=TPSum, TN=TNSum, FN=FNSum, FP=FPSum}, {f1=f1, acc=acc, rec=recall, prec=precision, iou=IoU, loss=lossSum/N, spec=specificity, fnr=fnRate, fpr=fpRate}}
end


function Trainer:computeMetrics(TP, TN, FN, FP)
   local acc = ((TP+TN)/(TP+FP+FN+TN))
   local prec = (TP/(TP+FP))
   local rec = (TP/(TP+FN))
   local fnr = (FN/(FN+TP))
   local fpr = (FP/(FP+TN))
   local f1 = (2*TP/(2*TP+FN+FP))
   local spec = (TN/(TN+FP))
   local iou = (TP/(TP+FP+FN))

   return f1, prec, rec, acc, iou, spec, fnr, fpr
end
-- new computerScore function used for segmentation task
function Trainer:computeScore(output, target, nCrops, n, epoch, threshold)
   -- threshold segmentation mask: is FG if prog >= 0.5
   threshold = threshold == nil and 0.5 or threshold
   output[output:ge(threshold)] = 1
   output[output:lt(threshold)] = 0
   -- local _, output = output2:max(2)
   -- output = output:float() - 1

   local targetArea = target:sum() -- TP + FN
   local outputArea = output:sum() -- TP + FP
   local intersection = torch.cmul(output,target):sum() -- TP
   local trueNeg = (output + target):eq(0):float():sum() -- TN
   
   local function treatNaN(a)
      -- -- if there is no foreground OUTPUT pixel: precision is NaN
      -- if torch.any(outputArea:eq(0)) then a[outputArea:eq(0)] = 0 end
      -- -- if there is no foreground TARGET pixel: recall is NaN
      -- if torch.any(targetArea:eq(0)) then a[targetArea:eq(0)] = 0 end
      -- -- if there is neither foreground OUTPUT pixel nor foreground TARGET pixel: precision and recall are NaN
      -- if torch.any(torch.cmul(outputArea:eq(0), targetArea:eq(0))) then
      --    a[torch.cmul(outputArea:eq(0), targetArea:eq(0))] = 1
      if outputArea == 0 and targetArea == 0 then
      	a = 1
      elseif outputArea == 0 or targetArea == 0 then
      	a = 0
      end
      return a
   end

   local function precision() -- TP/(TP+FP)
      -- return torch.cdiv(intersection, outputArea)
      return treatNaN(intersection/outputArea)
   end
   local function recall() -- TP/(TP+FN)
      -- return torch.cdiv(intersection, targetArea)
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
   local function specificity() -- TN/(TN+FP)
   	return trueNeg/(trueNeg + (outputArea - intersection))
   end
   local function falsePosRate() -- FP/(FP+TN)
   	return (outputArea - intersection)/((outputArea - intersection) + trueNeg)
   end
   local function falseNegRate() -- FN/(TP+FN)
	if targetArea == 0 then
	   return 0
	else
   	   return (targetArea - intersection)/targetArea
        end
   end
   -- local function classifErr() -- (FN+FP)/(TP+TN+FN+FP)
   -- 	return 1 - accuracy()
   -- end

   local recallVal = recall()
   local precisionVal = precision()
   local accuracyVal = accuracy()
   local IoUVal = IoU()
   local f1 = f1Direct()
   local specificityVal = specificity()
   local fnRateVal = falseNegRate()
   local fpRateVal = falsePosRate()
   -- local classiErrVal = classifErr()
   -- local precisionVal, recallVal, accuracyVal, IoUVal, specificityVal, fnRateVal, fpRateVal = 0, 0, 0, 0, 0, 0, 0
   return {{intersection, trueNeg, (targetArea - intersection), (outputArea - intersection)}, {f1, precisionVal, recallVal, accuracyVal, IoUVal, specificityVal, fnRateVal, fpRateVal}}

end

-- original computeScore() function used for clasification task
function Trainer:computeScore2(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end
function Trainer:getCenter(coord)
   local targetPatchCenter = torch.Tensor(coord:size(1),1,1,1)
   for i=1,coord:size(1) do
      targetPatchCenter[i]:copy(self.target[{ i, {} , {coord[{i, 1}]}, {coord[{i, 2}]} }])
   end
   self.targetPatchCenter = self.targetPatchCenter or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
   self.targetPatchCenter:resize(targetPatchCenter:size()):copy(targetPatchCenter)
   return targetPatchCenter
end
function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0

   if self.opt.model_init_LR > 0 and epoch < 6 then
      return self.opt.model_init_LR

   -- elseif self.opt.optimizer == 'adam' then
	-- sqrt decay used in the original adam paper: http://arxiv.org/abs/1412.6980
	-- decay = 1.0/math.sqrt(epoch)
	-- return self.opt.LR * decay

   -- elseif self.opt.dataset == 'imagenet' then
   --    decay = math.floor((epoch - 1) / self.opt.LR_decay_step)

   -- elseif self.opt.dataset == 'cifar10' then
   --    decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0

   -- elseif self.opt.dataset == 'cifar100' then
   --    decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0

   elseif self.opt.dataset == 'cdnet' then
      decay = math.floor((epoch - 1) /self.opt.LR_step)
      -- decay = epoch >=  and 2 or epoch >= 15 and 1 or 0
   end
   return self.opt.LR * math.pow(self.opt.LR_factor, decay)
end

return M.Trainer
