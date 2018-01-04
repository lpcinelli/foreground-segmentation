--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'dpnn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local plotting = require 'plotting'
local graph = require 'drawnet' 

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- opt.shortcutType == '' and 'A' or opt.shortcutType

-- Logger
-- local logger = optim.Logger(paths.concat(opt.save,'training.log'))
-- local lossLogger = optim.Logger(paths.concat(opt.save,'loss.log'))

local logger = optim.Logger( opt.trash == false and paths.concat(opt.save,'training.log') or nil)
local lossLogger = optim.Logger(  opt.trash == false and paths.concat(opt.save,'loss.log') or nil)
logger:setNames{"Epoch", "Train F1", 'Val F1', "Train Prec", "Testing Prec",
					"Train Rec", "Test Rec", "Train Acc", "Test Acc",
					"Train IoU", "Test IoU", "Train Spec", "Val Spec",
					"Train FN", "Test FN", "Train FP", "Val FP",
					"Train Loss", "Val Loss",
                                        "Train TP", "Val TP",
					"Train FP", "Val FP",
					"Train TN", "Val TN",
					"Train FN", "Val FN"}
-- The trainer handles the training loop and evaluation n validation set
local trainer = Trainer(model, criterion, opt, optimState, lossLogger)

if opt.testOnly then
   local pixelCountVal, metricsVal = table.unpack(trainer:test(0, valLoader))
   print((' * Validation (batch avg):: Loss %.3f  Prec %.3f  Rec %.3f  f1: %.3f  FNR: %.3f  FPR: %.3f'):format(
                metricsVal['loss'],
                metricsVal['prec'],
                metricsVal['rec'],
                metricsVal['f1'], 
                metricsVal['fnr'],
                metricsVal['fpr'] ))

   return
end

local trainingStats = { testLoss={}, trainLoss={},
			testF1={}, trainF1={},
			testPrec={}, trainPrec={},
			testRec={}, trainRec={},
			testAcc={}, trainAcc={},
			testFnRate={}, trainFnRate={},
			testFpRate={}, trainFpRate={},
			testSpecificity={}, trainSpecificity={},
			-- testClassiErr={}, trainClassiErr={},
			testIoU={}, trainIoU={} }

local lossHistory

-- local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local startEpoch = opt.epochNumber
local bestF1 = 0
local bestAcc = 0
local bestRec = 0
local bestPrec = 0
local bestIoU = 0
local bestLoss = 0
local bestSpecificity = 0
local bestFpRate = 0
local bestFnRate = 0
-- local bestClassiErr = 0
local bestEpoch = 0
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   -- local trainF1, trainAcc, trainRec, trainPrec, trainIoU, trainLoss, trainSpecificity, trainFnRate, trainFpRate, losses = trainer:train(epoch, trainLoader)
   -- print((' * Training:: loss: %1.4f  Prec %.3f  Rec %.3f  f1: %.3f  FNR: %.3f  FPR: %.3f'):format(
   --       trainLoss, trainPrec, trainRec, trainF1, trainFnRate, trainFpRate))

   local pixelCountTrain, metricsTrain = table.unpack(trainer:train(epoch, trainLoader))
   -- local metricsTrain = trainer:train(epoch, trainLoader)
   print((' * Training (batch avg):: Loss %.3f  Prec %.3f  Rec %.3f  f1: %.3f  FNR: %.3f  FPR: %.3f'):format(
   		metricsTrain['loss'],
                metricsTrain['prec'],
                metricsTrain['rec'], 
                metricsTrain['f1'],  
                metricsTrain['fnr'], 
                metricsTrain['fpr'] ))

   -- Run model on validation set
   local pixelCountVal, metricsVal = table.unpack(trainer:test(epoch, valLoader))
   -- print((' * Testing (mini-batch avg):: loss: %1.4f  Prec %.3f  Rec %.3f  f1: %.3f  FNR: %.3f  FPR: %.3f'):format(
   -- 		metricsVal['loss'], metricsVal['prec'],
		-- metricsVal['rec'], metricsVal['f1'],
		-- metricsVal['fnr'], metricsVal['fpr']))
   
   print((' * Validation (batch avg):: Loss %.3f  Prec %.3f  Rec %.3f  f1: %.3f  FNR: %.3f  FPR: %.3f'):format(
                metricsVal['loss'],
                metricsVal['prec'],
                metricsVal['rec'],
                metricsVal['f1'], 
                metricsVal['fnr'],
                metricsVal['fpr'] ))


   if opt.trash == false then
	   -- Update training stats
	   table.insert(trainingStats.trainF1, metricsTrain['f1'])
	   table.insert(trainingStats.testF1, metricsVal['f1'])

	   -- table.insert(trainingStats.trainAcc, trainAcc)
	   -- table.insert(trainingStats.testAcc, metricsVal['acc'])

	   table.insert(trainingStats.trainPrec, metricsTrain['prec'])
	   table.insert(trainingStats.testPrec, metricsVal['prec'])

	   table.insert(trainingStats.trainRec, metricsTrain['rec'])
	   table.insert(trainingStats.testRec, metricsVal['rec'])

	   table.insert(trainingStats.trainIoU, metricsTrain['iou'])
	   table.insert(trainingStats.testIoU, metricsVal['iou'])

	   table.insert(trainingStats.trainFnRate, metricsTrain['fnr'])
	   table.insert(trainingStats.testFnRate, metricsVal['fnr'])

	   table.insert(trainingStats.trainFpRate, metricsTrain['fpr'])
	   table.insert(trainingStats.testFpRate, metricsVal['fpr'])

	   -- table.insert(trainingStats.trainClassiErr, trainClassiErr)
	   -- table.insert(trainingStats.testClassiErr, testClassiErr)

	   table.insert(trainingStats.trainSpecificity, metricsTrain['spec'])
	   table.insert(trainingStats.testSpecificity, metricsVal['spec'])

	   table.insert(trainingStats.trainLoss, metricsTrain['loss'])
	   table.insert(trainingStats.testLoss, metricsTrain['loss'])

	   if lossHistory == nil then
	      lossHistory = losses
	   else
	      lossHistory = torch.cat(lossHistory, losses, 1)
	   end

	   -- Update logger
	   logger:add{tostring(epoch), metricsTrain['f1'],   metricsVal['f1'], 
	                               metricsTrain['prec'], metricsVal['prec'],
	                               metricsTrain['rec'],  metricsVal['rec'],
	                               metricsTrain['acc'],  metricsVal['acc'],
	                               metricsTrain['iou'],  metricsVal['iou'],
	                               metricsTrain['spec'], metricsVal['spec'],
	                               metricsTrain['fnr'],  metricsVal['fnr'],
	                               metricsTrain['fpr'],  metricsVal['fpr'],
	                               metricsTrain['loss'], metricsVal['loss'],
                                       pixelCountTrain['TP'],pixelCountVal['TP'],
                                       pixelCountTrain['FP'],pixelCountVal['FP'],
                                       pixelCountTrain['TN'],pixelCountVal['TN'],
                                       pixelCountTrain['FN'],pixelCountVal['FN'],
				 }
					 -- trainClassiErr, testClassiErr, trainLoss, testLoss}

	   -- Plot stat curves for this epoch
	   -- plotting.loss_curve(lossHistory[{ {}, {1} }]:squeeze(), opt)
	   -- plotting.loss_curve(lossHistory:squeeze(), opt)
	   plotting.curve(trainingStats.trainF1, trainingStats.testF1, 'F1 score','f1', opt, 'f1', 1)
	   -- plotting.curve(trainingStats.trainAcc, trainingStats.testAcc, 'Acc','acc', opt, 'acc', 1)
	   -- plotting.curve(trainingStats.trainRec, trainingStats.testRec, 'Recall','recall', opt, 'recall', 1)
	   -- plotting.curve(trainingStats.trainIoU, trainingStats.testIoU, 'IoU','IoU', opt, 'IoU', 1)
	   -- plotting.curve(trainingStats.trainPrec, trainingStats.testPrec, 'Precision','prec', opt, 'prec', 1)
	   -- plotting.curve(trainingStats.trainSpecificity, trainingStats.testSpecificity, 'Specificity','spec', opt, 'spec', 1)
	   -- plotting.curve(trainingStats.trainClassiErr, trainingStats.testClassiErr, 'Classification Err','classErr', opt, 'classErr', 1)
	   -- plotting.curve(trainingStats.trainFpRate, trainingStats.testFpRate, 'FP Rate','FP', opt, 'FP', 1)
	   -- plotting.curve(trainingStats.trainFnRate, trainingStats.testFnRate, 'FN Rate','FN', opt, 'FN', 1)
	   -- plotting.curve(trainingStats.trainLoss, trainingStats.testLoss, 'Loss curve','loss', opt, 'loss', torch.Tensor(trainingStats.testLoss):max())

	   local bestModel = false
	   if metricsVal['f1'] > bestF1 then
	      bestModel = true
	      bestF1 = metricsVal['f1']
	      bestRec = metricsVal['rec']
	      bestPrec = metricsVal['prec']
	      bestLoss = metricsVal['loss'] 
	      bestFnRate = metricsVal['fnr']
	      bestFpRate = metricsVal['fpr']
	      bestSpecificity = metricsVal['spec']
	      bestIoU = metricsVal['iou']
	      -- bestAcc = metricsVal['acc']
	      bestEpoch = epoch
              print(string.format(' * Best model:: Epoch %d  Loss %1.4f  Prec %.3f  Rec %.3f  f1: %.3f  FN: %.3f  ',
                                    bestEpoch, bestLoss, bestPrec, bestRec, bestF1, bestFnRate))
	   end

	   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
	end
end

logger:add{}
logger:add{tostring(bestEpoch), 'nil' , bestF1, 'nil',
           bestPrec, 'nil', bestRec, 'nil',
             -- bestAcc, 'nil', bestIoU, 'nil',
           bestIoU, 'nil',
           bestSpecificity, 'nil', bestFnRate, 'nil',
           bestFpRate,  'nil', bestLoss}

print(string.format(' * Finished:: Epoch %d  Loss %1.4f  Prec %.3f  Rec %.3f  f1: %.3f  FN: %.3f  FP: %.3f  ',
                       bestEpoch, bestLoss, bestPrec, bestRec, bestF1, bestFnRate, bestFpRate))

if opt.trash == false then
   local input = torch.CudaTensor(1,2,192,256)
   graph.DrawModel(model, input, opt.save .. '/' .. opt.netType)
end

