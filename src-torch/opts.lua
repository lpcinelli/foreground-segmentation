--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',    'imagenet', 'Options: imagenet | cifar10 | cifar100 | cdnet | sanity | test-hdf5')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        1, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-checkpoint',      'true',        'Save model after each epoch: true | false (true)')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   cmd:option('-trash',           'false',       'Discard all log and checkpoint: true | false (false)')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.1,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   cmd:option('-model_init_LR',     -1,  'Define a small LR to init model for 2 epochs. If it is below 0, ignored. (-1)')
   cmd:option('-LR_step',   -1,    'Define number of epochs between each LR decay')
   cmd:option('-LR_factor', -1,    'Define factor by which LR will be decayed')
   cmd:option('-optimizer',     'sgd',   'Optimizer algorithm: sgd | adam | rmsprop (sgd)')
   cmd:option('-alpha',       	0.99,    'RMSProp optimizer param alpha')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'resnet', 'Options: resnet | preresnet | lenet5')
   cmd:option('-depth',        0,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-shortcutType', 'B',      'Options: A | B | C')
   cmd:option('-base',         '',       'Base on the archi for: cifar10 | cifar100 | imagenet')
   cmd:option('-params',       'none',   'Path to model from which to get the params')
   cmd:option('-retrain',      'none',   'Path to model to retrain with')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         0,      'Number of classes in the dataset')
   cmd:option('-patchSize',        0,      'Patch size for pixel evaluation during training')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'
   opt.trash = opt.trash ~= 'false'
   opt.LR_step = opt.LR_step == -1 and 10 or opt.LR_step
   opt.LR_factor = opt.LR_factor == -1 and 0.5 or opt.LR_factor
   
   -- set folder name to save model checkpoints
   if opt.trash == false then
      if opt.resume ~= 'none' then
         opt.save = paths.concat(paths.dirname(opt.resume),'resume/')
      else
         opt.save = paths.concat(opt.save,
         cmd:string('', opt,
         {netType=true, optimState=true, gen=true, manualSeed=true,
         nThreads=true, checkpoint=true, data=true, retrain=true,
         save=true, shareGradInput=true, optnet=true, tenCrop=true,
         testOnly=true, resetClassifier=true,nClasses=true, trash=true, 
         params=true, resume=true}))
         -- add date/time
         opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ','-'))
      end
      if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
         cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
      end
   end

   if opt.dataset == 'imagenet' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing ImageNet data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
      end
      -- Default shortcutType=B and nEpochs=90
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs

   elseif opt.dataset == 'cifar10' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs

   elseif opt.dataset == 'cifar100' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs

   elseif opt.dataset == 'cdnet' or opt.dataset == 'sanity' or opt.dataset == 'test-hdf5' then
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 120 or opt.nEpochs
      opt.tenCrop = 'false'
      opt.patchSize = opt.patchSize == 0 and 33 or opt.patchSize
      -- opt.batchSize = opt.batchSize == 32 and 100 or opt.batchSize
      -- opt.depth = opt.depth == 34 and 33 or opt.depth

   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   -- set optimizer option
   print(" Use ".. opt.optimizer .. ' as the optimizer')

   return opt
end

return M
