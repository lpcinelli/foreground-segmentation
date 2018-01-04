#CUDA_VISIBLE_DEVICES=1 th main.lua -dataset 'cdnet' -data '/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean' -nGPU 1 -nEpochs 90 -batchSize 16 -LR 0.01 -weightDecay 2e-4 -LR_step 10 -LR_factor 0.5 -optimizer 'adam' -nThreads 1 -netType 'lenet5' -save 'checkpoints/lenet5/bilinear-upsample/ch-1/no-BN-layer/full-base'

#CUDA_VISIBLE_DEVICES=1 th main.lua -dataset 'cdnet' -data '/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean' -nGPU 1 -nEpochs 90 -batchSize 16 -LR 0.01 -weightDecay 2e-4 -LR_step 10 -LR_factor 0.5 -optimizer 'adam' -nThreads 2 -netType 'resnet' -base 'cifar10' -shortcutType 'B' -depth 32 -save 'checkpoints/resnet/cifar-10/bilinear-upsample/ch1/full-base'

#CUDA_VISIBLE_DEVICES=1 th main.lua -dataset 'cdnet' -data '/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean' -nGPU 1 -nEpochs 90 -batchSize 16 -LR 0.01 -weightDecay 2e-4 -LR_step 10 -LR_factor 0.5 -optimizer 'adam' -nThreads 2 -netType 'resnet' -base 'imagenet' -shortcutType 'B' -depth 34 -save 'checkpoints/resnet/imagenet/bilinear-upsample/full-base'




#CUDA_VISIBLE_DEVICES=0 th category-perf.lua -dataset cdnet -data /local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean -nGPU 1 -batchSize 32 -nThreads 2 -model checkpoints/resnet/cifar-10/deconv/ch-1/non-linear-deconv/better-init/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=90,optimizer=adam,weightDecay=0.0002/Tue-Apr--4-23:59:21-2017/resume/model_78.t7

#CUDA_VISIBLE_DEVICES=0 th category-perf.lua -dataset cdnet -data /local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean -nGPU 1 -batchSize 32 -nThreads 2 -model checkpoints/resnet/cifar-10/decode/shallow/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=80,optimizer=adam,weightDecay=0.0002/Mon-Apr-10-14:24:56-2017/model_80.t7

#CUDA_VISIBLE_DEVICES=0 th category-perf.lua -dataset cdnet -data /local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean -nGPU 1 -batchSize 32 -nThreads 2 -model checkpoints/resnet/cifar-10/decode/deep/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=12,dataset=cdnet,depth=32,nEpochs=80,optimizer=adam,weightDecay=0.0002/Mon-Apr-10-14:33:08-2017/model_78.t7

CUDA_VISIBLE_DEVICES=1 th main.lua -dataset 'cdnet' -data '/local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean' -nGPU 1 -batchSize 32 -testOnly true -retrain checkpoints/resnet/cifar-10/bilinear-upsample/ch1/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=90,optimizer=adam,weightDecay=0.0002/Fri-Mar-31-18:45:23-2017/resume/model_75.t7 -trash true

CUDA_VISIBLE_DEVICES=1 th main.lua -dataset 'cdnet' -data '/local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean' -nGPU 1 -batchSize 32 -testOnly true -retrain checkpoints/resnet/cifar-10/deconv/ch-1/linear-deconv/better-init/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=90,optimizer=adam,weightDecay=0.0002/Fri-Mar-31-20:49:15-2017/model_80.t7 -trash true

CUDA_VISIBLE_DEVICES=1 th main.lua -dataset 'cdnet' -data '/local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean' -nGPU 1 -batchSize 32 -testOnly true -retrain checkpoints/resnet/cifar-10/deconv/ch-1/non-linear-deconv/better-init/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=90,optimizer=adam,weightDecay=0.0002/Tue-Apr--4-23:59:21-2017/resume/model_78.t7 -trash true

CUDA_VISIBLE_DEVICES=1 th main.lua -dataset 'cdnet' -data '/local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean' -nGPU 1 -batchSize 32 -testOnly true -retrain checkpoints/resnet/cifar-10/decode/shallow/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=80,optimizer=adam,weightDecay=0.0002/Mon-Apr-10-14:24:56-2017/model_80.t7 -trash true

CUDA_VISIBLE_DEVICES=1 th main.lua -dataset 'cdnet' -data '/local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean' -nGPU 1 -batchSize 32 -testOnly true -retrain checkpoints/resnet/cifar-10/decode/deep/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=12,dataset=cdnet,depth=32,nEpochs=80,optimizer=adam,weightDecay=0.0002/Mon-Apr-10-14:33:08-2017/model_78.t7 -trash true
