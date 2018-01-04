#1 - erro no maxunpooling
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/lenet5/deconv/ch-1/better-init/linear-deconv/BN-layer/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Thu-Jan-26-21:12:31-2017/resume/model_97.t7' -trials 60

#2
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/cifar-10/deconv/ch-1/linear-deconv/bad-init/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,shortcutType=B,weightDecay=0.0002/Tue-Jan-24-13:00:13-2017/model_60.t7' -trials 60

#3
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/cifar-10/deconv/ch-1/non-linear-deconv/bad-init/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,shortcutType=B,weightDecay=0.0002/Tue-Jan-24-22:11:42-2017/model_59.t7' -trials 60

#4
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/imagenet/deconv/linear-deconv/,LR=0.01,base=imagenet,batchSize=16,dataset=cdnet,depth=34,nEpochs=60,optimizer=adam,weightDecay=0.0002/Fri-Feb-10-09:22:14-2017/model_58.t7'  -trials 60

#5
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/imagenet/deconv/non-linear-deconv/,LR=0.01,base=imagenet,batchSize=16,dataset=cdnet,depth=34,nEpochs=60,optimizer=adam,weightDecay=0.0002/Fri-Jan-27-01:04:18-2017/model_60.t7' -trials 60

#6
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/imagenet/deconv/non-linear-deconv/,LR=0.01,base=imagenet,batchSize=16,dataset=cdnet,depth=34,nEpochs=60,optimizer=adam,weightDecay=0.0002/Thu-Jan-26-21:23:51-2017/model_60.t7' -trials 60

#7
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/imagenet/deconv/non-linear-deconv/,LR=0.01,base=imagenet,batchSize=16,dataset=cdnet,depth=50,nEpochs=60,optimizer=adam,weightDecay=0.0002/Thu-Feb--9-19:41:31-2017/model_57.t7' -trials 60

#8
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/imagenet/deconv/linear-deconv/,LR=0.01,base=imagenet,batchSize=16,dataset=cdnet,depth=50,nEpochs=60,optimizer=adam,weightDecay=0.0002/Thu-Feb--9-12:55:44-2017/model_60.t ch' -trials 60

#9
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/lenet5/dilation/ch-1/BN-layer/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Wed-Jan-25-00:11:07-2017/model-60.t7' -trials 60

#10
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/cifar-10/bilinear-upsample/ch1/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Wed-Jan-25-19:50:57-2017/resume/model_97.t7' -trials 60

#11
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/cifar-10/dilated/ch-1/,LR=0.01,base=cifar10,batchSize=16,dataset=cdnet,depth=34,nEpochs=60,optimizer=adam,weightDecay=0.0002/Fri-Jan-27-14:07:46-2017/resume/model_97.t7' -trials 60

#12
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/cifar-10/dilated/ch-1/,LR=0.01,base=cifar10,batchSize=8,dataset=cdnet,depth=58,nEpochs=60,optimizer=adam,weightDecay=0.0002/Tue-Jan-31-00:33:22-2017/model_54.t7' -trials 60

#13
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/imagenet/dilated/ch-1/,LR=0.01,base=imagenet,batchSize=16,dataset=cdnet,depth=34,nEpochs=60,optimizer=adam,weightDecay=0.0002/Sun-Jan-29-04:07:19-2017/model_55.t7' -trials 60

#14
CUDA_VISIBLE_DEVICES=1 th tune-threshold.lua -data '/home/lpcinelli/Documents/cdnet2014/braham-split-70-30' -dataset 'cdnet' -batchSize 32 -nThreads 2 -model 'checkpoints/resnet/imagenet/dilated/ch-1/pretrained/,LR=0.01,base=imagenet,batchSize=16,dataset=cdnet,depth=34,nEpochs=60,optimizer=adam,weightDecay=0.0002/Sat-Feb-11-15:15:25-2017/model_60.t7' -trials 60


