require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

local t = require 'datasets/transforms'
-- datapath = '/local/home/lpcinelli/Documents/cdnet2014/distinct-videos-split/'
-- datapath = '/home/lpcinelli/Documents/sbrt2017/Figuras/input-output/igor-express/'
datapath = '/local/home/lpcinelli/Documents/cdnet2014/full-split-70-30-clean/'
split = 'val/'
-- img_num = '002000'

-- video_path = 'blue_box/'
-- img_num = 'frame_1787'
-- ref_num = 'frame_2050'

-- video_path = 'pink_bottle/'
-- img_num = 'frame_2541'
-- 'shadow/bungalows/groundtruth/gt001157.png'
-- ref_num = 'frame_2663'

-- video_path = 'white_jar/'
-- img_num = 'frame_2943'
-- ref_num = 'frame_3352'

inputPaths = {}
bgModelPaths = {}
ROIPaths = {}
modelPaths = {}
modelNames = {}
outputNames = {}

table.insert(inputPaths,   datapath .. 'val/baseline/highway/input/in001643.jpg')
table.insert(bgModelPaths, datapath .. 'background/baseline/highway/model/staticModel.jpg')
table.insert(ROIPaths,     datapath .. 'background/baseline/highway/ROI/ROI.jpg')
table.insert(outputNames, 'highway')

table.insert(inputPaths,   datapath .. 'val/lowFramerate/turnpike_0_5fps/input/in001126.jpg')
table.insert(bgModelPaths, datapath .. 'background/lowFramerate/turnpike_0_5fps/model/staticModel.jpg')
table.insert(ROIPaths,     datapath .. 'background/lowFramerate/turnpike_0_5fps/ROI/ROI.jpg')
table.insert(outputNames, 'turnpike')

table.insert(inputPaths,   datapath .. 'val/nightVideos/fluidHighway/input/in000834.jpg')
table.insert(bgModelPaths, datapath .. 'background/nightVideos/fluidHighway/model/staticModel.jpg')
table.insert(ROIPaths,     datapath .. 'background/nightVideos/fluidHighway/ROI/ROI.jpg')
table.insert(outputNames, 'fluidHighway')

table.insert(inputPaths,   datapath .. 'val/shadow/bungalows/input/in001656.jpg')
table.insert(bgModelPaths, datapath .. 'background/shadow/bungalows/model/staticModel.jpg')
table.insert(ROIPaths,     datapath .. 'background/shadow/bungalows/ROI/ROI.jpg')
table.insert(outputNames, 'bungalows')


-- table.insert(inputPaths,   datapath .. 'val/cameraJitter/sidewalk/input/in00XXX.jpg')
-- table.insert(bgModelPaths, datapath .. 'background/cameraJitter/sidewalk/model/staticModel.jpg')
-- table.insert(ROIPaths,     datapath .. 'background/cameraJitter/sidewalk/ROI/ROI.jpg')
-- table.insert(outputNames, 'sidewalk')

-- table.insert(inputPaths,   datapath .. 'val/badWeather/snowFall/input/in00XXX.jpg')
-- table.insert(bgModelPaths, datapath .. 'background/badWeather/snowFall/model/staticModel.jpg')
-- table.insert(ROIPaths,     datapath .. 'background/badWeather/snowFall/ROI/ROI.jpg')
-- table.insert(outputNames, 'snowFall')

-- table.insert(inputPaths,   datapath .. 'val/nightVideos/fluidHighway/input/in000597.jpg')
-- table.insert(bgModelPaths, datapath .. 'background/nightVideos/fluidHighway/model/staticModel.jpg')
-- table.insert(ROIPaths,     datapath .. 'background/nightVideos/fluidHighway/ROI/ROI.jpg')
-- table.insert(outputNames, 'fluidHighway')

-- table.insert(inputPaths,   datapath .. 'val/PTZ/continuousPan/input/in00XXX.jpg')
-- table.insert(bgModelPaths, datapath .. 'background/PTZ/continuousPan/model/staticModel.jpg')
-- table.insert(ROIPaths,     datapath .. 'background/PTZ/continuousPan/ROI/ROI.jpg')
-- table.insert(outputNames, 'continuousPan')

-- table.insert(inputPaths,   '/home/lpcinelli/Documents/cdnet2014/dataset/lowFramerate/turnpike_0_5fps/input/in001126.jpg')
-- table.insert(bgModelPaths, datapath .. 'background/lowFramerate/turnpike_0_5fps/model/staticModel.jpg')
-- table.insert(ROIPaths,     datapath .. 'background/lowFramerate/turnpike_0_5fps/ROI/ROI.jpg')
-- table.insert(outputNames, 'lowFramerate_turnpike_00126')

-- imagePath = data_path .. video_path .. 'input/' .. img_num .. '.jpg'
-- bgModelPath = data_path .. video_path .. 'reference/' .. ref_num .. '.jpg'
-- imagePath = data_path .. 'val/' .. video_path .. 'input/in' .. img_num .. '.jpg'
-- gtPath = data_path .. 'val/' .. video_path .. 'groundtruth/gt' .. img_num .. '.png'
-- bgModelPath = data_path .. 'background/' .. video_path .. 'model/staticModel.jpg'
-- ROIPath = data_path .. 'background/' .. video_path .. 'ROI/ROI.jpg'

-- modelPath = '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/resnet,LR=0.01,batchSize=16,dataset=cdnet,depth=46,nEpochs=50,optimizer=adam,shortcutType=B,weightDecay=0.0005/Wed-Jan-18-21:01:37-2017/model_42.t7'
-- modelPath = '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/lenet5,LR=0.001,dataset=cdnet,nEpochs=40,optimizer=adam,weightDecay=0.0005/Thu-Jan-19-15:16:56-2017/model_3.t7'
-- modelPath = '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/lenet5,LR=0.001,dataset=cdnet,nEpochs=40,optimizer=adam,weightDecay=0.0005/Thu-Jan-19-11:28:35-2017/model_7.t7'
--modelPath = '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/lenet5,LR=0.001,dataset=cdnet,nEpochs=40,optimizer=adam,weightDecay=0.0005/Wed-Jan-18-16:15:10-2017/model_40.t7'
-- modelPath = '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/lenet5,LR=0.001,dataset=cdnet,nEpochs=40,optimizer=adam,weightDecay=0.0005/Wed-Jan-18-16:15:10-2017/model_40.t7'

-- modelPath = '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/lenet5/deconv/ch-1/better-init/linear-deconv/BN-layer/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Thu-Jan-26-21:12:31-2017/model_60.t7'
-- imagePath = '/home/lpcinelli/Documents/cdnet2014/dataset/dynamicBackground/boats/input/in002001.jpg'
-- bgModelPath = '/home/lpcinelli/Documents/cdnet2014/deep-subtraction-split/background/dynamicBackground/boats/model/staticModel.jpg'
-- videoROIPath = '/home/lpcinelli/Documents/cdnet2014/deep-subtraction-split/background/dynamicBackground/boats/ROI/ROI.jpg'


------- LENET ---------

-- table.insert(modelPaths, '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/lenet5/bilinear-upsample/ch-1/BN-layer/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Wed-Jan-25-09:06:47-2017/model_60.t7')
-- table.insert(modelNames, 'lenet_bilinear')

-- table.insert(modelPaths, '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/lenet5/deconv/ch-1/better-init/linear-deconv/BN-layer/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Thu-Jan-26-21:12:31-2017/model_60.t7')
-- table.insert(modelNames, 'lenet_lin_deconv')

-- table.insert(modelPaths, '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/lenet5/dilation/ch-1/BN-layer/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Wed-Jan-25-00:11:07-2017/model_60.t7')
-- table.insert(modelNames, 'lenet_dilated')

------- RESNET1 ---------

-- table.insert(modelPaths, '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/resnet/cifar-10/bilinear-upsample/ch1/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,weightDecay=0.0002/Wed-Jan-25-19:50:57-2017/resume/model_97.t7')
-- table.insert(modelNames, 'resnet_cifar_32_bilinear')

-- table.insert(modelPaths, '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/resnet/cifar-10/deconv/ch-1/linear-deconv/bad-init/,LR=0.01,batchSize=16,dataset=cdnet,nEpochs=60,optimizer=adam,shortcutType=B,weightDecay=0.0002/Tue-Jan-24-13:00:13-2017/model_60.t7')
-- table.insert(modelNames, 'resnet_cifar_32_lin_deconv')

-- table.insert(modelPaths, '/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/resnet/cifar-10/dilated/ch-1/,LR=0.01,base=cifar10,batchSize=16,dataset=cdnet,depth=34,nEpochs=60,optimizer=adam,weightDecay=0.0002/Fri-Jan-27-14:07:46-2017/model_60.t7')
-- table.insert(modelNames, 'resnet_cifar_32_dilated')

table.insert(modelPaths, '/local/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/resnet/cifar-10/bilinear-upsample/ch1/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=90,optimizer=adam,weightDecay=0.0002/Fri-Mar-31-18:45:23-2017/resume/model_75.t7')
table.insert(modelNames, 'resnet-bilinear')

table.insert(modelPaths, '/local/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/resnet/cifar-10/decode/shallow/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=80,optimizer=adam,weightDecay=0.0002/Mon-Apr-10-14:24:56-2017/model_80.t7')
table.insert(modelNames, 'resnet-decoder-shallow')

table.insert(modelPaths, '/local/home/lpcinelli/repos/fb.resnet.torch.perso/checkpoints/resnet/cifar-10/deconv/ch-1/non-linear-deconv/better-init/full-base/,LR=0.01,LR_factor=0.5,LR_step=10,base=cifar10,batchSize=16,dataset=cdnet,depth=32,nEpochs=90,optimizer=adam,weightDecay=0.0002/Tue-Apr--4-23:59:21-2017/resume/model_78.t7')
table.insert(modelNames, 'resnet-non-lin-deconv')

function preprocess(input, target)
   local  transform = t.Compose{
      t.getImageROI(),
      t.OneGrayscaleImagePerChannel(),
      -- t.ScaleDim(256,'w'),
      -- t.ScaleDim(192,'h'),
      -- t.Lighting(0.1, pca.eigval, pca.eigvec),
      -- t.ColorNormalize(meanstd),
      -- t.HorizontalFlip(0.5),
      }
   return transform(input,target)
end

-- prep = t.Compose{
--          t.getImageROI(),
--          t.OneGrayscaleImagePerChannel(),
--          -- t.ScaleDim(256,'w'),
--          -- t.ScaleDim(192,'h'),
--        }
	
threshold = 0.5

for modelPathNb=1,#modelPaths do
	print('model ' .. modelNames[modelPathNb])
	for example=1,#inputPaths do
		local img = image.load(inputPaths[example], 3, 'float')
		local bgModel = image.load(bgModelPaths[example], 3, 'float')
		local videoROI = image.load(ROIPaths[example], 3, 'float')
		-- inputImg = torch.load(inputPaths[example]).input
		local input = torch.cat({videoROI:expandAs(img), bgModel:expandAs(img), img}, 4)
		-- input2, target2 = prep(input)
		inputImg = preprocess(input)
		

		model = torch.load(modelPaths[modelPathNb]):cuda()
		model:evaluate()
		output = model:forward(inputImg:cuda()):float():squeeze()
		outputThreshold = output:ge(threshold):float()
		
		image.save('examples/projeto-sergio/'.. modelNames[modelPathNb] .. '/' .. outputNames[example]  .. '.jpg', output)
		image.save('examples/projeto-sergio/'.. modelNames[modelPathNb] .. '/' .. outputNames[example]  .. '-threshold.jpg', outputThreshold)
		image.save('examples/projeto-sergio/input/' .. outputNames[example]  .. '.jpg', img)
		-- image.save('bg.jpg',inputImg[1]:view(1, table.unpack(inputImg[1]:size():totable())))
		-- image.save('in.jpg',inputImg[2]:view(1, table.unpack(inputImg[2]:size():totable())))
	end

end

-- videoROI = torch.FloatTensor(img:size())
-- videoROI[1]:fill(1)

-- videoROI = image.load(ROIPath, 3, 'float')
-- gt = image.load(gtPath, 1, 'float')
-- gt = gt:eq(1):float()

-- input2, target2 = prep(input, gt)


-- th_output = output:ge(threshold):float()

-- Metrics
-- local targetArea = target2:sum() -- TP + FN
-- local outputArea = th_output:sum() -- TP + FP
-- local intersection = torch.cmul(th_output,target2):sum() -- TP
-- local trueNeg = (th_output + target2):eq(0):float():sum() -- TN

-- print(('Accuracy: %g'):format(th_output:eq(target2):float():mean()))
-- print(('Precision: %g'):format(intersection/outputArea))
-- print(('Recall: %g'):format(intersection/targetArea))
-- print(('F1: %g'):format(2*intersection/(outputArea + targetArea)))

-- image.save('input-test.jpg', img)
-- image.save('target-test.jpg', target2)
-- image.save('output-bin-test.jpg', th_output)
