require 'image'
t = require 'datasets/transforms'

imagePath = '/home/lpcinelli/Documents/cdnet2014/train-val-split/train/dynamicBackground/boats/input/in002000.jpg'
gtPath = '/home/lpcinelli/Documents/cdnet2014/train-val-split/train/dynamicBackground/boats/groundtruth/gt002000.png'
bgModelPath = '/home/lpcinelli/Documents/cdnet2014/train-val-split/background/dynamicBackground/boats/model/staticModel.jpg'
ROIPath = '/home/lpcinelli/Documents/cdnet2014/train-val-split/background/dynamicBackground/boats/ROI/ROI.jpg'
-- imagePath = '/home/lpcinelli/Documents/cdnet2014/deep-subtraction-split/train/baseline/highway/input/in000860.jpg'
-- gtPath = '/home/lpcinelli/Documents/cdnet2014/deep-subtraction-split/train/baseline/highway/groundtruth/gt000860.png'
-- bgModelPath = '/home/lpcinelli/Documents/cdnet2014/deep-subtraction-split/background/baseline/highway/model/staticModel.jpg'
-- ROIPath = '/home/lpcinelli/Documents/cdnet2014/deep-subtraction-split/background/baseline/highway/ROI/ROI.jpg'


img = image.load(imagePath, 3, 'float')
bgModel = image.load(bgModelPath, 3, 'float')
videoROI = image.load(ROIPath, 3, 'float')
gt = image.load(gtPath, 1, 'float')
gt[gt:lt(1)] = 0
-- gt = gt:view(1,table.unpack(gt:size():totable()))

-- sample = torch.cat({	videoROI:expandAs(img),
-- 							bgModel:expandAs(img),
-- 							img}, 4)

-- preprocess1 = t.getImageROI()
-- tmp = preprocess1(sample)
-- bgModelROI = tmp[{{},{},{},{1}}]
-- imgROI = tmp[{{},{},{},{2}}]

-- print(bgModelROI:size())
-- print(imgROI:size())

-- image.save('bgModelROI.jpg',bgModelROI)
-- image.save('imgROI.jpg',imgROI)

-- preprocess2 = t.OneGrayscaleImagePerChannel()
-- tmp2 = preprocess2(tmp)
-- print(tmp:size())
-- print(img:size())

-- img_grey = tmp2[1]:view(1,table.unpack(tmp2[1]:size():totable()))
-- bg_grey = tmp2[2]:view(1,table.unpack(tmp2[2]:size():totable()))
-- image.save('img-grey.jpg', img_grey)
-- image.save('bg-rey.jpg', bg_grey)

-- preprocess3 = t.ScaleDim(100,'h')
-- tmp3,tmp4 = preprocess3(img,img)

-- function preprocess4()
-- 	return t.Compose{
-- 				    t.getImageROI(),
-- 				    t.OneGrayscaleImagePerChannel(),
-- 				    t.ScaleDim(256,'w'),
-- 				    t.ScaleDim(192,'h'),
-- 				 	 t.HorizontalFlip(0.5),}
-- 	end

-- preprocess = preprocess4()

-- tmp5, tmp6 = preprocess(sample, gt)
-- print(tmp5:size())
-- print(tmp6:size())
-- print(img:size())

-- flip = t.HorizontalFlip(1)
-- flipImg, flipGt = flip(img,gt)
-- print(flipImg:size())
-- print(flipGt:size())

-- image.save('flipImg.jpg',flipImg)
-- image.save('flipGt.jpg',flipGt)


scale = t.Compose{t.ScaleDim(256,'w'), t.ScaleDim(192,'h')}
scaled_img, scaled_gt = scale(img, gt)
image.save('scaled_img.jpg', scaled_img)
image.save('scaled_gt.jpg', scaled_gt)
image.save('gt.jpg', gt)