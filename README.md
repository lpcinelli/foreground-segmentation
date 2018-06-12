# Foreground Segmentation for Anomaly Detection in Surveillance Videos

Pytorch code for **SBRT 2017** paper **Foreground Segmentation for Anomaly Detection in Surveillance Videos Using Deep Residual Networks** available [here](https://www.researchgate.net/publication/319681001_Foreground_Segmentation_for_Anomaly_Detection_in_Surveillance_Videos_Using_Deep_Residual_Networks)

The aim of this work (under [deepeye](https://github.com/lpcinelli/foreground-segmentation/tree/master/deepeye) folder) is to detect and segment anomalies in a target video given a temporally aligned reference video (anomaly-free). The output segmentation map has the same resolution as the input video frame.

## Dataset

### CDNET

For our experiments, we used [CDNET](http://changedetection.net/) database. A database for identification of changing or moving areas in the field of view of a camera, covering a wide range of detection challenges and are representative of typical indoor and outdoor visual data captured today in surveillance:

* Dynamic background
* Camera jitter
* Intermittent object motion
* Shadows
* Thermal signatures
* Challenging weather
* Low frame-rate
* Acquisition at night
* PTZ capture
* Air turbulence

In this preliminary work, instead of a entire reference video, we use a single still reference frame by taking the median of each pixel throughout the first 150 frames of the considered target video. Although not ideal, this does not have much influence since videos in CDNET are recorded with a stationary camera (except for the PTZ class, for which the algorithm's performance naturally is worse). It is worth emphasizing that our algorithm allows the more general setting of using a whole video (with egomotion) for reference, and not a single still image, which is compared frame per frame with the target video.

### VDAO

The idea is now use it on the [VDAO](http://www02.smt.ufrj.br/~tvdigital/database), a video database containing annotated videos in a cluttered industrial environment, in which the videos were captured using a camera on a moving platform.
You can have a bunch of useful tools to play with VDAO database in the [VDAO_Access Project](https://github.com/rafaelpadilla/DeepLearning-VDAO/tree/master/VDAO_Access).

## Downloading the dataset
Once you have installed Python you can just prompt:
```bash
$ cd data; python download.py
```

## Training a model using the `main.py` script

This script allows training all models using a command-line interface. The call should be something like:
```bash
$ main.py --manifest TRAIN --img_path DIR --arch ARCH train \
          --epochs N --lr LR
```

Example of call which instantiates and trains a 20-layer ResNet with reconstruction by bilinear upsampling:
````bash
python main.py --img-dir ~/Documents/database/cdnet2014/dataset --shape 2,192,256 --arch resnet20 --arch-params 'up_mode=upsample' --manifest data/manifest.train --loss bce -b 16 train --epochs 90 --aug --lr 0.01 --wd 0.0002 --val data/manifest.val --save models/resnet20-bilinear.pth.tar
````

For more details, you may prompt
```bash
$ main.py --help
```
or just check out [main.py](../main.py).

This script will automatically save the model at every epoch.


## Evaluating a model using the `main.py` script

Evaluating a trained model can be done by simply

```bash
$ main.py --manifest EVAL --img_path DIR --arch ARCH \
          --load PATH eval
```

## Custom scripts

### Custom model

All models are defined by a class defined in the [models](codes/models/) package. A custom model can be defined as

```python

# Filename: codes/models/customnet.py

# this line is necessary
__all__ = ['CustomNet', 'customnet']


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        ...

    def forward(self, x):
        ...
        return out

# This method is required by the main script
def customnet(**kwargs):
    ...
    return CustomNet(**kwargs)
```

To make the `CustomNet` visible in `main.py`, we have to append the following code to [init](models/__init__.py) script

```python
  # Filename: codes/models/__init__.py

  from .customnet import *
```

### Custom Callback

All callbacks must inherit [Callback](codes/callbacks.py) and can optionally implement one of 8 calls. The default cycle is:
1. on_begin
2. on_epoch_begin
3. on_step_begin
4. on_batch_begin
5. on_batch_end
6. on_step_end
7. on_epoch_end
8. on_end

A simple custom callback that prints at the beginning and at the end of each epoch is given:
```python

class CustomCallback(object):

    def on_epoch_begin(self, epoch):
        print('epoch begin')

    def on_epoch_end(self, metrics):
        print('epoch end')

```

## Requirements

### Softwares

* Python
* [7za](http://www.7-zip.org/download.html)
* zip

### Python packages

* pytorch
* torchvision
* numpy
* pandas
* matplotlib
* pillow
* glob2
* inflection
* tqdm
* visdom

## Citation

If you use this code in your research, please use the following BibTeX entry.

````
@inproceedings{cinelli2017,
title = {Foreground Segmentation for Anomaly Detection in Surveillance Videos Using Deep Residual Networks},
author = {Cinelli, Lucas P and Thomaz, Lucas A and da Silva, Allan F and da Silva, Eduardo AB and Netto, Sergio L},
booktitle = {Simpósio Brasileiro de Telecomunicações e Processamento de Sinais (SBRT)},
month = September,
year = {2017}
}
````

## Acknowledge

THe download script, main[]().py structure, parts of readme, callbacks, and many others were done by [Igor Macedo Quintanilha](https://igormq.github.io/about), a good friend and colleague.

## License

See [LICENSE.md](LICENSE.md)

