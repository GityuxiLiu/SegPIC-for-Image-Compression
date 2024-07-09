# Region Adaptive Transform with Segmentation Prior for Image Compression
The [paper](https://arxiv.org/abs/2403.00628) has been accepted by ECCV2024！ The Arxiv version will be updated soon. Thank you for your attention!

_I am looking for PhD opportunities, so if you are interested in my research, feel free to ask me for a CV._

_我正在寻找博士学位的机会，如果您对我的研究感兴趣，欢迎向我询问简历。_

email: yuxiliu@bjtu.edu.cn

## About
Our SegPIC introduces proposed RAT and SAL based on [WACNN](https://github.com/Googolxx/STF).

![arch](https://github.com/GityuxiLiu/SegPIC-for-Image-Compression/blob/main/assets/arch.png)

We compare our SegPIC with previously well-performing methods.

![psnr](https://github.com/GityuxiLiu/SegPIC-for-Image-Compression/blob/main/assets/psnr.png)

Visualization of the reconstructed images kodim04 and kodim24 in Kodak. The metrics are (PNSR↑/bpp↓). It shows that our SegPIC can distinguish the objects’ contours more accurately, making the edges sharper with less bitrate.

![vis](https://github.com/GityuxiLiu/SegPIC-for-Image-Compression/blob/main/assets/vis.png)

## Installation
The code is based on [WACNN](https://github.com/Googolxx/STF) and [CompressAI](https://github.com/InterDigitalInc/CompressAI).
You can refer to them for installation. It is also recommended to adopt Pytorch-2.0 for faster training speed.

## Checkpoints
We provide 6 checkpoints optimized by MSE. See [Google Drive](https://drive.google.com/drive/folders/1rDyvCVkTiqzCq4urW60OsIKOTLWBp3si?usp=drive_link).

## Training Dataset
[COCO-train-2017](http://images.cocodataset.org/zips/train2017.zip) for training, [COCO-val-2017](http://images.cocodataset.org/zips/val2017.zip) for validation and [panoptic_annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) for .png masks. Images and masks correspond by the same filename (no suffix).
The data format is as follows:
```bash
- COCO-Stuff/
    - train2017/
        - img000.jpg
        - img001.jpg
    - val2017/
        - img002.jpg
        - img003.jpg
    - annotations/
        - panoptic_train2017/
            - img000.png
            - img001.png
        - panoptic_val2017/
            - img002.png
            - img003.png
```
## Training and Testing
The overall usage is the same as [WACNN](https://github.com/Googolxx/STF) and [CompressAI](https://github.com/InterDigitalInc/CompressAI). Please see `run.sh` and `test.sh`.
