# DIA UNet

#### Introduction
Document layout segmentation By UNet

#### Install

1.  conda create --name DIA python==3.9
2.  conda activate DIA
3.  pip install -r requirements.txt
4.  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

#### How to train

1.  input：data/JPEGImages   mask：data/SegmentationClass
2.  python train.py
3.  weights are saved in params
4.  rm -rv ./train_image
5.  rm -rv ./params/unet.pth
6.  rm -rv ./params/unet.pth ./train_image

#### How to test
1.  python test.py
