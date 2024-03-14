# DIA UNet

#### Introduction
Document layout segmentation By UNet

#### Install

1.  conda create --name DIA

2.  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

3.  conda activate DIA

#### How to train

1.  input：data/JPEGImages   mask：data/SegmentationClass
2.  python train.py
3.  weights are saved in params

#### How to test
1.  python test.py
