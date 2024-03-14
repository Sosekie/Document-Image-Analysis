import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from PIL import ImageFilter
import cv2
from torchvision import transforms
from torch.nn.functional import one_hot

class MedianFilter(object):
    def __init__(self, size=3):
        self.size = size

    def __call__(self, x):
        return x.filter(ImageFilter.MedianFilter(self.size))

class GaussianBlur(object):
    def __init__(self, kernel_size=7, sigma=1.5):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x):
        # 将PIL图像转换为NumPy数组
        np_image = np.array(x)
        # 应用高斯滤波
        np_image = cv2.GaussianBlur(np_image, (self.kernel_size, self.kernel_size), self.sigma)
        # 将NumPy数组转换回PIL图像
        return Image.fromarray(np_image)

transform = transforms.Compose([
    # MedianFilter(size=3),
    GaussianBlur(),
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index, resize = False):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('gif', 'jpg'))
        if resize == True:
            segment_image = keep_image_size_open(segment_path)
            image = keep_image_size_open_rgb(image_path)
        else:
            segment_image = Image.open(segment_path).convert("RGB")
            image = Image.open(image_path)
        return transform(image), transform(segment_image)
    
def Background_Threshold(out_image, threshold):
    result_image = out_image.clone()

    mask = torch.any(result_image%(200.0/255.0) > (55.0/255.0), dim=1, keepdim=True)
    mask = mask.expand_as(result_image)
    result_image[mask] = 0

    mask = torch.all(result_image < threshold, dim=1, keepdim=True)
    mask = mask.expand_as(result_image)
    result_image[mask] = 0

    return result_image


if __name__ == '__main__':
    data = MyDataset('data')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)
