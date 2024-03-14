import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
from torch.nn.functional import one_hot

transform = transforms.Compose([
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
    mask = torch.all(out_image < threshold, dim=1, keepdim=True)
    mask = mask.expand_as(out_image)
    out_image[mask] = 0
    return out_image


if __name__ == '__main__':
    data = MyDataset('data')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)
