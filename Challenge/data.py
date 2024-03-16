import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from PIL import ImageFilter
import cv2
from torchvision import transforms

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
        np_image = np.array(x)
        np_image = cv2.GaussianBlur(np_image, (self.kernel_size, self.kernel_size), self.sigma)
        return Image.fromarray(np_image)

transform = transforms.Compose([
    # MedianFilter(size=3),
    GaussianBlur(),
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, path, inputdir, maskdir):
        self.path = path
        self.maskdir = maskdir
        self.inputdir = inputdir
        self.name = os.listdir(os.path.join(path, maskdir))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index, resize = False):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, self.maskdir, segment_name)
        image_path = os.path.join(self.path, self.inputdir, segment_name.replace('gif', 'jpg'))
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

def reduce_yellow(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_color = np.array([255, 255, 0])
    replacement_color = np.array([0, 0, 0])

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_path = os.path.join(input_dir, filename)
            image = Image.open(file_path)
            image = image.convert('RGB')
            
            data = np.array(image)
            data[(data == target_color).all(axis=-1)] = replacement_color
            new_image = Image.fromarray(data, mode='RGB')

            new_image.save(os.path.join(output_dir, filename))

if __name__ == '__main__':

    input_directory = './data/SegmentationClass'
    output_directory = './data/SegmentationClass_noYellow'
    reduce_yellow(input_directory, output_directory)
