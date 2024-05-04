import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import *
from PIL import ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as TF
from collections import defaultdict

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

# transform = transforms.Compose([
#     # MedianFilter(size=3),
#     # GaussianBlur(),
#     transforms.ToTensor()
# ])


transform = transforms.Compose([
    transforms.Resize((960//2, 640//2)),
    transforms.ToTensor(),
])
    

class MyDataset(Dataset):
    def __init__(self, path, inputdir, maskdir, subset="train", seed=42):
        self.path = path
        self.inputdir = inputdir
        self.maskdir = maskdir
        all_names = os.listdir(os.path.join(path, maskdir))
        train_val_names, test_names = train_test_split(all_names, test_size=0.1, random_state=seed)
        train_names, val_names = train_test_split(train_val_names, test_size=1/9, random_state=seed)
        
        if subset == "train":
            self.names = train_names
        elif subset == "val":
            self.names = val_names
        elif subset == "test":
            self.names = test_names
        else:
            raise ValueError(f"Unknown subset: {subset}")

        self.unique_classes, self.color_map = self._get_unique_classes_and_colors()

    def _get_unique_classes_and_colors(self):
        unique_labels = [0, 1, 2, 3, 4, 5]
        color_map = {}
        colors = [(0,255,255), (255,0,255), (0,255,0), (255,0,0), (0,0,255), (0,0,0)]
        for label, color in zip(sorted(unique_labels), colors):
            color_map[label] = color
        return unique_labels, color_map

    def one_hot_to_rgb(self, one_hot_labels):
        batch_size, num_classes, height, width = 1, 6, 480, 320
        rgb_images = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
        one_hot_labels = (one_hot_labels > 0.5).astype(np.uint8)
        for c in range(num_classes):
            color = np.array(self.color_map[c], dtype=np.uint8)
            for i in range(batch_size):
                rgb_images[i][one_hot_labels[i, c] == 1] = color
        
        return rgb_images

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        segment_name = self.names[index]
        segment_path = os.path.join(self.path, self.maskdir, segment_name)
        image_path = os.path.join(self.path, self.inputdir, segment_name.replace('gif', 'jpg'))
        segment_image = Image.open(segment_path)
        segment_labels = np.array(segment_image)[::2, ::2]
        image = Image.open(image_path).convert("RGB")

        # Convert segment_labels to one-hot encoding
        one_hot_labels = np.zeros((len(self.unique_classes), segment_labels.shape[0], segment_labels.shape[1]), dtype=np.int32)
        for i, unique_label in enumerate(self.unique_classes):
            one_hot_labels[i, :, :] = (segment_labels == unique_label)

        return transform(image), one_hot_labels
    

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

def process_image(input_path, output_path):
    image = Image.open(input_path)
    image = image.convert('RGB')
    data = image.load()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = data[x, y]
            if r > 160 and g > 160 and b > 160:
                data[x, y] = (0, 0, 0)
    image.save(output_path)

if __name__ == '__main__':

    # input_directory = './data/SegmentationClass'
    # output_directory = './data/SegmentationClass_noYellow'
    # reduce_yellow(input_directory, output_directory)

    input_folder = 'data/JPEGImages'
    output_folder = 'data/JPEGImages_black'
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path)
            print(f'Processed and saved {filename} to {output_folder}')
