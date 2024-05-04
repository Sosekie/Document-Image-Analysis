import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, subset='train', seed=42, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subset = subset
        all_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

        # Split the dataset
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

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        file_name = self.names[idx]
        img_name = os.path.join(self.image_dir, file_name)
        mask_name = os.path.join(self.mask_dir, file_name.replace('.jpg', '.gif'))
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("RGB")
        
        # Resize images and masks
        resize = transforms.Resize([image.height // 2, image.width // 2])
        image = resize(image)
        mask = resize(mask)

        # Convert mask to one-hot encoding
        mask = np.array(mask)
        one_hot_mask = np.zeros((*mask.shape[:2], 6), dtype=np.uint8)
        colors = [(0,255,255), (255,0,255), (0,255,0), (255,0,0), (0,0,255), (0,0,0)]
        for i, color in enumerate(colors):
            one_hot_mask[:, :, i] = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            one_hot_mask = self.target_transform(one_hot_mask)

        return image, one_hot_mask

    def restore_mask(self, one_hot_mask):
        # Convert one-hot encoded mask back to RGB image
        colors = np.array([(0,255,255), (255,0,255), (0,255,0), (255,0,0), (0,0,255), (0,0,0)], dtype=np.uint8)
        rgb_mask = np.dot(one_hot_mask.astype(np.float32), colors)
        return Image.fromarray(rgb_mask.astype(np.uint8))