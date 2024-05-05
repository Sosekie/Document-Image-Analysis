import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

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

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        colors = [[0, 255, 255], [255, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255], [0, 0, 0]]
        file_name = self.names[idx]
        img_name = os.path.join(self.inputdir, file_name.replace('.gif', '.jpg'))
        mask_name = os.path.join(self.maskdir, file_name.replace('.jpg', '.gif'))
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).seek(0).convert("RGB")
        
        # Resize images and masks
        resize = transforms.Resize([image.height // 2, image.width // 2])
        image = resize(image)
        mask = resize(mask)

        # Convert mask to one-hot encoding
        mask = np.array(mask)
        one_hot_mask = np.zeros((*mask.shape[:2], 6), dtype=np.uint8)
        for i, color in enumerate(colors):
            match = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
            one_hot_mask[:, :, i] = match

        return image, one_hot_mask