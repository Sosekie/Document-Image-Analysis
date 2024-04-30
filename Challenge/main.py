import os
import torch
from torch.utils.data import DataLoader
from data import MyDataset_tvt_label_onehot
from net import UNet_simple, UNetPlusPlus
from train import train
from test import test


class MyPipeline:
    def __init__(self, data_path, weights_path='params', save_path='train_image', result_path='result', batch_size=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = weights_path
        self.weight_path = os.path.join(weights_path, 'best_model_weights.pth')
        self.data_path = data_path
        self.save_path = save_path
        self.result_path = result_path
        self.batch_size = batch_size

        self.prepare_directories()
        self.load_datasets()
        self.model = self.init_model()
        print(f'Using {self.device}')

    def prepare_directories(self):
        for path in [self.weights_path, self.save_path, self.result_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def load_datasets(self):
        self.train_dataset = MyDataset_tvt_label_onehot(self.data_path, inputdir='JPEGImages_black', maskdir='SegmentationClass_noYellow', subset="train")
        self.val_dataset = MyDataset_tvt_label_onehot(self.data_path, inputdir='JPEGImages_black', maskdir='SegmentationClass_noYellow', subset="val")
        self.test_dataset = MyDataset_tvt_label_onehot(self.data_path, inputdir='JPEGImages_black', maskdir='SegmentationClass_noYellow', subset="test")
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def init_model(self):
        model = UNetPlusPlus().to(self.device)
        if os.path.exists(self.weight_path):
            model.load_state_dict(torch.load(self.weight_path))
            print('Successful load weight!')
        else:
            print('Not successful load weight')
        return model

    def train(self, epochs=1):
        train(self.model, self.train_data_loader, self.val_data_loader, self.device, epochs = epochs)

    def test(self):
        self.model = self.init_model()
        test(self.model, self.test_data_loader, self.device)


if __name__ == '__main__':
    pipeline = MyPipeline(data_path='data', batch_size=1)
    pipeline.train(epochs=300)
    pipeline.test()