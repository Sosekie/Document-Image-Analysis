import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
weights_path = 'params'
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters(), lr=0.01)
    loss_fun = nn.CrossEntropyLoss()
    losses = []

    epoch = 1
    threshold = 0.0
    while epoch < 200:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)

            train_loss = loss_fun(out_image, segment_image)
            losses.append(train_loss.item())
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]
            _out_image_block = Background_Threshold(out_image, threshold)[0]
            if threshold < 0.9:
                threshold = threshold + 0.01

            img = torch.stack([_image, _segment_image, _out_image, _out_image_block], dim=0)
            save_image(img, f'{save_path}/{i}.gif')

            if i % 5 == 0:
                torch.save(net.state_dict(), weight_path)
                print('save successfully!')
                
        epoch += 1

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig('training_loss_over_time.png', dpi=300)
    plt.show()
