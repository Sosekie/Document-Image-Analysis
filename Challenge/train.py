import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# device = 'cpu'
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'
if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()
    losses = []

    epoch = 1
    while epoch < 200:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)

            # print(out_image.size())
            # print(segment_image.size())

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

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.gif')
        if epoch % 10 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')
        epoch += 1
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))  # 设置图表大小
    plt.plot(losses, label='Training Loss')  # 绘制损失值
    plt.xlabel('Iterations')  # X轴标签
    plt.ylabel('Loss')  # Y轴标签
    plt.title('Training Loss Over Time')  # 图表标题
    plt.legend()  # 显示图例
    plt.show()  # 显示图表
