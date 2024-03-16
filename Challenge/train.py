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
loss_path = 'result/loss.tensor'
data_path = r'data'
save_path = 'train_image'

batch_size = 1

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path, inputdir='JPEGImages', maskdir='SegmentationClass_noYellow'), batch_size = batch_size, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    # opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_fun = nn.CrossEntropyLoss()
    losses = []

    epoch = 1
    while epoch < 500:
    # while True:
        loss_epoch = 0.0
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            if i%3 == 0:
                view = True
                # torch.save(net.state_dict(), weight_path)
            else:
                view = False
            if view:
                out_image, middle_x = net(image, view)
            else:
                out_image = net(image, view)
            train_loss = loss_fun(out_image, segment_image)
            loss_epoch = loss_epoch + train_loss.item()
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 1 == 0:
                print(f'epoch:{epoch}-batch:{i}-train_loss===>>{train_loss.item()}')

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]
            _out_image_block = Background_Threshold(out_image, threshold = 0.95)[0]

            if view:
                _middle_x = []
                for stack_i in range(len(middle_x)):
                    if stack_i < len(middle_x)-0:
                        _middle_x.append(middle_x[stack_i][0].mean(dim=0).unsqueeze(0).expand(3, -1, -1))
                    else:
                        _middle_x.append(middle_x[stack_i][0])
                img = torch.stack([_image, 
                                _middle_x[0], 
                                _middle_x[1], 
                                _middle_x[2], 
                                _middle_x[3], 
                                _middle_x[4], 
                                _middle_x[5], 
                                _middle_x[6], 
                                _middle_x[7], 
                                _middle_x[8], 
                                _middle_x[9], 
                                _middle_x[10], 
                                _middle_x[11], 
                                _middle_x[12], 
                                # _middle_x[13], 
                                # _middle_x[14], 
                                # _middle_x[15], 
                                _out_image, 
                                _out_image_block, 
                                _segment_image], dim=0)

                save_image(img, f'{save_path}/{i}.gif')
            num_of_i = float(i)

        losses.append(loss_epoch / num_of_i)
        torch.save(net.state_dict(), weight_path)
        torch.save(losses, loss_path)
        print('save successfully!')
        print('losses:', losses)
                
        epoch += 1

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig('training_loss_over_time.png', dpi=300)
    plt.show()
