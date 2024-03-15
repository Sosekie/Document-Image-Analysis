import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)
    
class DownResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownResidual, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x, x_ori):
        x_ori = self.conv1x1(x_ori)
        x_ori = F.avg_pool2d(x_ori, 2)
        x = x + x_ori
        x = self.leaky_relu(x)
        return x

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)

class UpResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpResidual, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x, x_ori):
        x_ori = self.up_conv(x_ori)
        x = x + x_ori
        x = self.leaky_relu(x)
        return x

class BackConv2d(nn.Module):
    def __init__(self, kernel_size, channels, threshold, tolerance):
        super(BackConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.register_buffer('kernel', torch.ones((channels, 1, kernel_size, kernel_size)))
        self.padding = kernel_size // 2
        self.groups = channels
        self.threshold = threshold
        self.tolerance = tolerance
        self.kernel.requires_grad = False
    def forward(self, x):
        condition_met = x >= (self.threshold / 255.0)
        condition_met = condition_met.float()
        condition_met = F.pad(condition_met, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        condition_sum = F.conv2d(condition_met, self.kernel, groups=self.groups)
        black_pixels_mask = (condition_sum >= self.kernel_size * self.kernel_size * self.tolerance).float()
        output_img = x * (1 - black_pixels_mask)
        return output_img

class UNet(nn.Module):
    def __init__(self,num_classes = 3):
        super(UNet, self).__init__()
        self.bc1 = BackConv2d(25, 3, 170.0, 0.8)
        self.bc2 = BackConv2d(25, 3, 170.0, 0.8)
        self.c1 = Conv_Block(3,64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64,128)
        self.r2 = DownResidual(64,128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128,256)
        self.r3 = DownResidual(128,256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256,512)
        self.r4 = DownResidual(256,512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512,1024)
        self.r5 = DownResidual(512,1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024,512)
        self.r6 = UpResidual(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.r7 = UpResidual(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.r8 = UpResidual(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.r9 = UpResidual(128, 64)
        self.out = nn.Conv2d(64,num_classes,3,1,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self,x):

        xb1 = self.bc1(x)
        xb2 = self.bc2(xb1)

        R1 = self.c1(xb2)
        R1_v = R1
        
        R2 = self.c2(self.d1(R1))
        R2 = self.r2(R2,R1)
        R2_v = F.interpolate(R2,scale_factor=2,mode='nearest')

        R3 = self.c3(self.d2(R2))
        R3 = self.r3(R3,R2)
        R3_v = F.interpolate(R3,scale_factor=4,mode='nearest')

        R4 = self.c4(self.d3(R3))
        R4 = self.r4(R4,R3)
        R4_v = F.interpolate(R4,scale_factor=8,mode='nearest')

        R5 = self.c5(self.d4(R4))
        R5 = self.r5(R5,R4)
        R5_v = F.interpolate(R5,scale_factor=16,mode='nearest')

        O1 = self.c6(self.u1(R5, R4))
        O1 = self.r6(O1,R5)
        O1_v = F.interpolate(O1,scale_factor=8,mode='nearest')

        O2 = self.c7(self.u2(O1, R3))
        O2 = self.r7(O2,O1)
        O2_v = F.interpolate(O2,scale_factor=4,mode='nearest')

        O3 = self.c8(self.u3(O2, R2))
        O3 = self.r8(O3,O2)
        O3_v = F.interpolate(O3,scale_factor=2,mode='nearest')

        O4 = self.c9(self.u4(O3, R1))
        O4 = self.r9(O4,O3)
        O4_v = O4

        last = self.out(O4)

        output = self.sigmoid(last)

        middle_x = [xb1, xb2, R1_v, R2_v, R3_v, R4_v, R5_v, O1_v, O2_v, O3_v, O4_v, last]

        return output, middle_x

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=UNet()
    print(net(x).shape)