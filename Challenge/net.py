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

class UpSample_noc(nn.Module):
    def __init__(self,channel):
        super(UpSample_noc, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel,1,1)
        self.conv2 = nn.Conv2d(channel*2,channel,1,1)
    def forward(self,x,feature_map):
        up = F.interpolate(x,scale_factor=2,mode='nearest')
        # print('up:', up.size())
        out = self.conv1(up)
        # print('out:', out.size())
        out = torch.cat((out,feature_map),dim=1)
        out = self.conv2(out)
        return out

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

class BackEdgeConv2d(nn.Module):
    def __init__(self, kernel_size, channels, threshold, tolerance):
        super(BackEdgeConv2d, self).__init__()
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
        tolerance_low = self.kernel_size * self.kernel_size // 2 * self.tolerance
        tolerance_high = self.kernel_size * self.kernel_size // 2 * (1.0 - self.tolerance)
        black_pixels_mask = ((condition_sum >= tolerance_low) & (condition_sum <= tolerance_high)).float()
        output_img = x * (1 - black_pixels_mask)
        return output_img

class BackThres(nn.Module):
    def __init__(self, threshold):
        super(BackThres, self).__init__()
        self.threshold = threshold / 255.0

    def forward(self, x):
        output = x.clone()
        mask = output > self.threshold
        combined_mask = mask.all(dim=1, keepdim=True)
        combined_mask = combined_mask.expand_as(output)
        output[combined_mask] = 0
        return output

class BackAllConv2d(nn.Module):
    def __init__(self, kernel_size, threshold, tolerance, stride=1, padding=1):
        super(BackAllConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.tolerance = tolerance
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        C_out = x.shape[1]
        unfolded = unfolded.view(x.shape[0], C_out, self.kernel_size*self.kernel_size, -1)
        
        mask = unfolded > (self.threshold / 255)
        mask_ratio = mask.all(dim=1, keepdim=True).float().mean(dim=2, keepdim=True)
        mask_threshold = mask_ratio > self.tolerance

        mask_threshold_expanded = mask_threshold.expand(-1, C_out, self.kernel_size*self.kernel_size, -1)
        unfolded = unfolded * (~mask_threshold_expanded).float()

        unfolded = unfolded.view(x.shape[0], C_out * self.kernel_size*self.kernel_size, -1)

        output = F.fold(unfolded, output_size=(x.shape[2], x.shape[3]), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        with torch.no_grad():
            ones_input = torch.ones_like(x)
            count_overlap = F.fold(F.unfold(ones_input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), output_size=(x.shape[2], x.shape[3]), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            output = output / count_overlap.clamp(min=1)

        return output

class UNet(nn.Module):
    #ct: channel times
    def __init__(self,num_classes = 3, ct = 4):
        super(UNet, self).__init__()
        self.ba = BackAllConv2d(25, 160.0, 0.9)
        self.bc1 = BackConv2d(5, 3, 160.0, 0.8)
        # self.bc2 = BackConv2d(25, 3, 170.0, 0.8)
        # self.bec = BackEdgeConv2d(3, 3, 160.0, 0.6)
        self.bt = BackThres(170.0)
        self.c1 = Conv_Block(3,16*ct)
        self.d1 = DownSample(16*ct)
        self.c2 = Conv_Block(16*ct,32*ct)
        self.r2 = DownResidual(16*ct,32*ct)
        self.d2 = DownSample(32*ct)
        self.c3 = Conv_Block(32*ct,64*ct)
        self.r3 = DownResidual(32*ct,64*ct)
        self.d3 = DownSample(64*ct)
        self.c4 = Conv_Block(64*ct,128*ct)
        self.r4 = DownResidual(64*ct,128*ct)

        self.d4 = DownSample(128*ct)
        self.c5 = Conv_Block(128*ct,256*ct)
        self.r5 = DownResidual(128*ct,256*ct)
        
        self.d5 = DownSample(256*ct)
        self.cl1 = Conv_Block(256*ct,512*ct)
        self.rl1 = DownResidual(256*ct,512*ct)

        self.d6 = DownSample(512*ct)
        self.cl2 = Conv_Block(512*ct,1024*ct)
        self.rl2 = DownResidual(512*ct,1024*ct)

        self.ul2 = UpSample(1024*ct)
        self.cl2_2 = Conv_Block(1024*ct,512*ct)

        self.ul1 = UpSample(512*ct)
        self.cl1_1 = Conv_Block(512*ct,256*ct)

        self.u1 = UpSample(256*ct)
        self.c6 = Conv_Block(256*ct,128*ct)
        self.r6 = UpResidual(256*ct,128*ct)
        self.u2 = UpSample(128*ct)
        self.c7 = Conv_Block(128*ct, 64*ct)
        self.r7 = UpResidual(128*ct, 64*ct)
        self.u3 = UpSample(64*ct)
        self.c8 = Conv_Block(64*ct, 32*ct)
        self.r8 = UpResidual(64*ct, 32*ct)
        self.u4 = UpSample(32*ct)
        self.c9 = Conv_Block(32*ct, 16*ct)
        self.r9 = UpResidual(32*ct, 16*ct)
        self.out = nn.Conv2d(16*ct,num_classes,3,1,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self,x, view):

        # xb1 = self.bc1(x)
        # xba = self.ba(x)
        # x_both = xb1 + xba
        # x_both = self.relu(x_both)

        R1 = self.c1(x)
        R1_v = R1

        R2 = self.c2(self.d1(R1))
        R2 = self.r2(R2,R1)

        R3 = self.c3(self.d2(R2))
        R3 = self.r3(R3,R2)

        R4 = self.c4(self.d3(R3))
        R4 = self.r4(R4,R3)
        R5 = self.d4(R4)
        R5 = self.c5(R5)
        R5 = self.r5(R5,R4)
    
        # cause UNet use 224x224, but here x[960, 640]->[480, 320]->[240, 160]->[120, 80]->[60, 40]->[30, 20]->[15, 10]
        #[60, 40]->[30, 20]
        R5d = self.d5(R5)
        Rl1 = self.cl1(R5d)
        Rl1 = self.rl1(Rl1,R5)

        #[30, 20]->[15, 10]
        R6d = self.d6(Rl1)
        Rl2 = self.cl2(R6d)
        Rl2 = self.rl2(Rl2,Rl1)

        #[15, 10]->[30, 20]
        Ol1 = self.ul2(Rl2, Rl1)
        Ol1 = self.cl2_2(Ol1)

        #[30, 20]->[60, 40]
        Ol2 = self.ul1(Ol1, R5)
        Ol2 = self.cl1_1(Ol2)

        # [1024,60,40], R5, Ol 

        O1 = self.u1(Ol2, R4)
        O1 = self.c6(O1)
        # O1 = self.r6(O1,R5)

        O2 = self.c7(self.u2(O1, R3))
        # O2 = self.r7(O2,O1)

        O3 = self.c8(self.u3(O2, R2))
        # O3 = self.r8(O3,O2)

        O4 = self.c9(self.u4(O3, R1))
        # O4 = self.r9(O4,O3)

        last = self.out(O4)

        output = self.sigmoid(last)

        if view:
            R2_v = F.interpolate(R2,scale_factor=2,mode='nearest')
            R3_v = F.interpolate(R3,scale_factor=4,mode='nearest')
            R4_v = F.interpolate(R4,scale_factor=8,mode='nearest')
            R5_v = F.interpolate(R5,scale_factor=16,mode='nearest')
            Rl1_v = F.interpolate(Rl1,scale_factor=32,mode='nearest')
            Rl2_v = F.interpolate(Rl2,scale_factor=64,mode='nearest')
            Ol1_v = F.interpolate(Ol1,scale_factor=32,mode='nearest')
            Ol2_v = F.interpolate(Ol2,scale_factor=16,mode='nearest')
            O1_v = F.interpolate(O1,scale_factor=8,mode='nearest')
            O2_v = F.interpolate(O2,scale_factor=4,mode='nearest')
            O3_v = F.interpolate(O3,scale_factor=2,mode='nearest')
            O4_v = O4
            middle_x = [R1_v, R2_v, R3_v, R4_v, R5_v, Rl1_v, Rl2_v, Ol1_v, Ol2_v, O1_v, O2_v, O3_v, O4_v]
            return output, middle_x
        else:
            return output
        
class UNet_simple(nn.Module):
    def __init__(self,num_classes = 1):
        super(UNet_simple, self).__init__()
        self.c1 = Conv_Block(3,64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64,128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128,256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256,512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512,1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64,num_classes,3,1,1)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, view):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        last = self.out(O4).squeeze(1)
        # output = self.sigmoid(last)
        # output = F.log_softmax(last, dim=1)
        output = self.relu(last)

        if view:
            R1_v = R1
            R2_v = F.interpolate(R2,scale_factor=2,mode='nearest')
            R3_v = F.interpolate(R3,scale_factor=4,mode='nearest')
            R4_v = F.interpolate(R4,scale_factor=8,mode='nearest')
            R5_v = F.interpolate(R5,scale_factor=16,mode='nearest')
            O1_v = F.interpolate(O1,scale_factor=8,mode='nearest')
            O2_v = F.interpolate(O2,scale_factor=4,mode='nearest')
            O3_v = F.interpolate(O3,scale_factor=2,mode='nearest')
            O4_v = O4
            middle_x = [R1_v, R2_v, R3_v, R4_v, R5_v, O1_v, O2_v, O3_v, O4_v]
            return output, middle_x
        else:
            return output