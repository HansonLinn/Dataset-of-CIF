import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import numpy as np

#--------------------------------------------FSCNet-------------------------------------------
class Class_FSCNet(nn.Module):  #m2 conv3*3 
    def __init__(self, img_ch=1, output_ch=10):
        super(Class_FSCNet, self).__init__()
        self.Conv1 = nt_conv_block(ch_in=img_ch, ch_out=64,pad=1)
        self.Conv2 = nt_conv_block(ch_in=64, ch_out=128,pad=1)
        self.Conv3 = nt_conv_block(ch_in=128, ch_out=256,pad=1)
        self.Conv4 = nt_conv_block(ch_in=256, ch_out=512,pad=1)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up4 = nt_up_conv(ch_in=512, ch_out=256,pad=1)
        self.Up_conv4 = nt_conv_block(ch_in=512, ch_out=256)

        self.Up3 = nt_up_conv(ch_in=256, ch_out=128,pad=5)
        self.Up_conv3 = nt_conv_block(ch_in=256, ch_out=128)

        self.Up2 = nt_up_conv(ch_in=128, ch_out=64,pad=5)
        self.Up_conv2 = nt_conv_block(ch_in=128, ch_out=64,pad=1)

        self.cls_modual = nn.Sequential(
            nn.Conv2d(64, 64, (3,1), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64,  (1,3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            nn.Conv2d(64, 64, (3,1), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64,  (1,3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, output_ch, kernel_size=1, stride=1)
        )

        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x) #1，64，384，384

        x2 = self.Maxpool(x1) #1，64，192，192
        x2 = self.Conv2(x2) #1，128，192，192

        x3 = self.Maxpool(x2) #1，128，96，96
        x3 = self.Conv3(x3) #1，256，96，96

        x4 = self.Maxpool(x3) #1，256，48，48
        x4 = self.Conv4(x4) #1，512，48，48

        d4 = self.Up4(x4) #1，256，96，96
        d4 = torch.cat((x3, d4), dim=1) #1，512，96，96
        d4 = self.Up_conv4(d4)  #1，256，92，92

        d3 = self.Up3(d4) #1，128，184，184

        d3 = torch.cat((x2, d3), dim=1) #1，256，192，192
        d3 = self.Up_conv3(d3) #1，256，188，188

        d2 = self.Up2(d3) #1，64，384，384
        d2 = torch.cat((x1, d2), dim=1) #1，128，384，384
        d2 = self.Up_conv2(d2) #1，64，384，384

        cls = self.cls_modual(d2)
        cls = cls.view(cls.size(0),-1)

        d1 = self.Conv_1x1(d2)

        return cls,d1
    
#--------------------------------------------FSCNet-------------------------------------------

    
class nt_conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,pad=0):
        super(nt_conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, (3,1),stride=1,padding=pad,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            #384+2
            nn.Conv2d(ch_out, ch_out, (1,3),stride=1,padding=pad,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            #384+2+2
            nn.Conv2d(ch_out, ch_out, (3,1),stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch_out, ch_out, (1,3),stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )


    def forward(self,x):
        x = self.conv(x)
        return x
    
class nt_up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,pad=0):
        super(nt_up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ch_in, ch_out, (3,1),stride=1,padding=pad,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(ch_out, ch_out, (1,3),stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)

        )

    def forward(self,x):
        x = self.up(x)
        return x
    
#----------------------------------------------------LossFunction-------------------------------------------------------------------------------#

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

