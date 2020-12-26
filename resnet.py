import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

    
class SmallResBlock(nn.Module):
    
    def __init__(self, c1, c2, s=1, downsample=False):
        """
        Basic building block of resnet. Has three convolutions and a skip layer. Optionally can downsample by a factor of 2.
        
        :params:
        c1: number of input channels
        c2: number of output channels
        k: filter size
        s: stride
        downsample: decide whether or not to downsample
        """
        super(SmallResBlock, self).__init__()
        self.s = s
        self.d = downsample
        
        self.activation = nn.ReLU(inplace=True)
    
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        
        if self.d:
            self.downsample = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=1, stride=s, bias=False),
                                            nn.BatchNorm2d(c2))

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.d:
            identity = self.downsample(identity)
        
        x += identity
        self.activation(x)

        return x

    
class ResNet34_fmap(nn.Module):
    
    def __init__(self):
        """
        ResNet534 feature map, for use with feature pyramid style SSD
        """
        super(ResNet34_fmap, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        layer1_list = []
        for i in range(3):
            layer1_list.append(SmallResBlock(64, 64))
            
        layer2_list = []
        layer2_list.append(SmallResBlock(64, 128, s=2, downsample=True))
        for i in range(3):
            layer2_list.append(SmallResBlock(128, 128))
            
        layer3_list = []
        layer3_list.append(SmallResBlock(128, 256, s=2, downsample=True))
        for i in range(5):
            layer3_list.append(SmallResBlock(256, 256))
            
        layer4_list = []
        layer4_list.append(SmallResBlock(256, 512, s=2, downsample=True))
        for i in range(2):
            layer4_list.append(SmallResBlock(512, 512))
        
        self.layer1 = nn.Sequential(*layer1_list)
        self.layer2 = nn.Sequential(*layer2_list)
        self.layer3 = nn.Sequential(*layer3_list)
        self.layer4 = nn.Sequential(*layer4_list)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        self.activation(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        fmap0 = self.layer2(x)
        fmap1 = self.layer3(fmap0)
        fmap2 = self.layer4(fmap1)
        
        
        return fmap0, fmap1, fmap2


class ResNet18_fmap(nn.Module):
    
    def __init__(self):
        """
        ResNet534 feature map, for use with feature pyramid style SSD
        """
        super(ResNet18_fmap, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        layer1_list = []
        for i in range(2):
            layer1_list.append(SmallResBlock(64, 64))
            
        layer2_list = []
        layer2_list.append(SmallResBlock(64, 128, s=2, downsample=True))
        for i in range(1):
            layer2_list.append(SmallResBlock(128, 128))
            
        layer3_list = []
        layer3_list.append(SmallResBlock(128, 256, s=2, downsample=True))
        for i in range(1):
            layer3_list.append(SmallResBlock(256, 256))
            
        layer4_list = []
        layer4_list.append(SmallResBlock(256, 512, s=2, downsample=True))
        for i in range(1):
            layer4_list.append(SmallResBlock(512, 512))
        
        self.layer1 = nn.Sequential(*layer1_list)
        self.layer2 = nn.Sequential(*layer2_list)
        self.layer3 = nn.Sequential(*layer3_list)
        self.layer4 = nn.Sequential(*layer4_list)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        self.activation(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        fmap0 = self.layer2(x)
        fmap1 = self.layer3(fmap0)
        fmap2 = self.layer4(fmap1)
        
        
        return fmap0, fmap1, fmap2














    
