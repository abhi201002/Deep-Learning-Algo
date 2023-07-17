import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self,num_classes, in_channels, layer = [3,6,4,3]):
        super().__init__()
        self.in_channels = 64
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = 64,
            kernel_size = 7, 
            stride = 2,
            padding = 3
        )
        self.layer1 = self.make_layer(
            size = 64,
            repeats = layer[0],
            stride = 1
        )
        self.layer2 = self.make_layer(
            size = 128,
            repeats = layer[1],
            stride = 2
        )
        self.layer3 = self.make_layer(
            size = 256,
            repeats = layer[2],
            stride = 2
        )
        self.layer4 = self.make_layer(
            size = 512,
            repeats = layer[3],
            stride = 2
        )
        self.fc = nn.Linear(512*4, num_classes)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pooling(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x
    def make_layer(self, size, repeats, stride):
        layers = []
        downsample = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = size * 4,
            kernel_size = 1,
            stride = stride,
            bias = False,
        )
        
        layers.append(block(size = size, in_channels = self.in_channels, stride = stride, downsample = downsample))
        
        self.in_channels = size * 4
        
        for _ in range(repeats - 1):
            layers.append(block(size = size, in_channels = self.in_channels,stride = 1, downsample = None))
            
        return nn.Sequential(*layers)
    
class block(nn.Module):
    def __init__(self, in_channels, size, stride, downsample = None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = size,
            kernel_size = 1,
            stride = 1
        )
        self.conv2 = nn.Conv2d(
            in_channels = size,
            out_channels = size,
            kernel_size = 3,
            stride = stride,
            padding = 1
        )
        self.conv3 = nn.Conv2d(
            in_channels = size,
            out_channels = size * 4,
            kernel_size = 1,
            stride = 1,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(size)
        self.bn2 = nn.BatchNorm2d(size * 4)
        self.downsample = downsample
        self.layer1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu
        )
        self.layer2 = nn.Sequential(
            self.conv2,
            self.bn1,
            self.relu
        )
        self.layer3 = nn.Sequential(
            self.conv3,
            self.bn2,
            self.relu
        )
    
    def forward(self,x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)
    
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x + identity
        x = self.relu(x)
        return x
        
# image = torch.randn((1,3,224,224))
# # print(image.shape)

# model = ResNet([3,6,4,3],2,3)

# res = model(image)
# print(res.shape)

# Res
