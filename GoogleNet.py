import torch 
from torch import nn
cfg = [
    [7,2,112],
    'M',
    [3,1,56],
    'M',
    [64, 96, 128, 16, 32, 32],
    [128, 128, 192, 32, 96, 64],
    'M',
    [192, 96, 208, 16, 48, 64],
    [160, 112, 224, 24, 64, 64],
    [128, 128, 256, 24, 64, 64],
    [112, 144, 288, 32, 64, 64],
    [256, 160, 320, 32, 128, 128],
    'M',
    [256, 160, 320, 32, 128, 128],
    [384, 192, 384, 48, 128, 128],
]

class Conv(nn.Module):
    def __init__(self, kernel_size, stride, in_channel, out_channel):
        super().__init__()
        self.padding = 0
        if(kernel_size == 3):
            self.padding = 1
        elif kernel_size == 7:
            self.padding = 3
        else:
            self.padding = 2
        self.conv = nn.Conv2d(
            kernel_size = kernel_size,
            stride = stride,
            in_channels = in_channel,
            out_channels = out_channel,
            padding = self.padding
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Inception(nn.Module):
    def __init__(self, in_channel, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, maxp):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.layer1 = nn.Sequential(
            Conv(
                kernel_size = 1,
                stride = 1,
                out_channel= out_1x1,
                in_channel= in_channel
            )
        )
        self.layer2 = nn.Sequential(
            Conv(
                kernel_size = 1,
                stride= 1,
                in_channel= in_channel,
                out_channel= red_3x3
            ),
            Conv(
                kernel_size = 3,
                stride= 1,
                in_channel= red_3x3,
                out_channel= out_3x3
            )
        )
        self.layer3 = nn.Sequential(
            Conv(
                kernel_size = 1,
                stride= 1,
                in_channel= in_channel,
                out_channel= red_5x5
            ),
            Conv(
                kernel_size = 5,
                stride= 1,
                in_channel= red_5x5,
                out_channel= out_5x5
            )
        )
        self.layer4 = nn.Sequential(
            self.maxpool,
            Conv(
                kernel_size = 1,
                stride= 1,
                in_channel= in_channel,
                out_channel= maxp
            )
        )
    
    def forward(self, x):
        res = torch.cat([self.layer1(x), self.layer2(x), self.layer3(x), self.layer4(x)], axis = 1)
        return res

class GoogleNet(nn.Module):
    def __init__(self, num_classes = 2, in_channels = 3):
        super().__init__()
        self.in_channel = in_channels
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.layer = self.make_layer()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024,num_classes)
        )
    def forward(self, x):
        return self.fc(self.avg(self.layer(x)))
    
    def make_layer(self):
        layers = []
        for layer in cfg:
            if isinstance(layer,list):
                if(len(layer) == 3):
                    layers.append(
                        Conv(
                            kernel_size = layer[0],
                            stride = layer[1],
                            in_channel = self.in_channel,
                            out_channel = layer[2]
                        )
                    )
                    self.in_channel = layer[2]
                else:
                    layers.append(
                        Inception(
                            in_channel= self.in_channel,
                            out_1x1= layer[0],
                            red_3x3= layer[1],
                            out_3x3= layer[2],
                            red_5x5= layer[3],
                            out_5x5= layer[4],
                            maxp= layer[5],
                        )
                    )
                    self.in_channel = layer[0] + layer[2] + layer[4] + layer[5]
            else:
                layers.append(self.maxpool)
                
        return nn.Sequential(*layers)
    
img = torch.randn((20,3,224,224))

model = GoogleNet()

res = model(img)

print(res.shape)