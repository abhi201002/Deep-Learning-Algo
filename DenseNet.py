import torch
from torch import nn

cfg = [
    [7,2,64],
    'M',
    'Dense',
    [1,1],
    'A',
    'Dense',
    [1,1],
    'A',
    'Dense',
    [1,1],
    'A',
    'Dense',
]

class Conv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, out_channels = -1):
        super().__init__()
        self.padding = 0
        
        if(kernel_size == 3):
            self.padding = 1
        elif(kernel_size == 7):
            self.padding = 3
        
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            out_channels = out_channels,
            padding = self.padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class Dense(nn.Module):
    def __init__(self, num_repeats, in_channels, k):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.num_repeats = num_repeats
        # self.layer = self.make_layers()
    def forward(self, x):
        for _ in range(self.num_repeats):
            identity = x.clone()
            x = Conv(
                in_channels= self.in_channels,
                out_channels= self.in_channels // 4,
                kernel_size= 1,
                stride= 1,
            )(x)
            x = Conv(
                in_channels= self.in_channels // 4,
                out_channels= self.k,
                kernel_size= 3,
                stride= 1,
            )(x)
            self.in_channels += self.k; 
            x = torch.cat([identity,x],axis = 1)
        return x
class DenseNet(nn.Module):
    def __init__(self, num_classes, in_channels = 3, k = 32, net = [16, 24, 12, 6]):
        super().__init__()
        self.net = net
        self.in_channels = in_channels
        self.k = k
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
        self.layer = self.make_layer()
    
    def forward(self, x):
        return self.fc(self.layer(x))
    
    def make_layer(self):
        layers = []
        for layer in cfg:
            if isinstance(layer,str):
                if layer == 'M':
                    layers.append(self.maxpool)
                elif layer == 'A':
                    layers.append(self.avgpool)
                else:
                    num = self.net.pop()
                    layers.append(
                        Dense(
                            num_repeats= num,
                            k = self.k,
                            in_channels= self.in_channels
                        )
                    )
                    self.in_channels += self.k*(num)
            else:
                if(len(layer) == 3):
                    layers.append(
                        Conv(
                            kernel_size = layer[0],
                            stride = layer[1],
                            out_channels = layer[2],
                            in_channels = self.in_channels
                        )
                    )
                    self.in_channels = 64
                else:
                    layers.append(
                        Conv(
                            kernel_size = layer[0],
                            stride = layer[1],
                            out_channels = self.in_channels // 2,
                            in_channels = self.in_channels
                        )
                    )
                    self.in_channels = self.in_channels // 2
                    
        return nn.Sequential(*layers)
    
img = torch.randn((20,3,224,224))

model = DenseNet(num_classes= 2)

res = model(img)

print(res.shape)