import torch
from torch import nn

cfg = [
    [11,4,96],
    'M',
    [5,1,256],
    'M',
    [3,1,384],
    [3,1,384],
    [3,1,256],
    'M'
]
class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernel_size):
        super().__init__()
        padding = 0
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        
        self.block = nn.Conv2d(
            in_channels = in_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.bn(self.block(x)))
    
class AlexNet(nn.Module):
    def __init__(self, num_classes, in_channels = 3):
        super().__init__()
        self.in_channel = in_channels
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216,4096),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.Linear(4096,num_classes),
            # nn.Softmax()
        )
        self.layer = self.make_layer()
        self.flatten = nn.Flatten()
    def forward(self, x):
        return self.fc(self.flatten(self.layer(x)))
    def make_layer(self):
        layers = []
        for layer in cfg:
            if isinstance(layer,list):
                layers.append(Block(in_channel = self.in_channel, out_channel= layer[2], stride = layer[1], kernel_size= layer[0]))
                self.in_channel = layer[2]
            else:
                layers.append(nn.MaxPool2d(stride = 2, kernel_size = 3))
        return nn.Sequential(*layers)
    
    
image = torch.randn(10,3,227,227)

model = AlexNet(num_classes= 10)

res = model(image)

print(res.shape)