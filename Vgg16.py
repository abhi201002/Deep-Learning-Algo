import torch 
from torch import nn
from SPP import SPP

cfg = [
    [64,3,2],
    "M",
    [128,3,2],
    "M",
    [256,3,2],
    [256,1,1],
    "M",
    [512,3,2],
    [512,1,1],
    "M",
    [512,3,2],
    [512,1,1],
    "M",
]
class Conv(nn.Module):
    def __init__(self,in_channel,out_channel,size,padding,stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channel,
            out_channels = out_channel,
            kernel_size = size,
            padding = padding,
            stride = stride
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class Vgg16(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size = (2,2),stride = 2)
        # self.conv1 = nn.Conv2d(kernel_size = (1,1),stride = 1) 
        # self.conv3 = nn.Conv2d(kernel_size = (3,3),stride = 1,padding = 1) 
        # self.conv_block1 = nn.Sequential(
        #     self.conv3,
        #     self.conv3,
        #     self.conv1,
        # )
        # self.conv_block2 = nn.Sequential(
        #     self.conv3,
        #     self.conv3,
        #     self.conv1,
        # )
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features = 512 * 7 * 7,out_features = 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features = 4096,out_features = 1000),
            nn.ReLU(),
            nn.Linear(in_features = 1000,out_features = num_classes)
        )
        # self.spp = SPP([4,2,1])
        # self.flatten = nn.Flatten(0,2)
    def forward(self,x):
        in_channel = 3
        for layer in cfg:
            if isinstance(layer,str):
                x = self.max_pool(x)
            else:
                for _ in range(layer[2]):
                    x = Conv(
                        in_channel = in_channel,
                        out_channel = layer[0],
                        size = layer[1],
                        stride = 1,
                        padding = 1 if layer[1] == 3 else 0
                    )(x)
                    in_channel = layer[0]
        # x = self.spp(x)  
        # return self.fc(self.flatten(x))
        return self.fc(x)
    
model = Vgg16(num_classes=2)
img = torch.randn((1,3,224,224))
res = model(img)
# # res = torch.empty(0)
# # a = torch.tensor([1,2,3])
# # res = torch.cat([res,a],axis = 0)
print(res.shape)