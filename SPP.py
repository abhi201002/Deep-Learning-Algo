import torch
from torch import nn
import math

class Maxpool(nn.Module):
    def __init__(self,size,stride):
        super().__init__()
        self.maxp = nn.MaxPool2d(kernel_size = size,stride = stride)
    def forward(self,x):
        return self.maxp(x)

class SPP(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.size = size
        self.flatten = nn.Flatten(1,3)
        # self.pad = nn.ZeroPad2d
    def forward(self,x):
        res = torch.empty(0)
        _,_,h,w = x.shape
        for layer in self.size:
            h_w = math.ceil(h/layer)
            w_w = math.ceil(w/layer)
            h_s = math.ceil(h/layer)
            w_s = math.ceil(w/layer)
            
            pad_h = h_w*layer - h
            pad_w = w_w*layer - w
            x = nn.ZeroPad2d((pad_w,0,pad_h,0))(x)
            # x = nn.ZeroPad2d((pad_h,0,pad_w,0))(x)
            output = Maxpool(size = (h_w,w_w), stride = (h_s,w_s))(x)
            res = torch.cat([res,self.flatten(output)],axis = -1) 
        return res
    
    
# x = torch.randn(512,7,7)
# x = nn.ZeroPad2d((2,2,1,1))(x)

# print(x.shape)