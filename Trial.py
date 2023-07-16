import torch

a = [
    [1,2],
    [3,4]
]

b = [
    [11,21],
    [31,41]
]

a = torch.tensor(a)
b = torch.tensor(b)

print(torch.cat([a,b],axis = 1))