import torch
from torch import nn


class Meng(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input +1
        return output


meng=Meng()
x=torch.tensor(1.0)
output = meng(x)
print(output)