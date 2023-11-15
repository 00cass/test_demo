import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset  = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),
                                        download=True)

dataloader = DataLoader(dataset,batch_size = 64,drop_last=True)

class MYD(nn.Module):
    def __init__(self):
        super(MYD,self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output

myd = MYD()


for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = myd(output)
    print(output.shape)
