import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

class MYD(nn.Module):
    def __init__(self):
        super(MYD, self).__init__()
        self.con1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self, x):
        x = self.con1(x)
        return x

myd = MYD()

writer = SummaryWriter("logs")

step=0

for data in dataloader:
    imgs,targets = data
    output = myd(imgs)
    print(imgs.shape)
    print(output.shape)
#torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)

    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
#torch.Size([64, 6, 30, 30])
    step=step+1

writer.close()

