import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import writer

input = torch.tensor([
    [1, -0.5], [-1, 3]
])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./data",train=False,download=True,
                                       transform = torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

class MYD(nn.Module):
    def __init__(self):
        super(MYD, self).__init__()
        self.relu1 = ReLU()  ##inplace指的是是否原地操作，默认是false
        self.sigmoid1 = Sigmoid()  ##inplace指的是是否原地操作，默认是false


    def forward(self, input):
        output = self.sigmoid1(input)
        return output


myd = MYD()

writer = SummaryWriter("./logs_sig")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output=myd(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()
