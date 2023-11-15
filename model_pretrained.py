import torchvision.datasets
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained= False)#训练好的，不需要下载,网络模型参数是默认初始化的
vgg16_true = torchvision.models.vgg16(pretrained= True)#需要下载，对应参数
print(vgg16_true)

#vgg分为1000，数据集是10
train_data = torchvision.datasets.CIFAR10('./data',train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_false)