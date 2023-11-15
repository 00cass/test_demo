import torch
import torchvision

#保存方式1的加载模型
model = torch.load("vgg16_method1.pth")
# print(model)

#保存方式2的加载模型
vgg16=torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model=torch.load("vgg16_method2.pth")
print(vgg16)
