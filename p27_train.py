import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import dataloader
from p27_model import *
#p27_model引入全部的内容

train_data = torchvision.datasets.CIFAR10(root="./data",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data",train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)

#看训练集和测试集有多少张 len长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))#format把括号里的替换花括号的位置
print("测试数据集的长度为：{}".format(test_data_size))#format把括号里的替换花括号的位置

#利用dataloader加载数据集
train_dataloader = DataLoader(dataset=train_data,batch_size=64)
test_dataloader = DataLoader(dataset=test_data,batch_size=64)

#创建网络模型
myd = MYD()

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(myd.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 10

#添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i+1))


    #训练步骤开始
    myd.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = myd(imgs)
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step+1
        if total_train_step%100==0:
            print("训练次数：{}，Loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    myd.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            outputs = myd(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy+accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))

    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)

    total_test_step = total_test_step +1
    #保存每一轮结果
    torch.save(myd,"p27model_{}.pth".format(i))
    print("模型已保存")
writer.close()
