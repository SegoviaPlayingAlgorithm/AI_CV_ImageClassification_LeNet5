#环境配置
import torch
torch.manual_seed(666)
from LeNet_5 import LeNet_5 as LeNet_5
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# 全局变量
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': '10'}
matplotlib.rc("font", **font)
cor=0
num=0

###################         一、获取数据

#数据集： 载入MNIST训练集,使用dataloader把数据打散分批次，用于后面的训练,经测试一共3000个批次，60000张图片
'''dataloader的介绍
DataLoader，它是PyTorch中数据读取的一个重要接口，该接口定义在dataloader.py中，
只要是用PyTorch来训练模型基本都会用到该接口，该接口的目的：将自定义
的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的神秘迭代器（我真不知道类型名）
可以用for查看迭代器内容，迭代器元素是一个list，list【0】为输入，list【1】标签。
'''

dataset = datasets.MNIST(root='.',                      # 数据集目录
                               train=True,                    # 训练集标记
                               transform=transforms.ToTensor(),  # 转为Tensor变量
                               download=True
                               )

loader = DataLoader(dataset=dataset,  # 数据集加载
                          shuffle=True,           # 随机打乱数据
                          batch_size=100,drop_last=True)  # 批处理量20,不足20个丢了
data=iter(loader)
Xtest=[]
Ytest=[]
for i in range(500):
    next(data)
for i in range(100):
    temp=next(data)
    Xtest.append(temp[0])
    Ytest.append(temp[1])
model=LeNet_5()
model.load_state_dict(torch.load('.\modelpara.pth', map_location='cpu'))
for i in range(100):
    predict=model(Xtest[i])
    for j in range(100):
        if( int(torch.argmax(predict[j])) ==int(Ytest[i][j])  ):
            cor+=1
        num+=1
print("预测集准确率为"+str( 100*(cor/num) )+"%")

#结果正确率94.29%