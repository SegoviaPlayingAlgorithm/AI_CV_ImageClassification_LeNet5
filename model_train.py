#环境配置
import torch
from LeNet_5 import LeNet_5 as LeNet_5
torch.manual_seed(666)
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# 设置plot中文字体
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': '10'}
matplotlib.rc("font", **font)

###################         一、获取数据

#数据集： 载入MNIST训练集,使用dataloader把数据打散分批次，用于后面的训练,经测试一共3000个批次，60000张图片
'''dataloader的介绍
DataLoader，它是PyTorch中数据读取的一个重要接口，该接口定义在dataloader.py中，
只要是用PyTorch来训练模型基本都会用到该接口，该接口的目的：将自定义
的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的神秘迭代器（我真不知道类型名）
可以用for查看迭代器内容，迭代器元素是一个list，list【0】为输入，list【1】标签。
'''

train_dataset = datasets.MNIST(root='.',                      # 数据集目录
                               train=True,                    # 训练集标记
                               transform=transforms.ToTensor(),  # 转为Tensor变量
                               download=True
                               )

train_loader = DataLoader(dataset=train_dataset,  # 数据集加载
                          shuffle=True,           # 随机打乱数据
                          batch_size=100,drop_last=True)  # 批处理量20,不足20个丢了

data=iter(train_loader)
'''访问迭代器里面的数据---next(data)[0]得到的是
torch.Size([20, 1, 28, 28])的一个tensor，也就是各个维度信息为像素点的： 图片号，通道号，（深度），高度，宽度
一通道也就是灰度图
这也是torch里面图片的标准格式： Num x Channel x Depth x Height x Width（二维图没有depth）

'''

#可视化，展示数据集20张图片
# 辅助函数-展示图像
'''
numpy图片格式是H x W x C，因此当tensor转化成np的数组，numpy的0，1，2维由tensor图片的1，2，0维构成
强制转换是没有帮你做维度变换的，自己调用np.tranpose方法如下：
'''
def imshow(img,title=None):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))   #转换：tensor-->np.array
    plt.title(title)
    plt.show()


#imshow(make_grid(next(data)[0],nrow=5,padding=2,pad_value=0))#展示20张图片



###########################    二、超参
epochs=5
learn_rate=0.01
weight_decay=0.002
#  正则化0.02的衰减，会导致模型根本无法训练，而0.002则可以训练，可以看出，正则化超参数必须根据具体问题改动，
#  CV问题因为网络很深，参数范数很大，使用推荐值0.02实际上就相当于一些小网络取权重衰减0.2甚至2了，过大的
#  惩罚导致W分布在0附近，无法接近最优W，因而预测损失一直下不去






############################     三、训练和预测
Loss_history=[]
model=LeNet_5()
optimizer=torch.optim.Adam(model.parameters(),lr=learn_rate,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8) # α初始0.01，最终衰减到0.00003
criterion=nn.CrossEntropyLoss()
for epoch in range(0,epochs):
    data=iter(train_loader)
    print("--------------epochs="+str(epoch)+"-------------------")
    for batch_index in range(0,500):
        optimizer.zero_grad()
        x,y=next(data)#一批100张图片,一共训练50000张
        predict=model(x)
        #可以看到如果批次大于GPU容量，为了获得自动微分计算图中所需的x数据，就要在训练
        #中让GPU进行io操作，大大降低速度，batchsize以塞满GPU容量为准
        batch_loss=criterion(predict,y)
        batch_loss.backward()
        optimizer.step()
        Loss_history.append(float(batch_loss))
        if batch_index%50==0: print(batch_loss)
#存储参数
torch.save(model.state_dict(),'.\modelpara.pth')
plt.plot(Loss_history)
plt.xlabel("100张/批次")
plt.ylabel("损失函数")
plt.show()












