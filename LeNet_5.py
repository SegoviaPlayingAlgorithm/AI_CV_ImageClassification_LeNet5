import torch
from torch import nn as nn
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5,self).__init__()
        """LeNet-5是CNN模型的开山之作，最早用于32x32x1的数字图像识别，后来sigmoid被换成Relu，性能好
        输入：img=1x28x28，为黑白图像
        输出：10个soffmax预测类别概率

        lenet5结构：  CNN--池化--CNN--池化--全连接隐藏层+relu--全连接输出层+sofmax
        """  
        # 第一卷积层   原论文 1  6
        self.conv1 = nn.Conv2d(1,6,kernel_size = 5,padding=2)     
        self.relu1 = nn.ReLU()                                  
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)                   

        # 第二卷积层 6  16
        self.conv2 = nn.Conv2d(6,16,kernel_size = 5)     
        self.relu2 = nn.ReLU()                                  
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)       
        
        #全连接-隐藏层  16*5*5  120
        self.fc3=nn.Linear(16*5*5,120)
        self.relu3=nn.ReLU()

        #全连接softmax-输出
        self.fc4=nn.Linear(120,10)
        self.soft4=nn.Softmax(dim=-1)  #一维不用指定维度，直接softmax，高维，要制定在那个维度做softmax归一化
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=x.view(x.shape[0],-1)
        x=self.fc3(x)
        x=self.relu3(x)
        x=self.fc4(x)
        x=self.soft4(x)
        return x
if __name__ =='__main__':
    model=LeNet_5()
    x=torch.randn(20,1,28,28)
    y=model.forward(x)
    real=torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])
    criterion=torch.nn.CrossEntropyLoss()
    print(criterion(y,real))

