"""
model.py 建立网络
"""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类nn.Module的构造函数来初始化
        super(Net, self).__init__()
        # 定义第一个卷积层，输入通道为1（灰度图像），输出通道为10，卷积核大小为5*5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 定义第二个卷积层，输入通道为10（第一个卷积层的输出通道），输出通道为20，卷积核大小为5*5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 定义一个2D dropout层，用于在训练过程中随机“丢弃”一些特征图的元素，以减少过拟合
        self.conv2_drop = nn.Dropout2d()
        # 定义第一个全连接层，输入特征数为320，输出特征数为50
        self.fc1 = nn.Linear(320, 50)
        # 定义第二个全连接层，输入特征数为50，输出特征数为10（对应于10个类别的得分）
        self.fc2 = nn.Linear(50, 10)

    # 前向传播算法
    def forward(self, x):
        # 第一个卷积层的输出经过最大池化（池化核大小为2*2），然后通过ReLU激活函数
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 第二个卷积层的输出先通过dropout层，然后通过最大池化并通过ReLU激活函数
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 将二位特征图展平为一维特征向量，以便连接到全连接层 -1在view函数中表示自动计算该维度的大小，320是全连接层的输入特征数。
        x = x.view(-1, 320)
        # 第一个全连接层的输出通过ReLU函数
        x = F.relu(self.fc1(x))
        # 应用dropout到第一个全连接层的输出，确保dropout只在训练模式下激活
        x = F.dropout(x, training=self.training)
        # 计算第二个全连接层的输出
        x = self.fc2(x)
        # 计算输出的对数概率，用于后续的分类任务
        return F.log_softmax(x)
