import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
# GoogleNet网络结构如下：
# GoogleNet又称为Inception网络，是一个22层的深度卷积神经网络
# 1. 输入层：输入的图片大小为224*224
# 2. 9个Inception块：每个Inception块包含数个卷积层，后接一个最大池化层
# 3. 1个全连接层：输出个数是10
# Inception块的组成规律是：卷积层的输出通道数是4096的倍数
# Inception块的结构如下：
# 1. 1*1卷积层：输出通道数是64
# 2. 1*1卷积层：输出通道数是128
# 3. 1*1卷积层：输出通道数是192
# 4. 3*3卷积层：输出通道数是96
# 5. 5*5卷积层：输出通道数是16
# 6. 最大池化层：输出通道数是32
# Inception块的优点是：可以使用不同大小的卷积核来提取不同尺寸的特征，从而提高模型的泛化能力，模型
# 的参数数量也会减少，计算复杂度低


class Incption(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Incption, self).__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


def googleNet():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b3 = nn.Sequential(Incption(192, 64, (96, 128), (16, 32), 32),
                       Incption(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b4 = nn.Sequential(Incption(480, 192, (96, 208), (16, 48), 64),
                       Incption(512, 160, (112, 224), (24, 64), 64),
                       Incption(512, 128, (128, 256), (24, 64), 64),
                       Incption(512, 112, (144, 288), (32, 64), 64),
                       Incption(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(Incption(832, 256, (160, 320), (32, 128), 128),
                       Incption(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten())
    return nn.Sequential(b1, b2, b3, b4, b5)


X = torch.rand(size=(1, 1, 96, 96))
net = googleNet()
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# output:
# loss 0.234, train acc 0.911, test acc 0.892
