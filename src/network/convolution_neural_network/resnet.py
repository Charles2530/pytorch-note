import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
# 残差网络(ResNet)
# ResNet网络结构如下：
# ResNet网络是一个152层的深度卷积神经网络
# 1. 输入层：输入的图片大小为224*224
# 2. 4个残差块：每个残差块包含数个卷积层，后接一个最大池化层
# 3. 1个全连接层：输出个数是10
# 残差块的组成规律是：卷积层的输出通道数是4096的倍数
# 残差块的结构如下：
# 1. 1*1卷积层：输出通道数是64
# 2. 1*1卷积层：输出通道数是128
# 3. 1*1卷积层：输出通道数是192
# 4. 3*3卷积层：输出通道数是96
# 5. 5*5卷积层：输出通道数是16
# 6. 最大池化层：输出通道数是32
# ResNet的基本思想是：通过跨层的数据通道，使得网络可以学习残差函数，从而能够训练出更深的网络
# ResNet的优点是：可以训练出更深的网络，模型的参数数量也会减少，计算复杂度低
# ResNet使用+号来连接跨层的数据通道, 使得网络可以学习残差函数
# 1. 每个残差块的第一个卷积层的步幅为2，这样可以减小高和宽
# 2. 每个残差块的第一个卷积层的输出通道数是上一个残差块的输出通道数的4倍
# 3. 每个残差块的第一个卷积层的输出高和宽是上一个残差块的输出高和宽的一半

# 残差块
# 残差块使得训练较为深的卷积神经网络成为可能


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        # 1*1卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # 1*1卷积层
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 1*1卷积层
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        # 第一个残差块的步幅为2,如果是第一个残差块，需要减小高和宽
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                       use_1x1conv=True, strides=2))
        else:
            # 其他情况下，保持高和宽
            blk.append(Residual(num_channels, num_channels))
    return blk


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# output:
# loss 0.008, train acc 0.997, test acc 0.924
