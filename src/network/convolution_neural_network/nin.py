import torch
from torch import nn
from d2l import torch as d2l
# NiN网络结构如下：
# 1. 输入层：输入的图片大小为224*224
# 2. 5个NiN块：每个NiN块包含一个卷积层和两个1*1的卷积层
# 3. 3个全连接层：第一个全连接层的输出个数是4096，第二个全连接层的输出个数是4096，最后一个全连接层的输出个数是10
# NiN块的组成规律是：卷积层的输出通道数是4096的倍数
# NiN的特点是使用了多层感知机代替全连接层，从而减少了参数数量,不容易过拟合


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


def nin():
    return nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten())


net = nin()
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# output:
# loss 0.327, train acc 0.879, test acc 0.875
