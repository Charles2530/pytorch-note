import torch
from torch import nn
from d2l import torch as d2l
# VGG网络结构如下：
# 1. 输入层：输入的图片大小为224*224
# 2. 5个卷积块：每个卷积块包含数个卷积层，后接一个最大池化层
# 3. 3个全连接层：第一个全连接层的输出个数是4096，第二个全连接层的输出个数是4096，最后一个全连接层的输出个数是10
# VGG块的组成规律是：卷积层的个数越来越多，同时卷积层的输出通道数也越来越多，卷积层的输出通道数是4096的倍数


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# VGG网络结构, conv_arch是一个元组列表，每个元组中的两个元素分别代表卷积层的个数和输出通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


net = vgg(conv_arch)
X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

radio = 4
small_conv_arch = [(pair[0], pair[1] // radio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# output:
# loss 0.319, train acc 0.883, test acc 0.879
