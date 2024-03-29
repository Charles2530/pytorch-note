import torch
from torch import nn
from d2l import torch as d2l
# AlexNet的网络结构如下：
# 1. 输入层：输入的图片大小为224*224
# 2. 第一个卷积层：输入通道为3，输出通道为96，卷积核大小为11*11，激活函数为relu
# 3. 第一个池化层：池化核大小为3*3，步长为2
# 4. 第二个卷积层：输入通道为96，输出通道为256，卷积核大小为5*5，激活函数为relu
# 5. 第二个池化层：池化核大小为3*3，步长为2
# 6. 第三个卷积层：输入通道为256，输出通道为384，卷积核大小为3*3，激活函数为relu
# 7. 第四个卷积层：输入通道为384，输出通道为384，卷积核大小为3*3，激活函数为relu
# 8. 第五个卷积层：输入通道为384，输出通道为256，卷积核大小为3*3，激活函数为relu
# 9. 第三个池化层：池化核大小为3*3，步长为2
# 10. 第一个全连接层：输出通道为4096，激活函数为relu
# 11. 第二个全连接层：输出通道为4096，激活函数为relu
# 12. 输出层：输出通道为10，激活函数为relu


def alexnet():
    return nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10))


X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
net = alexnet()
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
