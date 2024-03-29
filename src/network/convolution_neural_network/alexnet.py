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

# AlexNet和LeNet的区别：
# 1. AlexNet使用了更多的卷积层和全连接层，使得网络更深
# 2. AlexNet使用了丢弃法来控制全连接层的模型复杂度
# 3. AlexNet引入了ReLU激活函数来增加非线性
# 4. AlexNet使用MaxPooling来控制过拟合


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

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# output:
# loss 0.327, train acc 0.880, test acc 0.877
