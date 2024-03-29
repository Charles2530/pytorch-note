import torch
from torch import nn
from d2l import torch as d2l
# 池化层
# 请注意，这里的池化层没有权重参数。作为一种特殊层，其主要目的是缩减数据的维度，从而减少模型的计算量。


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))


# 调用nn模块中的二维最大池化层MaxPool2d，我们构造一个池化层实例pool2d并对X进行前向计算。
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
pool2d = nn.MaxPool2d(3)
# 由于池化层中没有模型参数，所以我们不需要调用参数初始化函数。
# 下面的前向计算输出了X的最大池化层的结果。
print(pool2d(X))
