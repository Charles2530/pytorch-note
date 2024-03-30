import torch
import torchvision
from torch import nn
from d2l import torch as d2l
# 转置卷积
# 正常的卷积操作是将输入的特征图缩小，而转置卷积是将输入的特征图放大
# 对于卷积而言Y=X*W,对于转置卷积而言X=Y*W
# 例如：输入特征图的形状是3*3，卷积核的形状是2*2，那么输出特征图的形状是2*2
# 对于转置卷积而言，输入特征图的形状是2*2，卷积核的形状是2*2，那么输出特征图的形状是3*3
# 通过转置卷积可以将特征图放大，这样可以用于图像分割等任务


def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0]+h-1, X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i+h, j:j+w] += X[i, j]*K
    return Y


X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)

# 通过nn.ConvTranspose2d实现转置卷积
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)

# 验证
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2)
print(tconv(conv(X)).shape == X.shape)
