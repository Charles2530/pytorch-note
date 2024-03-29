import torch
from torch import nn

# 二维卷积层, 1个输入通道, 1个输出通道, 3x3的卷积核, 填充为1


def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 排除不关心的前两维：批量和通道
    return Y.reshape(Y.shape[2:])


# 请注意，这里的卷积核权重包括偏差。
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

# 二维卷积层, 1个输入通道, 1个输出通道, 5x3的卷积核, 填充为(2, 1)
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

# 步幅
# 二维卷积层, 1个输入通道, 1个输出通道, 3x3的卷积核, 填充为1, 步幅为2
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

# 二维卷积层, 1个输入通道, 1个输出通道, 3x3的卷积核, 填充为1, 步幅为2x1
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=(2, 1))
print(comp_conv2d(conv2d, X).shape)
