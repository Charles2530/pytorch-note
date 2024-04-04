import torch
from torch import nn
from d2l import torch as d2l
# 批量归一化层
# X: 输入张量
# gamma: 拉伸参数
# beta: 偏移参数
# moving_mean: 全局平均值
# moving_var: 全局方差
# eps: 为了防止分母为0，添加的常数
# momentum: 动量
# 批量归一化层的作用是：对卷积层的输出做归一化，使得每一层的输出数据的均值为0，方差为1
# 这样可以加速模型的训练，提高模型的泛化能力,加速模型的收敛速度，但不改变模型的表达能力

# 定义BatchNorm层。


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 训练模式下使用当前的均值和方差做标准化
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            # 这里保留第二维
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下使用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var


class BatchNormalization(nn.Module):
    # num_features: 来自于上一层的输出的数量
    # num_dims: 2表示全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super(BatchNormalization, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    BatchNormalization(6, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    BatchNormalization(16, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*4*4, 120),
    BatchNormalization(120, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    BatchNormalization(84, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
X = torch.rand(size=(1, 1, 28, 28))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# output:
# loss 0.244, train acc 0.909, test acc 0.882
