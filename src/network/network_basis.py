import torch
from torch import nn
from torch.nn import functional as F
# 层和块

# 自定义实现MLP


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


# 上述类等价于
# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# 实现Sequential类


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

# 在正向传播函数中执行代码


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


# 参数管理
# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# X = torch.rand(size=(2, 4))
# net(X)
# print(net[2].state_dict())
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


# X = torch.rand(size=(2, 4))
# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# rgnet(X)
# print(rgnet)

# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# net.apply(init_normal)
# print(net[0].weight.data[0])

# 参数绑定
# X = torch.rand(size=(2, 4))
# shared = nn.Linear(8, 8)
# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared,
#                     nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
# net(X)
# print(net[2].weight.data[0] == net[4].weight.data[0])
# net[2].weight.data[0, 0] = 100
# print(net[2].weight.data[0] == net[4].weight.data[0])

# 自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


# layer = CenteredLayer()
# print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


# dense = MyLinear(5, 3)
# print(dense.weight)
