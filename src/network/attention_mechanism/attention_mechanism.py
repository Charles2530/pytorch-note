import torch
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
# 注意力机制
# 卷积、全连接、池化层都只考虑不随意线索
# 注意力机制则显示的考虑随意线索，随意线索被称之为查询（query）
# 每个输入是一个值（value），每个查询是一个键（key），键是一个不随意线索
# 通过注意力池化层来有偏向性的选择某些输入
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train)*5)


def f(x):
    return 2 * torch.sin(x) + x**0.8


y_train = f(x_train) + torch.normal(0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)
print(x_train.shape, y_train.shape)


def plot_kernel_reg(y_hat):
    plt.plot(x_test, y_truth, label='Truth')
    plt.plot(x_test, y_hat, label='Pred')
    plt.scatter(x_train, y_train, label='Train')
    plt.legend()
    # plt.show()


x_repeat = x_train.repeat_interleave(n_train).reshape(-1, n_train)
attention_weights = nn.functional.softmax(
    -(x_repeat - x_repeat.t())**2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
# 带参数的注意力汇聚
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape(
            -1, keys.shape[1])
        self.attention_weights = nn.functional.softmax(
            -((queries - keys)**2) * self.w, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


# 底下几行会报错
# net = NWKernelRegression()
# d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
#                   xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
# d2l.plt.show()


# 注意力分数
# 注意力分数是注意力机制的输出，是一个概率分布
# 注意力分数是一个概率分布，所有元素之和为1
# 注意力分数是query和key的相似度，注意力权重是value的softmax结果
# 两种常见的分数计算:
# 1.将query和key合并起来进入一个单输出单隐藏层的多层感知机(MLP)
# 2.直接将query和key点积

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(
                valid_lens, shape[1], dim=0)
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
# 加性注意力


class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后进行加法
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


query, key = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
value = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(
    key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
print(attention(query, key, value, valid_lens))
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
# d2l.plt.show()
