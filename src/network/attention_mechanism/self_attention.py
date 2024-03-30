import math
import torch
from torch import nn
from d2l import torch as d2l
from ..transformer import transpose_qkv, MultiHeadAttention
# 自注意力机制
# 自注意力机制是指对于一个序列，计算序列中每个元素与其他元素的相关性
# 并将这些相关性作为权重来计算一个元素的输出
# 自注意力池化层将$x_i$当做key，value，query来对序列抽取特征

# 位置编码
# 跟RNN和CNN不同，自注意力并未记录位置信息
# 位置编码将位置信息注入输入中

# 自注意力机制完全并行、最长序列为1，但对长序列计算复杂度高

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(
    num_hiddens, num_hiddens, num_hiddens, num_hiddens, 2, 0.5)
attention.eval()


class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


encoding_dim, num_steps = 32, 60
pos_encoding = PositionEncoding(encoding_dim, 0)
pos_encoding.eval()
X = torch.zeros((1, num_steps, encoding_dim))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10], xlabel='Row (position)',
         ylabel='Column (encoding dimension)', figsize=(6, 2.5),
         legend=["Col %d" % d for d in [6, 7, 8, 9]])
d2l.plt.show()
