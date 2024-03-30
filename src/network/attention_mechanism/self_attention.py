import torch
from torch import nn
from d2l import torch as d2l
# 自注意力机制
# 自注意力机制是指对于一个序列，计算序列中每个元素与其他元素的相关性
# 并将这些相关性作为权重来计算一个元素的输出
# 自注意力池化层将$x_i$当做key，value，query来对序列抽取特征
