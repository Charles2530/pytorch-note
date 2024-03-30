import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from language_model import load_data_time_machine
from rnn import RNNModelScratch, train_ch8
# 门控循环单元(GRU)
# 门控循环单元是一种常用的循环神经网络，它在循环神经网络的基础上增加了门控机制
# 门控机制可以更好地捕捉时间序列数据中时间步距离较大的依赖关系
# 更新门：控制隐藏状态的更新
# 重置门：控制隐藏状态的重置
# 候选隐藏状态：计算候选隐藏状态
# GRU的隐藏状态计算公式：
# Ht = Zt * Ht-1 + (1 - Zt) * Ht~
# 其中Zt是更新门，Ht-1是上一个时间步的隐藏状态，Ht~是候选隐藏状态
# 候选隐藏状态计算公式：
# Ht~ = tanh(Wxt * Xt + Rht-1 * Ht-1 + Bx)
# 更新门计算公式：
# Zt = sigmoid(Wxz * Xt + Rhz-1 * Ht-1 + Bz)
# 重置门计算公式：
# Rt = sigmoid(Wxr * Xt + Rhr-1 * Ht-1 + Br)
# GRU和RNN的区别在于GRU引入了更新门和重置门
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    # 更新门参数
    W_xz, W_hz, b_z = three()
    # 重置门参数
    W_xr, W_hr, b_r = three()
    # 候选隐藏状态参数
    W_xh, W_hh, b_h = three()
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附上梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.mm(X, W_xz) + torch.mm(H, W_hz) + b_z)
        R = torch.sigmoid(torch.mm(X, W_xr) + torch.mm(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.mm(X, W_xh) + torch.mm(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                        init_gru_state, gru)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# output:
# perplexity 1.1, 15743.6 tokens/sec on cuda:0
