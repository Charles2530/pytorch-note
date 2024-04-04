import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from language_model import load_data_time_machine
from rnn import RNNModelScratch, train_ch8
from rnn_nn import RNNModel
# 长短期记忆网络(LSTM)
# 长短期记忆网络是一种常用的循环神经网络，它在循环神经网络的基础上增加了门控机制
# 门控机制可以更好地捕捉时间序列数据中时间步距离较大的依赖关系
# 忘记门：控制隐藏状态的遗忘
# 输入门：控制隐藏状态的更新
# 输出门：控制隐藏状态的输出
# LSTM的隐藏状态计算公式：
# Ct = Ft * Ct-1 + It * C~
# Ht = Ot * tanh(Ct)
# 其中Ct是记忆细胞，Ht是隐藏状态，Ft是忘记门，It是输入门，Ot是输出门，C~是候选记忆细胞
# 记忆细胞是指阅读到当前LSTM单元的信息状态或者记忆状态
# 隐藏状态是指LSTM单元的输出
# 候选记忆细胞计算公式：
# C~ = tanh(Wxc * Xt + Whc * Ht-1 + Bc)
# 忘记门计算公式：
# Ft = sigmoid(Wxf * Xt + Whf * Ht-1 + Bf)
# 输入门计算公式：
# It = sigmoid(Wxi * Xt + Whi * Ht-1 + Bi)
# 输出门计算公式：
# Ot = sigmoid(Wxo * Xt + Who * Ht-1 + Bo)
# LSTM和RNN的区别在于LSTM引入了忘记门、输入门和输出门
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆细胞参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附上梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f,
              W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho,
        b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.mm(X, W_xi) + torch.mm(H, W_hi) + b_i)
        F = torch.sigmoid(torch.mm(X, W_xf) + torch.mm(H, W_hf) + b_f)
        O = torch.sigmoid(torch.mm(X, W_xo) + torch.mm(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.mm(X, W_xc) + torch.mm(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                        init_lstm_state, lstm)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)

num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
