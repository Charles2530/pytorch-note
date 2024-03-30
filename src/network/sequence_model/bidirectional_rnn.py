import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from language_model import load_data_time_machine
from rnn import RNNModelScratch, train_ch8
from rnn_nn import RNNModel
# 双向循环神经网络
# 双向循环神经网络是一种常用的序列模型
# 双向循环神经网络通过反向更新的隐藏层来利用方向时间信息，通常用来对序列抽取特征、填空而不是预测未来
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
# 这里开启bidirectional=True是错误示例，不可以预训练模型开启双向循环神经网络，因为其不负责预测
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
