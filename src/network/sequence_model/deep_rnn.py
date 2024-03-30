import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from language_model import load_data_time_machine
from rnn import RNNModelScratch, train_ch8
from rnn_nn import RNNModel
# 深度循环神经网络
# 深度循环神经网络是一种常用的序列模型
# 深度循环神经网络的隐藏状态可以捕捉时间序列数据中的模式
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)
