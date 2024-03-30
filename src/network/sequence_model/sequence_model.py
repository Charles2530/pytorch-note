import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
# 序列模型：RNN、LSTM、GRU
# RNN: Recurrent Neural Network
# LSTM: Long Short-Term Memory
# GRU: Gated Recurrent Unit
# 序列模型常用于处理序列数据，如时间序列数据、文本数据等
# 序列模型的输入是一个序列，输出也是一个序列
T = 1000
time = torch.arange(1, T+1, dtype=torch.float32)
x = torch.sin(0.01*time)+torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[0, 1000], figsize=(6, 3))
# d2l.plt.show()
tau = 4
features = torch.zeros((T-tau, tau))
for i in range(tau):
    features[:, i] = x[i:T-tau+i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 训练数据集
train_iter = d2l.load_array(
    (features[:n_train], labels[:n_train]), batch_size, is_train=True)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


# 损失函数
loss = nn.MSELoss()


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(
            f'epoch {epoch+1}, loss {d2l.evaluate_loss(net, train_iter, loss)}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)
# 预测
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()],
         'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()
