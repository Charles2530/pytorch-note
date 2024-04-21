"""Multilayer Perceptron."""
import torch
from torch import nn
from d2l import torch as d2l
from softmax_regression import train_ch3

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# nn.Parameter is a kind of Tensor, and will be automatically registered as module's parameter
# randn returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    # in this network,there are two hidden layers
    X = X.reshape((-1, num_inputs))
    # between layers, we use the ReLU activation function
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)


loss = nn.CrossEntropyLoss()
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
