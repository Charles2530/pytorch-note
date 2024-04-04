import random
import torch
from d2l import torch as d2l

# synthetic data generation


def synthetic_data(w, b, num_examples):
    # normal is used to generate random numbers with a mean of 0 and a standard deviation of 1
    x = torch.normal(0, 1, (num_examples, len(w)))
    # @ is used for matrix multiplication,so this can be replaced by torch.matmul(x, w)+b
    y = torch.matmul(x, w)+b
    y += torch.normal(0, 0.01, y.shape)  # add noise
    return x, y.reshape((-1, 1))  # -1 means the size is inferred


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])
# set_figsize is used to set the size of the graph,default size is (3.5, 2.5)
d2l.set_figsize()
# scatter is used to draw a scatter plot
d2l.plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(), 1)
# d2l.plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w)+b


def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat-y.reshape(y_hat.shape))**2/2


def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()


if __name__ == '__main__':
    # The model is defined, we can now implement the training loop.
    lr = 0.03
    num_epochs = 3
    batch_size = 10
    net = linreg
    loss = squared_loss
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # Compute gradients and update parameters
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')

    print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
    print(f'error in estimating b: {true_b - b}')
