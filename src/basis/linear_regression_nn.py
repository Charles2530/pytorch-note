from torch import nn
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# Generating the Dataset
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    # data_arrays is a list of data arrays
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    # next(iter(data_iter)) is used to get the first batch of data
    next(iter(data_iter))
    # Defining the Model
    # The model is defined by an nn.Sequential, which is a chain of layers. Since our model,
    # within this single layer, is just a linear regression, we define a single nn.Linear
    net = nn.Sequential(nn.Linear(2, 1))
    # Initializing Model Parameters
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # Defining the Loss Function
    loss = nn.MSELoss()

    # Defining the Optimization Algorithm
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    # Training
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            # Because l is a vector whose elements are the losses for each example in the
            # minibatch, calling l.backward() adds together the gradients of all the examples
            # and stores the result in the model parameters.
            trainer.zero_grad()
            # zero_grad clears the gradients of the model parameters to prevent them from
            # accumulating during multiple iterations.
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('error in estimating w:', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('error in estimating b:', true_b - b)

# Summary
# We saw how a deep network can be implemented and optimized from scratch, using just
# tensors and auto differentiation, without any need for defining layers, fancy optimizers,
# etc. This only scratches the surface, and the full deep learning libraries are much more
# comprehensive and efficient. But hopefully, this example gave you a sense for what’s
# going on underneath the hood.

# Exercises
# 1. Review the PyTorch documentation to see what loss functions and initialization methods
# are provided. Replace the loss by Huber’s loss.
# 2. How do you access the gradient of net[0].weight?
# 3. What happens if you the number of dimensions in the true_w and features arguments
# of synthetic_data?
# 4. Sometimes, gradient descent has to be called with a smaller learning rate. What
# happens if you change it to 0.1?
# 5. What happens if you attempt to calculate the derivative of loss in the algorithm?
# 6. If the number of examples cannot be divided by the batch size, what happens to the
# data_iter function’s behavior?
# 7. What would you need to change if you wanted to train on a GPU?
# 8. Why is the reshape function needed in the squared loss function?
# 9. Experiment using different learning rate to train the model. What happens?
# 10. If the number of dimensions in the true_w and features arguments of synthetic_data
# is increased, what happens to the result?
# 11. What if we implement the squared loss function using Python’s broadcasting?
# 12. Review the PyTorch documentation to see what loss functions and initialization methods
# are provided. Replace the loss by Huber’s loss.
