import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """LeNet-5 network structure"""

    def __init__(self):
        super(Net, self).__init__()
        # define the convolutional layer in layer 1 ,input with 32x32,convolution kernel with 6x5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # define the convolutional layer in layer 3, input is 6 feature maps, output is 16 feature maps, convolution kernel with 16x5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # define the full connection layer in layer 5, 16*5*5 input nodes, 120 output nodes
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6x6 from image dimension
        # define the full connection layer in layer 6, 120 input nodes, 84 output nodes
        self.fc2 = nn.Linear(120, 84)
        # define the full connection layer in layer 7, 84 input nodes, 10 output nodes
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # input x size (32, 32)
        # first do convolution and relu ,then do pick the maximum value in 2x2 area
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # input x size (14, 14)
        # first do convolution and relu ,then do pick the maximum value in 2x2 area
        # if the size is square, you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # input x size (5, 5)
        # flatten the feature map to a vector
        x = x.view(-1, 16 * 5 * 5)
        # input x size 400
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # instantiate the network
    net = Net()
    print(net)

    # get the parameters of the network
    params = list(net.parameters())
    # print("params:", params)
    print(len(params))

    # get the parameters of the first convolutional layer
    print(params[0].size())

    # input data is a random tensor with size 32x32,see the output of the network forward popagation
    input = torch.randn(1, 3, 32, 32)
    out = net(input)
    print(out)

    # zero the gradient buffer of all parameters and backpropagate with random gradients
    net.zero_grad()

    # backpropagate with random gradients
    out.backward(torch.randn(1, 10))

    # calculate the loss function
    criterion = nn.MSELoss()
    target = torch.randn(10)  # random target
    target = target.view(1, -1)  # transform the target to a 1x10 tensor
    output = net(input)  # calculate the output
    loss = criterion(output, target)  # calculate the loss
    print(loss)

    # zero the gradient buffer of all parameters and backpropagate with random gradients
    net.zero_grad()
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    # backpropagate
    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # update the weights
    # define the optimizer SGD and set lr=0.01
    import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # zero the gradient buffer of all parameters
    optimizer.zero_grad()
    # calculate the output
    output = net(input)
    # calculate the loss
    loss = criterion(output, target)
    # backpropagate
    loss.backward()
    # update the weights
    optimizer.step()
