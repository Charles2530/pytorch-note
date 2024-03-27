import torch


def class1():
    print("class1")
    a = torch.ones((2, 5, 4))
    print(a.shape)
    print(a.sum(axis=0))
    print(a.sum(axis=1, keepdim=True))
    # keepdim=True的作用是保持原有的维度，所以axis=1的维度是5，保持后变为1。
    x = torch.arange(4.0)
    x.requires_grad_(True)
    y = 2*torch.dot(x, x)
    y.backward()
    print(x.grad)


if __name__ == '__main__':
    class1()
