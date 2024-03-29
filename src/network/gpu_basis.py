import torch
from torch import nn

torch.device('cpu')
torch.cuda.device('cuda')
print(torch.cuda.device_count())


def try_gpu(i=0):  # @save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  # @save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# x = torch.tensor([1, 2, 3])
# print(x.device)

if __name__ == '__main__':
    X = torch.ones(2, 3, device=try_gpu())
    net = nn.Sequential(nn.Linear(3, 1))
    net = net.to(device=try_gpu())
    net(X)
