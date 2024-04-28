import torch
from model import Net

model = Net()
model.load_state_dict(torch.load(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/FashionMNIST/model/model.pth'))
model.eval()
