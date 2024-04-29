import torch
from model import MLP
import pandas as pd


def predict(input_size, hidden_size, num_classes, data):
    model = MLP(input_size=input_size, hidden_size=hidden_size,
                num_classes=num_classes)
    model.load_state_dict(torch.load(
        '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Spaceship-Titanic/model/model.pth'))
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = torch.argmax(output, 1)
    return bool(pred.item())
