import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import torch


class TitanicDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.features = ['HomePlanet', 'CryoSleep',
                         'Destination', 'VIP']
        self.X = self.df[self.features].fillna('Unknown')
        print(self.X.shape)
        self.y = self.df['Transported'] if 'Transported' in self.df.columns else None
        self.X = pd.get_dummies(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X.iloc[idx].values
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(
            self.y.iloc[idx], dtype=torch.long) if self.y is not None else None
        return x, y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
