from model import Net
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import os
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
num_workers = 4  # Number of processes to use for data loading
lr = 1e-4
epochs = 20

# 首先设置数据变换
image_size = 28
data_transform = transforms.Compose([
    # transforms.ToPILImage(),  # 转换为PIL对象
    transforms.Resize(image_size),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
])

# 加载数据集
train_dataset = FashionMNIST(root='/home/charles/charles/python/pytorch/project/basis/instance/kaggle/FashionMNIST/data/',
                             train=True,
                             transform=data_transform,
                             download=True)
test_dataset = FashionMNIST(root='/home/charles/charles/python/pytorch/project/basis/instance/kaggle/FashionMNIST/data/',
                            train=False,
                            transform=data_transform,
                            download=True)

# 自定义数据集(读取csv文件)


class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        # df: pandas dataframe
        self.df = df
        self.transform = transform
        self.images = self.df.iloc[:, 1:].values.astype(np.uint8)
        self.labels = self.df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


# train_df = pd.read_csv(
#     '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/FashionMNIST/data/fashion-mnist_train.csv')
# test_df = pd.read_csv(
#     '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/FashionMNIST/data/fashion-mnist_test.csv')
# train_dataset = FMDataset(train_df, transform=None)
# test_dataset = FMDataset(test_df, transform=None)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers)

image, label = next(iter(train_loader))
plt.imshow(image[0][0], cmap='gray')
# plt.show()

# 定义模型


model = Net()
model.to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


def evaluate(epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels = np.concatenate(gt_labels)
    pred_labels = np.concatenate(pred_labels)
    accuracy = np.sum(gt_labels == pred_labels) / len(gt_labels)
    print('Epoch: {} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
        epoch, val_loss, accuracy))


for epoch in range(1, epochs+1):
    train(epoch)
    evaluate(epoch)

# gpu_info = !nvidia-smi -i 0
# gpu_info = '\n'.join(gpu_info)
# print(gpu_info)

save_path = '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/FashionMNIST/model/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save(model.state_dict(), save_path+'model.pth')
