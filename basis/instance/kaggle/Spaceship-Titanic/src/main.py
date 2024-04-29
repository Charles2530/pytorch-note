import torch
from torch.utils.data import DataLoader
import pandas as pd
from model import TitanicDataset, MLP
from predict import predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Spaceship-Titanic/data/train.csv')
train_dataset = TitanicDataset(train_df)
batch_size = 256
num_workers = 4
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
data, label = next(iter(train_loader))

model = MLP(input_size=data.size(1), hidden_size=100, num_classes=2)
model.to(device)

lr = 1e-3
max_epochs = 200
criterion = torch.nn.CrossEntropyLoss()
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


# for epoch in range(max_epochs):
#     train(epoch)
torch.save(model.state_dict(),
           '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Spaceship-Titanic/model/model.pth')
test_df = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Spaceship-Titanic/data/test.csv')
test_dataset = TitanicDataset(test_df)
predictions = []
for index, (data, _) in enumerate(test_dataset):
    data = data.unsqueeze(0)
    pred = predict(input_size=data.size(1), hidden_size=100,
                   num_classes=2, data=data)
    predictions.append(pred)
result = pd.DataFrame(
    {'PassengerId': test_df.PassengerId, 'Transported': predictions})
result.to_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Spaceship-Titanic/data/MLP_predictions.csv', index=False)
