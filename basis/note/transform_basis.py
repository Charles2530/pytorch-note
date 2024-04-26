from net_basis import Net
import matplotlib.pyplot as plt
import pickle
import os
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
from torchvision import transforms
import numpy as np

# create a transform object,transform a three-channel image to a tensor with (-1,1)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load the CIFAR10 dataset
batch_size = 4


def load_cifar_batch(filename):
    """load a batch of the CIFAR10 dataset"""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        Y = np.array(Y)
        return X, Y


def load_cifar(ROOT):
    images = []
    labels = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        batch_images, batch_labels = load_cifar_batch(f)
        images.extend(batch_images)
        labels.extend(batch_labels)
    data_train = np.array(images), np.array(labels)
    test_images, test_labels = load_cifar_batch(
        os.path.join(ROOT, 'test_batch'))
    data_test = np.array(test_images), np.array(test_labels)

    return data_train, data_test


class cifar(Dataset):
    def __init__(self, root, segment='train', transform=None):
        if segment == 'train':
            self.data = load_cifar(root)[0]
        elif segment == 'test':
            self.data = load_cifar(root)[1]
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index][0]
        if self.transform:
            data = (self.transform(data))
        else:
            data = (torch.from_numpy(data))
        label = self.data[index][1]
        return data, label

    def __len__(self):
        return len(self.data)


trainset = cifar(root='/home/charles/charles/python/pytorch/data/cifar-10-batches-py',
                 segment='train', transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=2)
testset = cifar(root='/home/charles/charles/python/pytorch/data/cifar-10-batches-py',
                segment='test', transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size))
      )

net2 = Net()
criterion2 = torch.nn.CrossEntropyLoss()
optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)

# train the network
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer2.zero_grad()

        # forward + backward + optimize
        outputs = net2(inputs)
        loss = criterion2(outputs, labels)
        loss.backward()
        optimizer2.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# predict the class label of the test dataset
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' %
      classes[labels[j]] for j in range(batch_size)))

outputs = net2(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(batch_size)))

# calculate the accuracy of the network on the whole dataset
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# save the trained model
PATH = './models/cifar_net.pth'
torch.save(net2.state_dict(), PATH)
# load the trained model
pretrained_net = torch.load(PATH)
net3 = Net()
net3.load_state_dict(pretrained_net)
