import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from logger import Logger
import numpy as np

from FashionMNIST.budget_baseline.net import Net


def test_F_MNIST():
    confusion = np.zeros((19, 19), dtype=np.uint32)  # First index actual, second index predicted
    correct = 0
    n = 0
    N = len(test_dataset)
    for d, l in test_dataset:
        d = Variable(d.unsqueeze(0))
        outputs = net.forward(d)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print(confusion)
    F1 = 0
    for nr in range(17):
        TP = confusion[nr, nr]
        FP = sum(confusion[:, nr]) - TP
        FN = sum(confusion[nr, :]) - TP
        F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
    print('F1: ', F1)
    print('Accuracy: ', acc)
    return F1

class F_budget(Dataset):

    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0]), 1), l



net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
criterion = nn.NLLLoss()


train_dataset = F_budget(
    torchvision.datasets.FashionMNIST(root='FashionMNIST/data', train=True, download=True, transform=transform),
    'FashionMNIST/budget_baseline/train_budget_base_data.txt')
test_dataset = F_budget(
    torchvision.datasets.FashionMNIST(root='FashionMNIST/data', train=False, download=True, transform=transform),
    'FashionMNIST/budget_baseline/test_budget_base_data.txt')
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

i = 1
test_period = 500
log_period = 50
running_loss = 0.0
log = Logger()

for epoch in range(1):

    for data in trainloader:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)     
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data
        if i % log_period == 0:
            print('Iteration: ', i * 2, '\tAverage Loss: ', running_loss / log_period)
            log.log('loss', i * 2, running_loss / log_period)
            running_loss = 0
        if i % test_period == 0:
            log.log('F1', i * 2, test_F_MNIST())
        i += 1