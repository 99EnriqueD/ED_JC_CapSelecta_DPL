import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from logger import Logger
import numpy as np
from graphs.graphs import save_data, clear_file, save_cm

from FashionMNIST.budget.budget_baseline.net import Net

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

if __name__ == '__main__':

    clear_file("budget_baseline_F1.txt")
    clear_file("budget_baseline_acc.txt")
    clear_file("budget_baseline_dist.txt")

    num_classes = 17

    labelMap = {0:20,1:30,2:35,3:40,4:45,5:50,6:55,7:60,8:65,9:70,10:75,11:80,12:85,13:90,14:100,15:110,16:120}

    def test_F_MNIST(iteration):
        confusion = np.zeros((num_classes, num_classes), dtype=np.uint32)  # First index actual, second index predicted
        correct = 0
        n = 0
        total_distance = 0
        for d, l in test_dataset:
            d = d.to(device)
            d = Variable(d.unsqueeze(0))
            outputs = net.forward(d)
            _, out = torch.max(outputs.data, 1)
            c = int(out.squeeze())
            confusion[l, c] += 1
            if c == l:
                correct += 1
            n += 1
            total_distance += abs(labelMap[c] - labelMap[l])
        acc = correct / n
        print(confusion)
        save_cm(confusion,"budget_baseline_cm.txt")

        average_distance = total_distance / n

        F1 = 0
        for nr in range(num_classes):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
        print('F1: ', F1)
        print('Accuracy: ', acc)

        
        save_data(iteration,average_distance,"budget_baseline_dist.txt")
        save_data(iteration,F1,"budget_baseline_F1.txt")
        save_data(iteration,acc,"budget_baseline_acc.txt")
        return F1

    


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    criterion = nn.NLLLoss()


    train_dataset = F_budget(
        torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform),
        'FashionMNIST/budget/budget_baseline/train_budget_base_data.txt')
    test_dataset = F_budget(
        torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform),
        'FashionMNIST/budget/budget_baseline/test_budget_base_data.txt')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

    i = 1
    test_period = 500
    log_period = 50
    running_loss = 0.0
    log = Logger()

    for epoch in range(1):

        for data in trainloader:
            inputs, labels = data
            inputs,labels = inputs.to(device),labels.to(device)
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
                log.log('F1', i * 2, test_F_MNIST(i * 2))
            i += 1

    log.write_to_file('graphs/budget_baseline')