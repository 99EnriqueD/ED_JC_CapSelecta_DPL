import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from logger import Logger
import numpy as np
from metrics.metric_recording import save_data, clear_file, save_cm


from neural_networks.net_outfit_baseline import Net

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

    num_classes = 8

    clear_file("outfit_baseline_F1.txt")
    clear_file("outfit_baseline_acc.txt")
    clear_file("outfit_baseline_dist.txt")

    labelMap={0:[0,0,0], 1:[0,0,1],2:[0,1,0],3:[0,1,1],4:[1,0,0],5:[1,0,1],6:[1,1,0],7:[1,1,1]}

    def hamming_dist(l, c) :
        bl = labelMap[l]
        bc = labelMap[c]
        distance = 0
        for index in range(len(bl)):
            distance += abs(bc[index] - bl[index])
        return distance

    def test_F_MNIST(iteration):
        confusion = np.zeros((num_classes, num_classes), dtype=np.uint32)  # First index actual, second index predicted
        correct = 0
        total_distance = 0
        n=0
        N = len(test_dataset)
        for d, l in test_dataset:
            d = Variable(d.unsqueeze(0))
            outputs = net.forward(d)
            _, out = torch.max(outputs.data, 1)
            c = int(out.squeeze())
            confusion[l, c] += 1
            if c == l:
                correct += 1
            else :
                n+=1
            total_distance += hamming_dist(l,c)
        acc = correct / N
        print(confusion)
        save_cm(confusion,"outfit_baseline_cm.txt")

        avg_distance = total_distance / n

        F1 = 0
        for nr in range(num_classes):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
        print('F1: ', F1)
        print('Accuracy: ', acc)
        
        save_data(iteration,avg_distance,"outfit_baseline_dist.txt")
        save_data(iteration,F1,"outfit_baseline_F1.txt")
        save_data(iteration,acc,"outfit_baseline_acc.txt")
        return F1


    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    criterion = nn.NLLLoss()


    train_dataset = F_budget(
        torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform),
        'FashionMNIST/outfit/outfit_baseline/train_outfit_base_data.txt')
    test_dataset = F_budget(
        torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform),
        'FashionMNIST/outfit/outfit_baseline/test_outfit_base_data.txt')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

    i = 1
    test_period = 500
    log_period = 50
    running_loss = 0.0
    log = Logger()

    for epoch in range(3):

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
                log.log('F1', i * 2, test_F_MNIST(i*2))
            i += 1

    log.write_to_file('metrics/outfit_baseline')