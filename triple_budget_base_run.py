import matplotlib.pyplot as plt
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


from FashionMNIST.triple_budget.triple_budget_base.net import TB_Net

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
            i1, i2, i3, l = self.data[index]
            # print("SHAPE1 : ", self.dataset[i1][0].shape)
            # print("SHAPE2 : ", self.dataset[i2][0].shape)
            # print("SHAPE3 : ", self.dataset[i1][0].shape)
            return torch.cat((self.dataset[i1][0], self.dataset[i2][0], self.dataset[i3][0]), 1), l

if __name__ == '__main__':

    clear_file("triple_budget_baseline_F1.txt")
    clear_file("triple_budget_baseline_acc.txt")
    clear_file("triple_budget_baseline_dist.txt")

    num_classes = 27

    labelMap ={0: 30, 1: 40, 2: 45, 3: 50, 4: 55, 5: 60, 6: 65, 7: 70, 8: 75, 9: 80, 10: 85, 11: 90, 12: 95, 13: 100, 14: 105, 15: 110, 16: 115, 17: 120, 18: 125, 19: 130, 20: 135, 21: 140, 22: 145, 23: 150, 24: 160, 25: 170, 26: 180}
    
    def test_F_MNIST(iteration):
        confusion = np.zeros((num_classes, num_classes), dtype=np.uint32)  # First index actual, second index predicted
        correct = 0
        N = len(test_dataset)
        total_distance = 0
        n = 0
        for d, l in test_dataset:
            d = d.to(device)
            d = Variable(d.unsqueeze(0))
            outputs = net.forward(d)
            _, out = torch.max(outputs.data, 1)
            c = int(out.squeeze())
            confusion[l, c] += 1
            if c == l:
                correct += 1
            else :
                n += 1
            total_distance += abs(labelMap[c] - labelMap[l])
        acc = correct / N
        print(confusion)
        save_cm(confusion,"triple_budget_baseline_cm.txt")
        
        average_distance = total_distance / n

        F1 = 0
        for nr in range(num_classes):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
        print('F1: ', F1)
        print('Accuracy: ', acc)

        
        save_data(iteration,average_distance,"triple_budget_baseline_dist.txt")
        save_data(iteration,F1,"triple_budget_baseline_F1.txt")
        save_data(iteration,acc,"triple_budget_baseline_acc.txt")
        return F1

    


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = TB_Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    criterion = nn.NLLLoss()


    train_dataset = F_budget(
        torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform),
        'FashionMNIST/triple_budget/triple_budget_base/train_triple_budget_base_data.txt')
    test_dataset = F_budget(
        torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform),
        'FashionMNIST/triple_budget/triple_budget_base/test_triple_budget_base_data.txt')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)



    i = 1
    test_period = 500
    log_period = 50
    running_loss = 0.0
    log = Logger()

    b = True
    for epoch in range(3):

        for data in trainloader:
        #     if b:
        #         a = np.squeeze(data[0][0].permute(1, 2, 0))
        #         plt.imshow(a, cmap='gray')
        #         plt.show()
        #         b = False
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
                net.eval()
                log.log('F1', i * 2, test_F_MNIST(i * 2))
                net.train()
            i += 1