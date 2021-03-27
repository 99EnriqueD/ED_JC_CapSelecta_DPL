import torch
from torch import from_numpy
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from logger import Logger
import numpy as np
from FashionMNIST.advanced_outfit.advanced_outfit_baseline.FashionDatasetClass import FashionTrainDataset,FashionTestDataset
from graphs.graphs import save_data, clear_file, save_cm
from torch.optim import lr_scheduler
from FashionMNIST.advanced_outfit.net import Fashion_DF_CNN  

class Advanced_outfit(Dataset):
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
            # return self.dataset[i1],
            # print("IN GET ITEM")
            # print("SHAPE 1 : ", len(self.dataset[i1]))
            # print("SHAPE 1[0]", self.dataset[i1][0].size(), " -- ", self.dataset[i1][0])
            # print("SHAPE 1[1]", self.dataset[i1][1])
            # print("SHAPE 2 : ", len(self.dataset[i2]))
            # print("SHAPE 2[0]", self.dataset[i2][0].size(), " -- ", self.dataset[i2][0])
            # print("SHAPE 2[1]", self.dataset[i2][1])
            # print("LABEL : ", l)

            return torch.cat((self.dataset[i1][0],self.dataset[i2][0]), 1), l


if __name__ == '__main__':

    # Wardrobe specific parameters:
    num_ftrs_wardrobe = 8
    clear_file("advanced_outfit_baseline_F1.txt")
    clear_file("advanced_outfit_baseline_acc.txt")
    clear_file("advanced_outfit_baseline_dist.txt")

    ####
    
   # Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((250,250)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = Advanced_outfit(FashionTrainDataset(transform= transform),"FashionMNIST/advanced_outfit/advanced_outfit_baseline/train_advanced_outfit_base_data.txt")
    test_dataset = Advanced_outfit(FashionTestDataset(transform= transform),"FashionMNIST/advanced_outfit/advanced_outfit_baseline/test_advanced_outfit_base_data.txt" )
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    net = torchvision.models.resnet50(pretrained=True)

    # Freeze feature extraction weights to speed up training (these parameters will not be changed during back propagation)
    for param in net.parameters():
        param.requires_grad = False


    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs_resnet = net.fc.in_features
    net.fc = nn.Linear(num_ftrs_resnet, num_ftrs_wardrobe)
    
    # net = Fashion_DF_CNN()
    # net.fc = nn.Sequential(
    #         nn.Linear(num_ftrs_resnet, 120),
    #         nn.ReLU(),
    #         nn.Linear(120, 84),
    #         nn.ReLU(),
    #         nn.Linear(84, num_ftrs_wardrobe),
    #         # nn.Softmax(1)
    #         )
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    net.eval()
    labelMap={0:[0,0,0], 1:[0,0,1],2:[0,1,0],3:[0,1,1],4:[1,0,0],5:[1,0,1],6:[1,1,0],7:[1,1,1]}

    def hamming_dist(l, c) :
        bl = labelMap[l]
        bc = labelMap[c]
        distance = 0
        for index in range(len(bl)):
            distance += abs(bc[index] - bl[index])
        return distance

    # Confusion matrix
    def test_DF(iteration):
        confusion = np.zeros((num_ftrs_wardrobe, num_ftrs_wardrobe), dtype=np.uint32)  # First index actual, second index predicted
        correct = 0
        n = 0
        total_distance = 0
        N = len(test_dataset)
        for d, l in test_dataset:
            d = d.to(device)
            d = Variable(d.unsqueeze(0))
            outputs = net.forward(d)
            _, out = torch.max(outputs.data, 1)
            c = int(out.squeeze())
            confusion[l, c] += 1
            if c == l:
                correct += 1
            else:
                n += 1
            total_distance += hamming_dist(l, c)

        acc = correct / N
        average_distance = total_distance/ n 
        print(confusion)
        save_cm(confusion,"advanced_outfit_baseline_cm.txt")
        F1 = 0
        for nr in range(num_ftrs_wardrobe):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
        print('F1: ', F1)
        print('Accuracy: ', acc)
        save_data(iteration,average_distance,"advanced_outfit_baseline_dist.txt")
        save_data(iteration,F1,"advanced_outfit_baseline_F1.txt")
        save_data(iteration,acc,"advanced_outfit_baseline_acc.txt")
        return F1


   
    # train
    i = 1
    test_period = 500
    log_period = 50
    running_loss = 0.0
    # log = Logger()
    optimizer = optim.Adam(net.fc.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(3):

        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)     
            optimizer.zero_grad()
            
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            if i % log_period == 0:
                print('Iteration: ', i * 2, '\tAverage Loss: ', running_loss / log_period)
                # log.log('loss', i * 2, running_loss / log_period)
                running_loss = 0
            if i % test_period == 0:
                # log.log('F1', i * 2, test_DF(i*2))
                # net.eval()
                test_DF(i*2)
                # net.train()
                # exp_lr_scheduler.step()
            i += 1
