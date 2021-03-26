
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
            return torch.cat((self.dataset[i1][0],self.dataset[i2][0]), 1), l


if __name__ == '__main__':

    # Wardrobe specific parameters:
    num_ftrs_wardrobe = 8
    clear_file("advanced_outfit_baseline_F1.txt")
    clear_file("advanced_outfit_baseline_acc.txt")

    ####
    
   # Load the dataset
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor(),
        # transforms.RandomResizedCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = Advanced_outfit(FashionTrainDataset(transform= transform),"FashionMNIST/advanced_outfit/advanced_outfit_baseline/train_advanced_outfit_base_data.txt")
    test_dataset = Advanced_outfit(FashionTestDataset(transform= transform),"FashionMNIST/advanced_outfit/advanced_outfit_baseline/test_advanced_outfit_base_data.txt" )
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    model_conv = torchvision.models.resnet18(pretrained=True)

    # Freeze feature extraction weights to speed up training (these parameters will not be changed during back propagation)
    for param in model_conv.parameters():
        param.requires_grad = False


    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs_resnet = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs_resnet, num_ftrs_wardrobe)
    model_conv.fc = nn.Sequential(
            nn.Linear(num_ftrs_resnet, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_ftrs_wardrobe),
            # nn.Softmax(1)
            )
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = model_conv.to(device)

    net.train()

    # Confusion matrix
    def test_DF(iteration):
        confusion = np.zeros((num_ftrs_wardrobe, num_ftrs_wardrobe), dtype=np.uint32)  # First index actual, second index predicted
        correct = 0
        n = 0
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
            n += 1
            # TODO add distance
        acc = correct / n
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
        save_data(iteration,F1,"advanced_outfit_baseline_F1.txt")
        save_data(iteration,acc,"advanced_outfit_baseline_acc.txt")
        return F1


   
    # train
    i = 1
    test_period = 500
    log_period = 50
    running_loss = 0.0
    # log = Logger()
    optimizer = optim.Adam(net.fc.parameters(), lr=0.01)
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
                net.train()
                test_DF(i*2)
                net.train()
                # exp_lr_scheduler.step()
            i += 1