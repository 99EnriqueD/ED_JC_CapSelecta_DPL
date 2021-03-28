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

from problog.logic import Var
from FashionMNIST.advanced_outfit.dpl_utils import neural_predicate
from FashionMNIST.advanced_outfit.advanced_outfit_baseline.FashionDatasetClass import FashionTrainDataset,FashionTestDataset
from train import train_model, train_model2
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
from graphs.graphs import save_data, clear_file, save_cm

#### NET

# Wardrobe specific parameters:
num_ftrs_wardrobe = 7
nr_output = 3
num_outputs_cm = 8

net = torchvision.models.resnet50(pretrained=True)

# Freeze feature extraction weights to speed up training (these parameters will not be changed during back propagation)
for param in net.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs_resnet = net.fc.in_features
# net.fc = nn.Linear(num_ftrs_resnet,num_ftrs_wardrobe)
net.fc = nn.Sequential(
            nn.Linear(num_ftrs_resnet, num_ftrs_wardrobe),
            # nn.ReLU(),
            # nn.Linear(120, 84),
            # nn.ReLU(),
            # nn.Linear(84, num_ftrs_wardrobe),
            nn.Softmax(1)
            )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)


clear_file("advanced_outfit_dist.txt")
clear_file("advanced_outfit_acc.txt")
clear_file("advanced_outfit_F1.txt")

def labelVector(bits) :
    return 1*int(bits[2]) + 2*int(bits[1]) + 4*int(bits[0])


labelMap={0:[0,0,0], 1:[0,0,1],2:[0,1,0],3:[0,1,1],4:[1,0,0],5:[1,0,1],6:[1,1,0],7:[1,1,1]}

def hamming_dist(l, c) :
    bl = labelMap[l]
    bc = labelMap[c]
    distance = 0
    for index in range(len(bl)):
        distance += abs(bc[index] - bl[index])
    return distance

def test(model,iteration):
    net.eval()
    for param in net.fc.parameters() :
        param.requires_grad = False
    n=0
    correct = 0
    N = len(test_queries)
    confusion = np.zeros((num_outputs_cm, num_outputs_cm), dtype=np.uint32)  # First index actual, second index predicted
    total_distance=0
    for d in test_queries:
        args = list(d.args)
        label = args[-nr_output:]
        args[-nr_output:] = [Var('X_{}'.format(i)) for i in range(nr_output)]
        q = d(*args)
        out = model.solve(q, None, True)
        out = max(out, key=lambda x: out[x][0])
        if out == d:
            correct += 1
        else:
            n += 1
        l = labelVector(label)
        c = labelVector(list(out.args)[-nr_output:])
        confusion[l, c] += 1
        total_distance += hamming_dist(l, c)

    save_cm(confusion,"advanced_outfit_cm.txt")
    print(confusion)
    F1 = 0
    for nr in range(num_outputs_cm):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N

    acc = correct / N
    print("Acc : " + str(acc))
    avg_distance = total_distance / n
    save_data(iteration,avg_distance,"advanced_outfit_dist.txt")
    save_data(iteration,acc,"advanced_outfit_acc.txt")
    save_data(iteration,F1,"advanced_outfit_F1.txt")
    
    for param in net.fc.parameters() :
        param.requires_grad = True
    net.train()
    return 


#### DPL

rel_path = "FashionMNIST/advanced_outfit/"

pl_file_path = rel_path + 'advanced_outfit.pl'

train_queries = load(rel_path + 'train_advanced_outfit_data.txt')
test_queries = load(rel_path + 'test_advanced_outfit_data.txt')

with open(pl_file_path) as f:
    problog_string = f.read()

net.optimizer = torch.optim.Adam(net.fc.parameters(), lr=0.001)   
# Might need to make multiple nets and add them all to model
netwrk = Network(net,'fashion_df_net', neural_predicate)

model = Model(problog_string,[netwrk],caching=False)
optimizer = Optimizer(model,2)

# train_model(model,train_queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)
train_model2(model,train_queries,3,optimizer,test_iter=5000,test=test,snapshot_iter=10000)
