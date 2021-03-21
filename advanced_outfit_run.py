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

model_conv = torchvision.models.resnet18(pretrained=True)

# Freeze feature extraction weights to speed up training (these parameters will not be changed during back propagation)
# for param in model_conv.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs_resnet = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs_resnet,num_ftrs_wardrobe)
model_conv.fc = nn.Sequential(
    nn.Linear(num_ftrs_resnet, num_ftrs_wardrobe),
    nn.Softmax(1)
)
model_conv.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
netwrk = model_conv.to(device)


clear_file("advanced_outfit_acc.txt")
clear_file("advanced_outfit_F1.txt")

def labelVector(bits) :
    return 1*int(bits[2]) + 2*int(bits[1]) + 4*int(bits[0])

def test(model,iteration):
    n=0
    correct = 0
    N = len(test_queries)
    confusion = np.zeros((num_ftrs_wardrobe, num_ftrs_wardrobe), dtype=np.uint32)  # First index actual, second index predicted
    
    for d in test_queries:
        args = list(d.args)
        label = args[-nr_output:]
        args[-nr_output:] = [Var('X_{}'.format(i)) for i in range(nr_output)]
        q = d(*args)
        out = model.solve(q, None, True)
        out = max(out, key=lambda x: out[x][0])
        if out == d:
            correct += 1
        confusion[labelVector(label), labelVector(list(out.args)[-nr_output:])] += 1
        n+=1

    save_cm(confusion,"advanced_outfit_cm.txt")
    print(confusion)
    F1 = 0
    for nr in range(num_ftrs_wardrobe):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N

    acc = correct / N
    print("Acc : " + str(acc))
    save_data(iteration,acc,"advanced_outfit_acc.txt")
    save_data(iteration,F1,"advanced_outfit_F1.txt")
    return 


#### DPL

rel_path = "FashionMNIST/advanced_outfit/"

pl_file_path = rel_path + 'advanced_outfit.pl'


train_queries = load(rel_path + 'train_advanced_outfit_data.txt')
test_queries = load(rel_path + 'test_advanced_outfit_data.txt')

with open(pl_file_path) as f:
    problog_string = f.read()
    print(problog_string)
    
# Might need to make multiple nets and add them all to model
net = Network(netwrk,'fashion_df_net', neural_predicate)

net.optimizer = torch.optim.Adam(netwrk.parameters(), lr=0.001)
model = Model(problog_string,[net],caching=False)
optimizer = Optimizer(model,2)

# train_model(model,train_queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)
train_model2(model,train_queries,1,optimizer,test_iter=1000,test=test,snapshot_iter=10000)