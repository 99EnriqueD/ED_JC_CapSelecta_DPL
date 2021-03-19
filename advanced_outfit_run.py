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

from FashionMNIST.advanced_outfit.dpl_utils import neural_predicate
from FashionMNIST.advanced_outfit.advanced_outfit_baseline.FashionDatasetClass import FashionTrainDataset,FashionTestDataset
from train import train_model
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network


#### NET

# Wardrobe specific parameters:
num_ftrs_wardrobe = 7

model_conv = torchvision.models.resnet18(pretrained=True)

# Freeze feature extraction weights to speed up training (these parameters will not be changed during back propagation)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs_resnet = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs_resnet,num_ftrs_wardrobe)
model_conv.fc = nn.Sequential(
    nn.Linear(num_ftrs_resnet, num_ftrs_wardrobe),
    nn.Softmax(1)
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_conv = model_conv.to(device)
network = model_conv.eval()


#### DPL

rel_path = "FashionMNIST/advanced_outfit/"

pl_file_path = rel_path + 'advanced_outfit.pl'


train_queries = load(rel_path + 'train_advanced_outfit_data.txt')
test_queries = load(rel_path + 'test_advanced_outfit_data.txt')

with open(pl_file_path) as f:
    problog_string = f.read()
    
# Might need to make multiple nets and add them all to model
net = Network(network,'fashion_df_net', neural_predicate)

net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string,[net],caching=False)
optimizer = Optimizer(model,2)

train_model(model,train_queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)
