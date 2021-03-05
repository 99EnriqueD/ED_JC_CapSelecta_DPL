from train import train_model
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
import torch

pl_file_path = 'deepStylist.pl'


train_queries = load('')
test_queries = load('')

with open('deepStylist.pl') as f:
    problog_string = f.read()
    
# Might need to make multiple nets and add them all to model
network = Fabric_Net()
net = Network(network,'DF_Net', neural_predicate)


net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string,[net],caching=False)
optimizer = Optimizer(model,2)

train_model(model,queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)