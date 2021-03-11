from train import train_model
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
from FashionMNIST.fashionCNN import Fashion_MNIST_CNN, neural_predicate, test_Fashion_MNIST
import torch

pl_file_path = 'appropriate_wardrobe.pl'


train_queries = load('train_wardrobe_data.txt')
test_queries = load('test_wardrobe_data.txt')

with open(pl_file_path) as f:
    problog_string = f.read()
    
# Might need to make multiple nets and add them all to model
network = Fashion_MNIST_CNN()
net = Network(network,'fashion_mnist_net', neural_predicate)


net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string,[net],caching=False)
optimizer = Optimizer(model,2)

train_model(model,train_queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)