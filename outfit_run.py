from train import train_model,train_model2
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
from problog.logic import Var
from FashionMNIST.outfit.net import Fashion_MNIST_CNN, neural_predicate, test_Fashion_MNIST
import torch
from graphs.graphs import save_data, clear_file, save_cm
import numpy as np

rel_path = "FashionMNIST/outfit/"

pl_file_path = rel_path + 'outfit.pl'

clear_file("outfit_acc.txt")
clear_file("outfit_F1.txt")

nr_output = 3
num_classes = 8

def labelVector(bits) :
    return 1*int(bits[2]) + 2*int(bits[1]) + 4*int(bits[0])

def test(model,iteration):
    
    n=0
    correct = 0
    N = len(test_queries)
    confusion = np.zeros((num_classes, num_classes), dtype=np.uint32)  # First index actual, second index predicted
    
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

    save_cm(confusion,"outfit_cm.txt")
    print(confusion)
    F1 = 0
    for nr in range(num_classes):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N

    acc = correct / N
    print("Acc : " + str(acc))
    save_data(iteration,acc,"outfit_acc.txt")
    save_data(iteration,F1,"outfit_F1.txt")
    return 

train_queries = load(rel_path + 'train_outfit_data.txt')
test_queries = load(rel_path + 'test_outfit_data.txt')

with open(pl_file_path) as f:
    problog_string = f.read()
    
# Might need to make multiple nets and add them all to model
network = Fashion_MNIST_CNN()
net = Network(network,'fashion_mnist_net', neural_predicate)


net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string,[net],caching=False)
optimizer = Optimizer(model,2)

# train_model(model,train_queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)
train_model2(model,train_queries,1,optimizer,test_iter=1000,test=test,snapshot_iter=10000)