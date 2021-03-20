from train import train_model2, train_model
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
from FashionMNIST.fashionCNN import Fashion_MNIST_CNN, neural_predicate, test_Fashion_MNIST
import torch
from problog.logic import Var
import numpy as np
from graphs.graphs import save_data, clear_file, save_cm


rel_path = 'FashionMNIST/budget/'

pl_file_path = rel_path + 'budget.pl'

clear_file("budget_acc.txt")
clear_file("budget_F1.txt")

nr_output = 1
num_classes = 17

labelMap = {20:0,30:1,35:2,40:3,45:4,50:5,55:6,60:7,65:8,70:9,75:10,80:11,85:12,90:13,100:14,110:15,120:16}

def test(model,iteration):
    
    n=0
    correct = 0
    N = len(test_queries)
    confusion = np.zeros((num_classes, num_classes), dtype=np.uint32)  # First index actual, second index predicted
    
    for d in test_queries:
        args = list(d.args)
        label = args[-1]
        args[-nr_output:] = [Var('X_{}'.format(i)) for i in range(nr_output)]
        q = d(*args)
        out = model.solve(q, None, True)
        out = max(out, key=lambda x: out[x][0])
        if out == d:
            correct += 1
        confusion[labelMap[list(out.args)[-1]], labelMap[label]] += 1
        n+=1

    save_cm(confusion,"budget_cm.txt")
    print(confusion)
    F1 = 0
    for nr in range(num_classes):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N

    acc = correct / N
    print("Acc : " + str(acc))
    save_data(iteration,acc,"budget_acc.txt")
    save_data(iteration,F1,"budget_F1.txt")
    return 

train_queries = load(rel_path + 'train_fashion_data.txt')
test_queries = load(rel_path + 'test_fashion_data.txt')

with open(pl_file_path) as f:
    problog_string = f.read()
    
# Might need to make multiple nets and add them all to model
network = Fashion_MNIST_CNN()
net = Network(network,'fashion_mnist_net', neural_predicate)


net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string,[net],caching=False)
optimizer = Optimizer(model,2)

# logger = train_model(model,train_queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)
# logger.write_to_file('graphs/budget')

train_model2(model,train_queries,1,optimizer,test_iter=1000,test=test,snapshot_iter=10000)