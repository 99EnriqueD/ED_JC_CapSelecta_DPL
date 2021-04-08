from train import train_model2, train_model
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
from FashionMNIST.advanced_budget.advanced_budget_net import Advanced_budget_net,test_advanced_budget,neural_predicate 
import torch
from problog.logic import Var
import numpy as np
from graphs.graphs import save_data, clear_file, save_cm


rel_path = 'FashionMNIST/advanced_budget/'

pl_file_path = rel_path + 'advanced_budget.pl'

clear_file("advanced_budget_acc.txt")
clear_file("advanced_budget_F1.txt")
clear_file("advanced_budget_dist.txt")

nr_output = 1
num_classes = 17

labelMap = {20:0,30:1,35:2,40:3,45:4,50:5,55:6,60:7,65:8,70:9,75:10,80:11,85:12,90:13,100:14,110:15,120:16}

def test(model,iteration):
    
    total_distance = 0
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
        l = labelMap[label]
        c = labelMap[list(out.args)[-1]]
        confusion[l, c] += 1
        total_distance += abs(l-c)
    save_cm(confusion,"advanced_budget_cm.txt")
    print(confusion)
    F1 = 0
    for nr in range(num_classes):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N

    acc = correct / N
    avg_distance = total_distance / N
    print("Acc : " + str(acc))

    save_data(iteration,avg_distance,"advanced_budget_dist.txt")
    save_data(iteration,acc,"advanced_budget_acc.txt")
    save_data(iteration,F1,"advanced_budget_F1.txt")
    return 

train_queries = load('FashionMNIST/advanced_budget/train_advanced_data.txt')
test_queries = load('FashionMNIST/advanced_budget/test_advanced_data.txt')

with open(pl_file_path) as f:
    problog_string = f.read()
    
# Might need to make multiple nets and add them all to model
network = Advanced_budget_net()
net = Network(network,'advanced_budget_net', neural_predicate)


net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string,[net],caching=False)
optimizer = Optimizer(model,2)

# logger = train_model(model,train_queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)
# logger.write_to_file('graphs/budget')

train_model2(model,train_queries,1,optimizer,test_iter=1000,test=test,snapshot_iter=10000)
