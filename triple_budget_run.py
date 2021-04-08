from DeepProbLog.train import train_model2
from DeepProbLog.data_loader import load
from DeepProbLog.model import Model
from DeepProbLog.optimizer import Optimizer
from DeepProbLog.network import Network
from neural_networks.defaultCNN import Fashion_MNIST_CNN, neural_predicate
import torch
from problog.logic import Var
import numpy as np
from metrics.metric_recording import save_data, clear_file, save_cm


pl_file_path = 'pl_files/triple_budget.pl'

clear_file("triple_budget_acc.txt")
clear_file("triple_budget_F1.txt")
clear_file("triple_budget_dist.txt")

nr_output = 1
num_classes = 27

labelMap = {30:0, 40:1, 45:2, 50:3, 55:4, 60:5, 65:6, 70:7, 75:8, 80:9, 85:10, 90:11, 95:12, 100:13, 105:14, 110:15, 115:16, 120:17, 125:18, 130:19, 135:20, 140:21, 145:22, 150:23, 160:24, 170:25, 180:26}

def test(model,iteration):
    
    total_distance = 0
    correct = 0
    N = len(test_queries)
    confusion = np.zeros((num_classes, num_classes), dtype=np.uint32)  # First index actual, second index predicted
    n = 0
    for d in test_queries:
        args = list(d.args)
        label = args[-1]
        args[-nr_output:] = [Var('X_{}'.format(i)) for i in range(nr_output)]
        q = d(*args)
        out = model.solve(q, None, True)
        out = max(out, key=lambda x: out[x][0])
        if out == d:
            correct += 1
        else :
            n += 1
        l = labelMap[label]
        c = labelMap[list(out.args)[-1]]
        confusion[l, c] += 1
        total_distance += abs(int(label)-int(list(out.args)[-1]))
    save_cm(confusion,"triple_budget_cm.txt")
    print(confusion)
    F1 = 0
    for nr in range(num_classes):
            TP = confusion[nr, nr]
            FP = sum(confusion[:, nr]) - TP
            FN = sum(confusion[nr, :]) - TP
            F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N

    acc = correct / N
    avg_distance = total_distance / n
    print("Acc : " + str(acc))

    save_data(iteration,avg_distance,"triple_budget_dist.txt")
    save_data(iteration,acc,"triple_budget_acc.txt")
    save_data(iteration,F1,"triple_budget_F1.txt")
    return 

train_queries = load('data/generated_data/train_triple_budget_data.txt')
test_queries = load('data/generated_data/test_triple_budget_data.txt')

with open(pl_file_path) as f:
    problog_string = f.read()
    
# Might need to make multiple nets and add them all to model
network = Fashion_MNIST_CNN()
net = Network(network,'fashion_mnist_net', neural_predicate)


net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string,[net],caching=False)
optimizer = Optimizer(model,2)

# logger = train_model(model,train_queries,1,optimizer, test_iter=1000,test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)
# logger.write_to_file('metrics/budget')

train_model2(model,train_queries,3,optimizer,test_iter=1000,test=test,snapshot_iter=10000)