# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# from torch.autograd import Variable

# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import confusion_matrix

import torchvision
import random
import torchvision.transforms as transforms


train_set = torchvision.datasets.FashionMNIST("../data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("../data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()])) 


datasets = {'train': train_set, 'test': test_set}

priceMapping = {0:10, 1:25, 2:20, 3:25, 4:50, 5:40, 6:20, 7:30, 8:60, 9:20}
labelMap = {20:0,30:1,35:2,40:3,45:4,50:5,55:6,60:7,65:8,70:9,75:10,80:11,85:12,90:13,100:14,110:15,120:16}
def next_example(dataset, i):
    x, y = next(i), next(i)
    (_, c1), (_, c2) = dataset[x], dataset[y]
    return x, y, labelMap[priceMapping[c1] + priceMapping[c2]]


def gather_examples(dataset_name, filename):
    dataset = datasets[dataset_name]
    examples = list()
    i = list(range(len(dataset)))
    random.shuffle(i)
    i = iter(i)
    while True:
        try:
            examples.append(next_example(dataset, i))
        except StopIteration:
            break

    with open(filename, 'w') as f:
        for example in examples:
            args = tuple('{}'.format(e) for e in example[:-1])
            f.write('{} {} {}\n'.format(*args, example[-1]))


# image, label = test_set[4104]

# plt.imshow(image.squeeze(), cmap="gray")
# print(label)

# image, label = test_set[1867]

# plt.imshow(image.squeeze(), cmap="gray")
# print(label)

gather_examples('train', '../data/generated_data/train_budget_base_data.txt')
gather_examples('test', '../data/generated_data/test_budget_base_data.txt')