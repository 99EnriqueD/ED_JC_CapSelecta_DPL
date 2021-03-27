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


train_set = torchvision.datasets.FashionMNIST("data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()])) 


datasets = {'train': train_set, 'test': test_set}

priceMapping = {0:10, 1:25, 2:20, 3:25, 4:50, 5:40, 6:20, 7:30, 8:60, 9:20}
labelMap = {30: 0, 40: 1, 45: 2, 50: 3, 55: 4, 60: 5, 65: 6, 70: 7, 75: 8, 80: 9, 85: 10, 90: 11, 95: 12, 100: 13, 105: 14, 110: 15, 115: 16, 120: 17, 125: 18, 130: 19, 135: 20, 140: 21, 145: 22, 150: 23, 160: 24, 170: 25, 180: 26}

def next_example(dataset, i):
    x, y, z = next(i), next(i), next(i)
    (_, c1), (_, c2), (_,c3) = dataset[x], dataset[y], dataset[z]
    return x, y, z, labelMap[priceMapping[c1] + priceMapping[c2] + priceMapping[c3]]


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
            f.write('{} {} {} {}\n'.format(*args, example[-1]))


# image, label = test_set[4104]

# plt.imshow(image.squeeze(), cmap="gray")
# print(label)

# image, label = test_set[1867]

# plt.imshow(image.squeeze(), cmap="gray")
# print(label)

gather_examples('train', 'FashionMNIST/triple_budget/triple_budget_base/train_triple_budget_base_data.txt')
gather_examples('test', 'FashionMNIST/triple_budget/triple_budget_base/test_triple_budget_base_data.txt')