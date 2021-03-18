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


piecesGoodForRain = {2,4}

def isGoodForRain(pieces) :
    for piece in pieces:
        if piece in piecesGoodForRain :
            return 1
    return 0

def isFormal(pieces) :
    for piece in pieces :
        if piece == 0 or piece == 3 : 
            return 0
    return 1

tops = {0,2,4,5}
bottoms = {1,3}

def hasFullOutfit(pieces) :
    if hasPieceInSet(pieces,tops) and hasPieceInSet(pieces,bottoms) :
        return 1
    return 0


def hasPieceInSet(pieces, setToCheck) :
    for piece in pieces :
        if piece in setToCheck:
            return True
    return False

# We are not working with shoes in this application so we take them out and map the correct ones to a new order
old_to_new = {0:0,1:1,2:2,3:3,4:4,5:10,6:5,7:10,8:6,9:10}

def next_example(dataset, i):
    x = next(i)
    (_,c1) = dataset[x]
    while old_to_new[c1] == 10 :
        x = next(i)
        (_,c1) = dataset[x]
    y = next(i)
    (_,c2) = dataset[y]
    while old_to_new[c2] == 10 :
        y = next(i)
        (_,c2) = dataset[y]
    
    pieces = [old_to_new[c1],old_to_new[c2]]
    return x, y, isGoodForRain(pieces), isFormal(pieces), hasFullOutfit(pieces)


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
            args = tuple('{}({})'.format(dataset_name, e) for e in example[:-3])
            f.write('appropriateWardrobe({},{},{},{},{}).\n'.format(*args, example[-3], example[-2], example[-1],))
            # appropriateWardrobe(I1, I2, I3, Rain, Formal, Warm, Full)

# image, label = test_set[1886]
# print(label)

# image, label = test_set[9949]
# print(label)

# image, label = test_set[5381]
# print(label)

gather_examples('train', 'FashionMNIST/outfit/train_outfit_data.txt')
gather_examples('test', 'FashionMNIST/outfit/test_outfit_data.txt')