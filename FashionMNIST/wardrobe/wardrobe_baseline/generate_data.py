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


train_set = torchvision.datasets.FashionMNIST("../../data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("../../data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()])) 

datasets = {'train': train_set, 'test': test_set}




piecesGoodForRain = {2,4}
piecesGoodForWarm = {0,5}

def isGoodForWarm(pieces) :
    piecesSet = set(pieces)
    if piecesSet >= piecesGoodForWarm :
        return 1
    else :
        return 0 

def isGoodForRain(pieces) :
    for piece in pieces:
        if piece in piecesGoodForRain :
            return 1
    return 0

def isFormal(pieces) :
    for piece in pieces :
        if piece in piecesGoodForWarm :
            return 0
    return 1

shoes = {5,7,8}
tops = {0,2,6}
bottoms = {1}
dress = {3}
bag = {8}
coat = {4}

def hasFullOutfit(pieces) :
    hasShoes = hasPieceInSet(pieces,shoes)
    if hasShoes and hasPieceInSet(pieces,tops) and hasPieceInSet(pieces,bottoms) :
        return 1
    if hasShoes and hasPieceInSet(pieces,dress):
        if hasPieceInSet(pieces,bag) or hasPieceInSet(pieces,coat) :
            return 1
    return 0


def hasPieceInSet(pieces, setToCheck) :
    for piece in pieces :
        if piece in setToCheck:
            return True
    return False


def label(bits) :
    return 1* bits[3] + 2*bits[2] + 4*bits[1] + 8*bits[0]

def next_example(dataset, i):
    x, y, z = next(i), next(i), next(i)
    (_, c1), (_, c2), (_, c3) = dataset[x], dataset[y], dataset[z]
    pieces = [c1,c2,c3]
    bits =  [isGoodForRain(pieces), isFormal(pieces), isGoodForWarm(pieces), hasFullOutfit(pieces)]
    return x, y, z, label(bits)


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
            f.write('{},{},{},{}\n'.format(*args, example[-1]))
            # appropriateWardrobe(I1, I2, I3, Rain, Formal, Warm, Full)

image, label = test_set[367]
print(label)

image, label = test_set[6186]
print(label)

image, label = test_set[4099]
print(label)

gather_examples('train', 'train_wardrobe_base_data.txt')
gather_examples('test', 'test_wardrobe_base_data.txt')