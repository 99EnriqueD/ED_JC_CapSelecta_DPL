
import torchvision
import random
import torchvision.transforms as transforms
from advanced_outfit_baseline.FashionDatasetClass import FashionTrainDataset,FashionTestDataset

train_set = FashionTrainDataset(transform= transforms.Compose([transforms.ToTensor()]))
test_set = FashionTestDataset(transform= transforms.Compose([transforms.ToTensor()]))

datasets = {'train': train_set, 'test': test_set}

mapping = {1:6,2:6,3:0,4:6,5:0,6:2,7:0,8:0,9:0,10:2,11:2,12:1,13:6,14:6,15:1,16:1,17:0,18:0,19:0,20:1,
21:4,22:4,23:4,24:2,25:3,26:3,27:3,28:3,29:3,30:3,31:5,32:2,
33:5,34:3,35:2,36:2,37:5,38:6,39:6,40:2,41:5,42:5,43:5,44:5,45:5,
46:5,47:5,48:5,49:5,50:5}

good_rain= {1,6}
tops= {0,1,6}
bottoms = {2,3,4,5}

def isGoodForRain(pieces) :
    for piece in pieces:
        if piece in good_rain:
            return 1
    return 0

def isFormal(pieces) :
    for piece in pieces :
        if piece == 3 or piece == 1: # casual trousers/trui
            return 0
    return 1

def hasFullOutfit(pieces) :
    if hasPieceInSet(pieces,tops) and hasPieceInSet(pieces,bottoms) :
        return 1
    return 0

def hasPieceInSet(pieces, setToCheck) :
    for piece in pieces :
        if piece in setToCheck:
            return True
    return False

# def labelVector(bits) :
#     # This will 
#     #return 1* bits[3] + 2*bits[2] + 4*bits[1] + 8*bits[0]
#     return 4*bits[0] + 2*bits[1] + 1*bits[2]

def next_example(dataset, i):
    x,y = next(i), next(i)
    (_, c1), (_, c2)= dataset[x], dataset[y]
    pieces =[mapping.get(c1),mapping.get(c2)]
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
            args = tuple('{}({})'.format(dataset_name,e) for e in example[:-3])
            f.write('appropriateWardrobe({},{},{},{},{}).\n'.format(*args, example[-3],example[-2],example[-1]))

gather_examples('train', 'FashionMNIST/advanced_outfit/train_advanced_outfit_data.txt')
gather_examples('test', 'FashionMNIST/advanced_outfit/test_advanced_outfit_data.txt')