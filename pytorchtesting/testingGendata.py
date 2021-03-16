
import torchvision
import random
import torchvision.transforms as transforms
from testingpytorch import FashionDataset

train_set = FashionDataset(transform= transforms.Compose([transforms.ToTensor()]))

datasets = {'train': train_set, 'test': 0}

print(train_set[0][0])



def next_example(dataset, i):
    x, y= next(i), next(i)
    (_, c1), (_, c2)= dataset[x], dataset[y]
    return x, y, c1+c2


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
            print(example)
            args = tuple('{}'.format(e) for e in example[:-1])
            f.write('{} {} {}\n'.format(*args, example[-1]))

gather_examples('train', 'train_testje.txt')