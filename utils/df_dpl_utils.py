from torch.autograd import Variable
import torch
from torchvision import transforms
import numpy as np
from utils.FashionDatasetClass import FashionTrainDataset,FashionTestDataset


def test_Fashion_MNIST(model,max_digit=7,name='fashion_df_net'):
    confusion = np.zeros((max_digit,max_digit),dtype=np.uint32) # First index actual, second index predicted
    N = 0
    for d,l in df_test_data:
        if l < max_digit:
            N += 1
            d = Variable(d.unsqueeze(0))
            outputs = model.networks[name].net.forward(d)
            _, out = torch.max(outputs.data, 1)
            c = int(out.squeeze())
            confusion[l,c] += 1
    print(confusion)
    F1 = 0
    for nr in range(max_digit):
        TP = confusion[nr,nr]
        FP = sum(confusion[:,nr])-TP
        FN = sum(confusion[nr,:])-TP
        F1 += 2*TP/(2*TP+FP+FN)*(FN+TP)/N
    print('F1: ',F1)
    return [('F1',F1)]


def neural_predicate(network, i):
    dataset = str(i.functor)
    i = int(i.args[0])
    if dataset == 'train':
        d, l = df_train_data[i]
    elif dataset == 'test':
        d, l = df_test_data[i]
    d = Variable(d.unsqueeze(0))
    output = network.net(d)
    return output.squeeze(0)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(300),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
df_train_data = FashionTrainDataset(transform=transform)
df_test_data = FashionTestDataset(transform=transform)