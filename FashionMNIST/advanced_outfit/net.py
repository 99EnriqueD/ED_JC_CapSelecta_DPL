import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Fashion_DF_CNN(nn.Module):
    def __init__(self, N=8):
        super(Fashion_DF_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  6, 5),
            nn.MaxPool2d(2, 2), # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2), # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier =  nn.Sequential(
            nn.Linear(115168, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            # nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 115168)
        x = self.classifier(x)
        return x