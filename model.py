import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *
from math import log2

class Net(nn.Module):
    def __init__(self, num_channels, num_features, scale_factor):
        super(Net, self).__init__()
        self.phase = int(log2(scale_factor))
    
        n_resgroups = 10
        n_resblocks = 20
        
        self.head = ConvBlock(num_channels, num_features, 3, 1, 1, activation=None, norm=None)
        
        necks = [
            ResidualGroup(num_features, kernel_size = 3, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        self.neck = nn.Sequential(*necks)
        
        self.body = UAB(features=num_features, phase=self.phase, activation=False, M=3)

        self.tail = ConvBlock(num_features, num_channels, 3, 1, 1, activation=None, norm=None)
        
            
    def forward(self, x):
        x = self.head(x)
        res = self.neck(x)
        x = x + res
        x = self.body(x)
        x = self.tail(x)
        
        return x
    