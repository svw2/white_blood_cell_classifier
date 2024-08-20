import torch
import torch.nn as nn

class NWK(nn.Module):
    def __init__(self, values, isize):
        super().__init__()
        self.values = values
        self.isize = isize
        self.L1 = nn.Linear(isize, 128)
        self.L2 = nn.Linear(128, 64)
        self.L3 = nn.Linear(64, 32)
        self.L4 = nn.Linear(32, values)
        self.relu = nn.ReLU()
        self.LS= nn.LogSoftmax(dim = -1)
        

    def forward(self, x):
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        x = self.relu(x)
        x = self.L4(x)
        x = self.relu(x)
        x = self.LS(x)
        return x
    



