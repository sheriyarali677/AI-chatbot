import torch
import torch.nn as nn


class Modelnet(nn.Module):
    def __init__(self, size_input, size_hidden, num_classes):
        super(Modelnet, self).__init__()
        self.linear1 = nn.Linear(size_input, size_hidden) 
        self.linear2 = nn.Linear(size_hidden, size_hidden) 
        self.linear3 = nn.Linear(size_hidden, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)

        return out