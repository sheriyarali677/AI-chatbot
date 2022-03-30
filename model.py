import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, size_input, size_hidden, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(size_input, size_hidden) 
        self.l2 = nn.Linear(size_hidden, size_hidden) 
        self.l3 = nn.Linear(size_hidden, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out