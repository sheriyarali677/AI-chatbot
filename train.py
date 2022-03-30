# Importing Necessary Modules
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

#Import necessary inputs from natural language toolkit
from model import NeuralNet
from nltk_utils import bag_of_words
from nltk_utils import tokenize
from nltk_utils import stem

with open('dataintents.json', 'r') as f:
    dataints = json.load(f)

tags = []
word_bag = []
xy = []

for int in dataints ['dataintents']:
    # add to tag list
    tags.append(int['tag'])
    for pat in int['patterns']:
        # tokenize each word in the sentence
        n = tokenize(pat)
        # add to our words list
        word_bag.extend(n)
        # add to xy pair
        xy.append((n, int['tag']))

# stem and lower each word
word_bag = [stem(n) for n in word_bag if n not in ['?', '.', '!']]

# remove duplicates and sort
tags = sorted(set(tags))
word_bag = sorted(set(word_bag))

print(len(tags), "tags:", tags)
print(len(word_bag), "unique stemmed words:", word_bag)
print(len(xy), "patterns")

# create training data
X_train_data = []
y_train_data = []
for (pat_sent, tag) in xy:

    bag = bag_of_words(pat_sent, word_bag)
    label = tags.index(tag)
    
    X_train_data.append(bag)
    y_train_data.append(label)

y_train_data = np.array(y_train_data)
X_train_data = np.array(X_train_data)

# parameters 
size_batch = 8
lr = 0.001
epoch_count = 1000
size_hidden = 8
size_input = len(X_train_data[0])
size_output = len(tags)
print(size_input, size_output)

class ChatDataset(Dataset):

    def __init__(self):
        self.sample_count = len(X_train_data)
        self.x = X_train_data
        self.y = y_train_data

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.sample_count

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=size_batch, num_workers=0, shuffle=True)

device = torch.device('cpu')
model = NeuralNet(size_input, size_hidden, size_output).to(device)

# Loss and optimiser
opt = torch.optim.Adam(model.parameters(), lr=lr)
crit = nn.CrossEntropyLoss()

# Train the model
for epoch in range(epoch_count):
    for (words, lbls) in train_loader:
        
        lbls = lbls.to(dtype=torch.long).to(device)
        words = words.to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # lbls = torch.max(lbls, 1)[1]
        loss = crit(outputs, lbls)
        
        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{epoch_count}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"size_output": size_output,
"model_state": model.state_dict(),
"size_hidden": size_hidden,
"tags": tags,
"size_input": size_input,
"word_bag": word_bag
}

filename = "data_output.pth"
torch.save(data, filename)

print(f'Training finished and file saved to {filename}')
