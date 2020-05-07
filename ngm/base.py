#!/usr/bin/env python
# coding: utf-8

# Initialize Pytorch

from torch.utils.data import Dataset
import torch.nn as nn
import torch

class MyDataset(Dataset):
    def __init__(self, values, labels):
        super(MyDataset, self).__init__()
        self.values = values
        self.labels = labels
    def __len__(self):
        return len(self.values)  # number of samples in the dataset
    def __getitem__(self, index):
        return self.values[index], self.labels[index]

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 10312 (input data) -> 50 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 50 (hidden node) -> 39 (output class)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

def train(model,A,dataloader,dataset,learning_rate,num_epochs):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    for epoch in range(num_epochs):
        for i, (X, labels) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)
            batch_size = len(labels)
            optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
            outputs = model(X)                   # Forward pass: compute the output class given a image
            try:
                loss = criterion(outputs, labels.long())          # Compute the loss: difference between the output class and the pre-given label
            except:
                import pdb; pdb.set_trace()

            loss.backward()                                   # Backward pass: compute the weight
            optimizer.step()                                  # Optimizer: update the weights of hidden nodes

            if (i+1) % 20 == 0:                              # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.item()))

def prepare_for_evaluation(model,testloader):
    all_predicted = []
    all_labels = []
    for i, (X, labels) in enumerate(testloader):
        outputs = model(X) 
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        all_predicted.extend([int(v) for v in predicted])
        all_labels.extend([int(v) for v in labels])
    return all_predicted,all_labels