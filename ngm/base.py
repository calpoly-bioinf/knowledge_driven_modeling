#!/usr/bin/env python
# coding: utf-8

# Initialize Pytorch

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import pandas as pd
import numpy as np

import copy

class Net1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net1 , self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 10312 (input data) -> 50 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 50 (hidden node) -> 39 (output class)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def train(model,dataloader,**kwargs):
    # model is such as Net1
    
    default_kwargs = {
        'num_epochs': 5,        # The number of times entire dataset is trained
        'batch_size': 10,       # The size of input data took for one iteration
        'learning_rate':0.001,  # The speed of convergence
    }
    
    options = {**default_kwargs,**kwargs}

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'])

    for epoch in range(options['num_epochs']):
        for i, (images, labels) in enumerate(dataloader):     # Load a batch of images with its (index, data, class)
            images = Variable(images.view(-1, 1*10312))       # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
            labels = Variable(labels)

            optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
            outputs = model(images.float())                   # Forward pass: compute the output class given a image
            #if epoch == num_epochs-1:
            #    print(outputs, labels.long())
            loss = criterion(outputs, labels.long())          # Compute the loss: difference between the output class and the pre-given label

            loss.backward()                                   # Backward pass: compute the weight
            optimizer.step()                                  # Optimizer: update the weights of hidden nodes

            if (i+1) % 100 == 0:                              # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.item()))