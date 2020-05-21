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
    
    
def next_batch(h_edges, start, finish):
    """
    Helper function for the iterator, note that the neural graph machines,
    due to its unique loss function, requires carefully crafted inputs
    Refer to the Neural Graph Machines paper, section 3 and 3.3 for more details
    """
    edges_ll = list()
    edges_lu = list()
    edges_uu = list()
    weights_ll = list()
    weights_lu = list()
    weights_uu = list()
    batch_edges = h_edges[start:finish]
    #batch_edges = np.asarray(batch_edges)
    
    #randomly assign labelled, unlabelled 
    
    edges_ll = batch_edges.sample(int(len(batch_edges)/3), replace=False)
    edges_ul = batch_edges.sample(int(len(batch_edges)/3), replace=False)
    edges_uu = batch_edges
    
    weights_ll = list(edges_ll[0])
    edges_ll = edges_ll[['level_0', 'level_1']]

    u_ll = [e[0] for e in edges_ll]

    # number of incident edges for nodes u
    c_ull = [1 / len(graph.edges(n)) for n in u_ll]
    v_ll = [e[1] for e in edges_ll]
    c_vll = [1 / len(graph.edges(n)) for n in v_ll]
    nodes_ll_u = X[u_ll]

    labels_ll_u = np.zeros((0,2))
    if len(nodes_ll_u) > 0:
        labels_ll_u = np.vstack([label(n) for n in u_ll])

    nodes_ll_v = X[v_ll]

    labels_ll_v = np.zeros((0,2))
    if len(nodes_ll_v) > 0:
        labels_ll_v = np.vstack([label(n) for n in v_ll])

    u_lu = [e[0] for e in edges_lu]
    c_ulu = [1 / len(graph.edges(n)) for n in u_lu]
    nodes_lu_u = X[u_lu]
    nodes_lu_v = X[[e[1] for e in edges_lu]]

    labels_lu = np.zeros((0,2))
    if len(nodes_lu_u) > 0:
        labels_lu = np.vstack([label(n) for n in u_lu])

    nodes_uu_u = X[[e[0] for e in edges_uu]]
    nodes_uu_v = X[[e[1] for e in edges_uu]]

    return nodes_ll_u, nodes_ll_v, labels_ll_u, labels_ll_v, \
           nodes_uu_u, nodes_uu_v, nodes_lu_u, nodes_lu_v, \
           labels_lu, weights_ll, weights_lu, weights_uu, \
           c_ull, c_vll, c_ulu
    

    
    
def batch_iter(batch_size, c_merged, X_df, y_df):
    edges = c_merged
    edges.values[[np.arange(len(edges))]*2] = np.nan
    edges = edges.stack().reset_index()
    
    num_batches = len(edges) / batch_size
    
    if data_size % batch_size > 0:
        num_batches = int(data_size / batch_size) + 1
        
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield next_batch(edges,start_index,end_index)

    
    

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


# Looking at edges -- labelled, unlabelled, etc. 
# Getting weight of the edge
# distance between nodes
# last hidden layer + criterion function for each node
