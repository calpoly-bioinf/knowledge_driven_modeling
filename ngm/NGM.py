from torch.utils.data import Dataset
import tensorflow as tf
import torch.nn as nn
import torch
import numpy as np

def next_batch(training_edges, start, finish, edges, X, y):
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
    batch_edges = training_edges[start:finish]
    batch_edges = [tuple(x) for x in batch_edges[['level_0', 'level_1', 0]].to_numpy()]

    label = y

    batch_edges = np.array_split(batch_edges, 3)
    
    #randomly assign labelled, unlabelled -- TO DO: Fix This in pre-processing.
    edges_ll = batch_edges[0]
    edges_lu = batch_edges[1]
    edges_uu = batch_edges[2]

    
    weights_ll = [x[2] for x in edges_ll]
    edges_ll = [(x[0], x[1]) for x in edges_ll]

    u_ll = [int(e[0]) for e in edges_ll]

    # number of incident edges for nodes u
    c_ull = [1 / len(edges[edges['level_0'] == n]) for n in u_ll]
    v_ll = [e[1] for e in edges_ll]
    c_vll = [1 / len(edges[edges['level_0'] == n]) for n in v_ll]

    nodes_ll_u = X.iloc[u_ll].to_numpy()

    labels_ll_u = np.zeros((0,2))
    if len(nodes_ll_u) > 0:
        labels_ll_u = np.vstack([label.loc[n] for n in u_ll])

    nodes_ll_v = X.iloc[v_ll].to_numpy()

    labels_ll_v = np.zeros((0,2))
    if len(nodes_ll_v) > 0:
        labels_ll_v = np.vstack([label.loc[n] for n in v_ll])

        
    weights_lu = [x[2] for x in edges_lu]
    edges_lu = [(x[0], x[1]) for x in edges_lu]
    u_lu = [e[0] for e in edges_lu]
    c_ulu = [1 / len(edges[edges['level_0'] == n]) for n in u_lu]
    nodes_lu_u = X.iloc[u_lu].to_numpy()
    nodes_lu_v = X.iloc[[e[1] for e in edges_lu]].to_numpy()

    labels_lu = np.zeros((0,2))
    if len(nodes_lu_u) > 0:
        labels_lu = np.vstack([label.loc[n] for n in u_lu])

        
    weights_uu = [x[2] for x in edges_uu]
    edges_uu = [(x[0], x[1]) for x in edges_uu]
    nodes_uu_u = X.iloc[[e[0] for e in edges_uu]].to_numpy()
    nodes_uu_v = X.iloc[[e[1] for e in edges_uu]].to_numpy()


    return torch.from_numpy(nodes_ll_u), torch.from_numpy(nodes_ll_v), torch.from_numpy(labels_ll_u), torch.from_numpy(labels_ll_v), \
           torch.from_numpy(nodes_uu_u), torch.from_numpy(nodes_uu_v), torch.from_numpy(nodes_lu_u), torch.from_numpy(nodes_lu_v), \
           torch.from_numpy(labels_lu), torch.FloatTensor(weights_ll), torch.FloatTensor(weights_lu), torch.FloatTensor(weights_uu), \
           torch.FloatTensor(c_ull), torch.FloatTensor(c_vll), torch.FloatTensor(c_ulu)
    # Note as of now all incident edges are the same!

    
    
def batch_iter(batch_size, training_size, c_merged, X, y, batch_edges):
    edges = c_merged
    edges.values[[np.arange(len(edges))]*2] = np.nan
    edges = edges.stack().reset_index()
    
    training_edges = edges[edges['level_0'].isin(batch_edges.index)]
    

    #num_batches = len(training_edges) / batch_size
    data_size = len(training_edges)
    

    
    if data_size % batch_size > 0:
        num_batches = int(data_size / batch_size) + 1
    else:
        num_batches = data_size / batch_size
        

    

    for batch_num in range(int(num_batches)):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield next_batch(training_edges,start_index,end_index,edges,X,y)


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


    
def my_softmax_cross_entropy_with_logits(logits, labels):
    return torch.sum(- labels * torch.nn.functional.log_softmax(logits, -1), -1)
    
def my_loss(scores_u1, scores_v1, scores_u2, scores_v2, scores_u3, scores_v3, u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu):
    loss_function = torch.sum(w_ll * my_softmax_cross_entropy_with_logits(scores_u1, torch.nn.functional.softmax(scores_v1))) \
                            + c_ull * my_softmax_cross_entropy_with_logits(scores_u1, lu1) \
                            + c_vll * my_softmax_cross_entropy_with_logits(scores_v1, lv1) \
                            + torch.sum(w_lu * my_softmax_cross_entropy_with_logits(scores_u2, torch.nn.functional.softmax(scores_v2))) \
                            + c_ulu * my_softmax_cross_entropy_with_logits(scores_u2, lu2) \
                            + torch.sum(w_uu * my_softmax_cross_entropy_with_logits(scores_u3, torch.nn.functional.softmax(scores_v3)))
    return loss_function
                                        
                                        
def my_test_loss(scores_u1, scores_v1, scores_u2, scores_v2, scores_u3, scores_v3, u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu):
    loss_function = torch.sum(w_ll * my_softmax_cross_entropy_with_logits(scores_u1, torch.nn.functional.softmax(scores_v1))) \
                    + torch.sum(w_lu * my_softmax_cross_entropy_with_logits(scores_u2, torch.nn.functional.softmax(scores_v2))) \
                    + torch.sum(w_uu * my_softmax_cross_entropy_with_logits(scores_u3, torch.nn.functional.softmax(scores_v3)))
    return loss_function

def train(model,training_size,learning_rate,num_epochs, batch_size, c_merged, X_df, y, X_train):
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    
    for epoch in range(0,30):
        for i, (u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu) in enumerate(batch_iter(batch_size, training_size, c_merged, X_df, y, X_train)):   # Load a batch of images with its (index, data, class)

            
            #batch_size = len(labels)
            optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
            
            #print(torch.from_numpy(u1))
            
            scores_u1 = model(u1.float())            # Forward pass: compute the output class given a image
            scores_v1 = model(v1.float())
            scores_u2 = model(u2.float())
            scores_v2 = model(v2.float())
            scores_u3 = model(u3.float())
            scores_v3 = model(v3.float())
            
       
            loss = my_loss(scores_u1, scores_v1, scores_u2, scores_v2, scores_u3, scores_v3, u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu).mean()


            loss.backward()                                   # Backward pass: compute the weight
            optimizer.step()                                  # Optimizer: update the weights of hidden nodes

            if (i+1) % 20 == 0:                              # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, 0, loss.item()))
 

def prepare_for_evaluation(model, batch_size, training_size, c_merged, X_df, y, X_test):
    all_predicted = []
    all_labels = []
    for i, (u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu) in enumerate(batch_iter(batch_size, training_size, c_merged, X_df, y, X_test)):   # Load a batch of images with its (index, data, class)
        # TODO: 
        outputs = model(u1.float()) 
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        all_predicted.extend([int(v) for v in predicted])
        all_labels.extend([int(v) for v in lu1])
    return all_predicted,all_labels