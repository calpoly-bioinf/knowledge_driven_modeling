#!/usr/bin/env python
# coding: utf-8

# ### Load and preprocess data

# In[1]:


from collections import defaultdict, OrderedDict

DATA_DIR = "data/PubMed-Diabetes/"
EDGE_PATH = DATA_DIR + "Pubmed-Diabetes.DIRECTED.cites.tab"
NODE_PATH = DATA_DIR + "Pubmed-Diabetes.NODE.paper.tab"
TF_IDF_DIM = 500

# Load and process graph links
print("Loading and processing graph links...")
node_pairs = set()
with open(EDGE_PATH, 'r') as f:
    next(f)  # skip header
    next(f)  # skip header
    for line in f:
        columns = line.split()
        src = int(columns[1][6:])
        dest = int(columns[3].strip()[6:])
        node_pairs.add((src, dest))
        
# Load and process graph nodes
print("Loading and processing graph nodes...")
node2vec = OrderedDict()
node2label = dict()
class_1 = list()
class_2 = list()
class_3 = list()
with open(NODE_PATH, 'r') as f:
    next(f)  # skip header
    vocabs = [e.split(':')[1] for e in next(f).split()[1:]]
    for line in f:
        columns = line.split()
        node = int(columns[0])
        label = int(columns[1][-1])
        tf_idf_vec = [0.0] * TF_IDF_DIM

        for e in columns[2:-1]:
            word, value = e.split('=')
            tf_idf_vec[vocabs.index(word)] = float(value)

        node2vec[node] = tf_idf_vec
        node2label[node] = label - 1
        if label == 1:
            class_1.append(node)
        elif label == 2:
            class_2.append(node)
        elif label == 3:
            class_3.append(node)

# Debug statistics
print("Number of links:", len(node_pairs))
assert len(node2vec) == (len(class_1) + len(class_2) + len(class_3))
print("Number of nodes:", len(node2vec))
print("Number of nodes belong to Class 1", len(class_1))
print("Number of nodes belong to Class 2", len(class_2))
print("Number of nodes belong to Class 3", len(class_3))


# ### Neural Network related parameters

# In[2]:


MODEL_DIR = "model/"
TEST_SIZE = 1000
SEED_NODES = 20
NUM_CATEGORIES = 3

ALPHA = 0.2
HIDDEN_1_DIM = 250
HIDDEN_2_DIM = 100

NUM_EPOCH = 12
BATCH_SIZE = 100
LEARNING_RATE = 0.0001


# ### Split data into train/test set

# In[3]:


# Important variables from previous cells: node_pairs, class_1, class_2, class_3
test_nodes = class_1[-TEST_SIZE:] + class_2[-TEST_SIZE:] + class_3[-TEST_SIZE:]
train_node_pairs = []
for src, dest in node_pairs:
    if not (src in test_nodes or dest in test_nodes):
        train_node_pairs.append((src, dest))

seed_nodes = class_1[:SEED_NODES] + class_2[:SEED_NODES] + class_3[:SEED_NODES]


# ### Model Architecture

# In[14]:


import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm

class MY_NGM_FFNN(nn.Module):
    def __init__(self, alpha, input_dim, hidden1_dim, hidden2_dim, output_dim, device=torch.device('cpu')):
        super(MY_NGM_FFNN, self).__init__()

    
        
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)
        self.alpha = alpha
        
        self.device = device
        self.to(device)
        




    def save(self, output_dir, model_name):
        print("Saving model...")
        torch.save(self.state_dict(), output_dir + model_name + ".pt")
        print("Model saved.")

    def load(self, output_dir, model_name):
        print("Loading model...")
        self.load_state_dict(torch.load(output_dir + model_name + ".pt"))
        print("Model loaded.")
        
    def forward(self, tf_idf_vec):
        # First feed-forward layer
        hidden1 = F.relu(self.hidden1(tf_idf_vec))

        # Second feed-forward layer
        hidden2 = F.relu(self.hidden2(hidden1))
        
        
        # Output layer
        return F.log_softmax(self.output(hidden2), -1)
        
    
    def reset_parameters(self):
        self.hidden1.reset_parameters()
        self.hidden2.reset_parameters()
        self.output.reset_parameters()
    
    def get_last_hidden(self, tf_idf_vec):
        # First feed-forward layer
        hidden1 = F.relu(self.hidden1(tf_idf_vec))

        # Second feed-forward layer
        return F.relu(self.hidden2(hidden1))

    
    def train_(self, seed_nodes, train_node_pairs, node2vec, node2label, 
               num_epoch, batch_size, learning_rate):
        print("Training...")
        self.train()
        
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        node2neighbors = defaultdict(list)
        for src, dest in train_node_pairs:
            node2neighbors[src].append(dest)
            node2neighbors[dest].append(src)
            
        labeled_nodes = dict()
        for node in seed_nodes:
            labeled_nodes[node] = node2label[node]

        iteration = 1
        while iteration < 2:
            print("=" * 80)
            print("Generation: {} (with {} labeled nodes)".format(iteration, len(labeled_nodes)))
            iteration += 1

            for e in range(NUM_EPOCH):
                train_node_pairs_cpy =  train_node_pairs[:]
                total_loss = 0
                count = 0
                while train_node_pairs_cpy:
                    optimizer.zero_grad()
                    
                    loss = torch.tensor(0, dtype=torch.float32, device=self.device)
                    #label_label_loss = defaultdict(list)
                    #label_unlabel_loss = defaultdict(list)
                 
                    
                    try:
                        batch = random.sample(train_node_pairs_cpy, batch_size)
                    except ValueError:
                        break
                        
                        
                    for (src, dest) in batch:
                        #print(src, dest)
                        count += 1
                        train_node_pairs_cpy.remove((src, dest))
                        #print(len(train_node_pairs_cpy))
                        
                        src_vec = torch.tensor(node2vec[src])
                        dest_vec = torch.tensor(node2vec[dest])
                        
                        
                        if src in labeled_nodes:
                            # LU / LL LOSS
                            src_target = torch.tensor([labeled_nodes[src]])
                            src_softmax = self.forward(torch.tensor(src_vec))
                            src_incident_edges = len(node2neighbors[src])
                            loss += loss_function(src_softmax.view(1, -1), src_target) * (1 / src_incident_edges)
                            
                        if dest in labeled_nodes:
                            # LL LOSS
                            dest_target = torch.tensor([labeled_nodes[dest]])
                            dest_softmax = self.forward(torch.tensor(dest_vec))
                            dest_incident_edges = len(node2neighbors[dest])
                            loss += loss_function(dest_softmax.view(1, -1), dest_target) * (1 / dest_incident_edges)
                        
                        loss += self.alpha * torch.dist(self.get_last_hidden(src_vec), self.get_last_hidden(dest_vec))
                                
                

                   
                    if loss.item() != 0:
                        assert not torch.isnan(loss)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        del loss
      
                    

                avg_loss = total_loss / len(labeled_nodes)
                print("Epoch: {} Loss: {} (avg: {})".format(e + 1, total_loss, avg_loss))
            
            for node in list(labeled_nodes.keys()):
                label = labeled_nodes[node]
                for neighbor in node2neighbors[node]:
                    if neighbor not in labeled_nodes:
                        labeled_nodes[neighbor] = label

         

    def predict(self, tf_idf_vec):
        return torch.argmax(self.forward(tf_idf_vec)).item()
        
    def evaluate(self, test_nodes, node2vec, node2label):
        self.eval()
        print("hi")
        correct_count = 0
        for node in test_nodes:
            predicted = self.predict(torch.tensor(node2vec[node], device=self.device))
            print(predicted)
            if predicted == node2label[node]:
                correct_count += 1

        return float(correct_count) / len(test_nodes)


# ### Baseline feed-forward neural network

# In[16]:


# Important variable from previous cells: node_pairs, node2vec, node2label, seed_nodes, train_node_pairs, test_nodes
from datetime import datetime
baseline_model = MY_NGM_FFNN(0.2, TF_IDF_DIM, HIDDEN_1_DIM, HIDDEN_2_DIM, NUM_CATEGORIES)
start = datetime.now()
baseline_model.train_(seed_nodes, train_node_pairs, node2vec, node2label, NUM_EPOCH, BATCH_SIZE, LEARNING_RATE)
baseline_time = (datetime.now()-start).total_seconds()


# ### Evaluations

# In[17]:


# Important variable from previous cells: node2vec, node2label, test_nodes

print(baseline_model.evaluate(test_nodes, node2vec, node2label))


# In[9]:


print([node2label[i] for i in test_nodes])


# ### Save model

# In[9]:


baseline_model.save(MODEL_DIR, "PubMed_baseline")


# In[33]:



# In[ ]:




