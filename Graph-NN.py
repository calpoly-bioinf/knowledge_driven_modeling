#!/usr/bin/env python
# coding: utf-8

# # Initialize Pytorch

# In[1]:


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
from numpy import genfromtxt
import scipy
from sklearn.model_selection import train_test_split


# # Load Data Set

# In[2]:


node_values = genfromtxt('./BlogCatalog-dataset/data/group-edges.csv', delimiter=',').astype(int)
node_values = np.sort(node_values, axis=0)
#my_data = np.delete(my_data, 0, 1).reshape(-1)
node_values


# In[3]:


nodes = genfromtxt('./BlogCatalog-dataset/data/nodes.csv', delimiter=',').astype(int)
groups = genfromtxt('./BlogCatalog-dataset/data/groups.csv', delimiter=',').astype(int)
edges = genfromtxt('./BlogCatalog-dataset/data/edges.csv', delimiter=',').astype(int)


# In[4]:


edges_df = pd.read_csv('./BlogCatalog-dataset/data/edges.csv', names=['node', 'node(2)'])


# In[5]:


edges_df = pd.crosstab(edges_df['node'], edges_df['node(2)'])
idx = edges_df.columns.union(edges_df.index)
edges_df = edges_df.reindex(index = idx, columns=idx, fill_value=0)


# In[6]:


edges_df


# In[7]:


data = scipy.sparse.csr_matrix(edges_df.values)


# In[8]:


node_values_df = pd.read_csv('./BlogCatalog-dataset/data/group-edges.csv', names=['node(2)', 'val'])
node_values_df.drop_duplicates(subset ="node(2)", keep = "last", inplace = True) 
node_values_df.sort_values('node(2)', inplace=True)
node_values_df = node_values_df.set_index('node(2)')
group_data = node_values_df.values
group_data = np.delete(group_data, 0, 1).reshape(-1)


# In[9]:


node_values_df['is_val'] = (node_values_df['val'] == 1)


# In[10]:


node_values_df


# In[11]:


new_df = pd.merge(edges_df, node_values_df, left_index=True, right_index=True)


# In[12]:


new_df['is_val'] = new_df['is_val'].astype(int)


# In[13]:


new_df.to_csv('./data.csv')


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(new_df[new_df.columns[:-2]], new_df['is_val'], test_size=0.33, random_state=42)
X_train = torch.tensor(X_train.values).float()
y_train = torch.tensor(y_train.values).float()


# In[15]:


X_test = torch.tensor(X_test.values).float()
y_test = torch.tensor(y_test.values).float()


# In[16]:


import torch.utils.data as data_utils
#X_train = torch.tensor(X_train.values).float()
#labels = new_df[new_df.columns[-1:]].values.flatten()
#labels = torch.tensor(labels).float()
#train = data_utils.TensorDataset(new_df[new_df.columns[:-2]], new_df[new_df.columns[-1:]])


# In[17]:


from torch.utils.data import Dataset
class MyDataset(Dataset):
  def __init__(self, values, labels):
    super(MyDataset, self).__init__()
    self.values = values
    self.labels = labels

  def __len__(self):
    return len(self.values)  # number of samples in the dataset

  def __getitem__(self, index):
    return self.values[index], self.labels[index]


# In[18]:



# For unbalanced dataset we create a weighted sampler                       

#train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True,                              
#                                                             sampler = sampler, num_workers=args.workers, pin_memory=True)     


# In[19]:


from torch.utils.data import DataLoader
import copy
batch_size=10
dataset = MyDataset(X_train, y_train)

counts = pd.Series(y_train).value_counts()
weights = len(y_train) / (len(np.unique(y_train)) * counts)
print(weights)
y_train_weighted = np.array(y_train)
for i in counts.index:
    print(weights.loc[i])
    y_train_weighted[y_train_weighted==i] = weights.loc[i]
y_train_weighted
#weights = make_weights_for_balanced_classes(y_train, len(np.unique(y_train)))  
#print(weights)
#weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(y_train_weighted, len(y_train_weighted))                     

dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# In[20]:


testset = MyDataset(X_test, y_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


# # Initialize NN

# ## Initialize Hyperparameters

# In[21]:


input_size = 10312   # The image size = 1x10312 = 10312
hidden_size = 50       # The number of nodes at the hidden layer
num_classes = 2       # The number of output classes. In this case, from 1 to 39
num_epochs = 5         # The number of times entire dataset is trained
batch_size = 10       # The size of input data took for one iteration
learning_rate = 0.001  # The speed of convergence
train_test_split = 0.8


# In[22]:


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


# In[23]:


model = Net(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


# In[ ]:





# In[29]:


for epoch in range(num_epochs):
#for epoch in range(0,1):
    for i, (images, labels) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)
        images = Variable(images.view(-1, 1*10312))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
        labels = Variable(labels)
        
        
        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        outputs = model(images.float())                             # Forward pass: compute the output class given a image
        #if epoch == num_epochs-1:
        #    print(outputs, labels.long())
        loss = criterion(outputs, labels.long())                 # Compute the loss: difference between the output class and the pre-given label
        
        loss.backward()                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes
        
        if (i+1) % 100 == 0:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.item()))


# In[30]:


correct = 0
total = 0
all_predicted = []
all_labels = []
for i, (images, labels) in enumerate(testloader):
    images = Variable(images.view(-1, 1*10312)) 
    outputs = model(images.float()) 
    _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
    total += labels.size(0)                    # Increment the total count
    correct += (predicted == labels).sum()     # Increment the correct count
    all_predicted += list(predicted)
    all_labels += (labels)
    
print('Accuracy: %d %%' % (100 * correct / total))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(all_labels,all_predicted))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




