from audioop import bias
from re import A
from tkinter import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
from layers import ConvolutionLayer, DGNNConvolutionLayer, AAAgregationLayer, EGNNCLayer, DAAAgregationLayer

import utils as utils


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)


class EGNNC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, channel_dim, num_layers=4, dropout=0.5, device='cpu'):
        super(EGNNC, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.channel_dim = channel_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.elu = nn.ELU()

        # First layer
        self.egnn_layers = nn.ModuleList([EGNNCLayer(input_dim, hidden_dim, channel_dim, device=device).to(device)])

        # Hidden layers
        for _ in range(self.num_layers - 2):  # -2 because we have a separate first and last layer
            self.egnn_layers.append(EGNNCLayer(hidden_dim*channel_dim, hidden_dim, channel_dim, device=device).to(device))

        # Last layer
        self.egnn_layers.append(EGNNCLayer(hidden_dim*channel_dim, output_dim, channel_dim, device=device).to(device))

    def forward(self, features, edge_features):
        features, edge_features = features.to(self.device), edge_features.to(self.device)

        x = features
        for i in range(self.num_layers):
            x = self.egnn_layers[i](x, edge_features)
            if i < self.num_layers - 1:  # No activation and dropout for the last layer
                x = self.elu(x)
                x = self.dropout(x)
        
        return x

class MLPTwoLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.5, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layer. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Probability of setting an element to 0 in dropout layer. Default: 0.5.
        device : string
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(MLPTwoLayers, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True).to(device)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True).to(device)

         
    def forward(self, features):
        x = F.relu(self.linear1(features))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)

        if self.output_dim == 1:
            out = x.reshape(-1)
            return self.sigmoid(out)

        else:
            return F.softmax(x, dim=1)
         
    
class gCNN(nn.Module):
    def __init__(self, num_channels = 13, output_dim = 1, dropout=0.5, device='cpu', input_size = 16, training = True):
        super(gCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, stride=1, padding=1).to(device)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1).to(device)
        self.fc1 = nn.Linear(32 * input_size, 128, bias=True).to(device)  # Adjust the input size based on your data
        self.fc2 = nn.Linear(128, output_dim, bias=True).to(device)  # Output size is 1  for regression
        self.dropout = dropout
        self.training = training
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print("x: ", x.shape)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        out = x.reshape(-1)
        
        return self.sigmoid(out)

class CombinedModel(nn.Module):
    def __init__(self, gnn, classifier, type = 'mlp', device='cpu'):
        super(CombinedModel, self).__init__()
        self.gnn = gnn
        self.classifier = classifier
        self.type = type
        self.sigmoid = nn.Sigmoid()
        self.device = device
         

    def forward(self, features, edge_features, nodes, graphlet_size = 10, adj = None):
        # Pass graphs through GNN to get node embeddings
        #features, edge_features = features.to(device), edge_features.to(device)
        out = self.gnn(features, edge_features)
      
        # Get node embeddings for nodes in the batch
        
        if self.type == 'mlp':
            #print("out: ", out.shape)
            node_embeddings = out[nodes]
            out = self.classifier(node_embeddings)
        elif self.type == 'gCNN':
            graphlets_emb = utils.get_graphlets(out, nodes, adj, graphlet_size, self.device)
            graphlets_emb = graphlets_emb.to(self.device)
        
            # Pass node embeddings through classifier
            out = self.classifier(graphlets_emb)
        
        return out
 