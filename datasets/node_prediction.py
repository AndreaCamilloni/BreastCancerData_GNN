from __future__ import division
from math import floor
import math
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from torchvision import models
import torch.nn as nn


import utils
# try:
#     from neoConnector import all_cells_with_n_hops_in_area, get_all_edges
# except ImportError:
#     from .neoConnector import all_cells_with_n_hops_in_area, get_all_edges

np.random.seed(0)


class_map = {0: 'AMBIGUOUS', 1: 'nonTIL_stromal', 2: 'other', 3: 'sTIL', 4: 'tumor'}


class BCSSGraphDatasetSUBGCN(Dataset):

    def __init__(self, path, mode='train',
                 num_layers=2,
                 data_split=[0.8, 0.2], add_self_edges=False):
        """
        Parameters
        ----------
        path : list
            List with filename, coordinates and path to annotation. For example, ['P7_HE_Default_Extended_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt']
        mode : str
            One of train, val or test. Default: train.
        num_layers : int
            Number of layers in the computation graph. Default: 2.
        data_split: list
            Fraction of edges to use for graph construction / train / val / test. Default: [0.85, 0.08, 0.02, 0.03].
        """
        super().__init__()

        self.path = path
        self.mode = mode
        self.num_layers = num_layers
        self.data_split = data_split

        print('--------------------------------')
        print('Reading edge dataset from {}'.format(self.path[0]))

        ########## MINE ###########
        # Cells, distance_close_to_edges
        edge_path = path[1]
        node_path = path[2]
    

        # with glob
        edges = pd.read_csv(edge_path, dtype={'source': np.int16, 'target': np.int16, 'distance': np.float16})
        nodes = pd.read_csv(node_path)

 


        #edges["type"] = 0

        #edges_crossing = edges.copy()
        #edges_crossing = edges[edges["type"] == 1]

        #edges['type'] = edges['type'].replace(1, 0)

        col_row_len = len(nodes['id'])
        
         

       # Initialize a Numpy array for better performance
        distances_close_to_edges = np.zeros((col_row_len, col_row_len))


        #distances_close_to_edges = pd.DataFrame(0, index=np.arange(col_row_len), columns=np.arange(col_row_len))
        #delta_entropy_edges = pd.DataFrame(0, index=np.arange(col_row_len), columns=np.arange(col_row_len))
        #neighborhood_similarity_edges = pd.DataFrame(0, index=np.arange(col_row_len), columns=np.arange(col_row_len))

        for i, row in edges.iterrows():
            #source = row['source']
            #target = row['target']

            distances_close_to_edges[int(row['source']), int(row['target'])] = float(row['distance'])
            distances_close_to_edges[int(row['target']), int(row['source'])] = float(row['distance'])
            
            #distance = float(row['distance'])
           
            #delta_entropy = float(row['Delta_Entropy'])
            #sorenson_neigh_similarity = float(row['Sorenson_Similarity'])
        
            #distances_close_to_edges[source][target] = distance
            #distances_close_to_edges[target][source] = distance

            #delta_entropy_edges[source][target] = delta_entropy
            #delta_entropy_edges[target][source] = delta_entropy

            #neighborhood_similarity_edges[source][target] = sorenson_neigh_similarity
            #neighborhood_similarity_edges[target][source] = sorenson_neigh_similarity

        
        # Convert the Numpy array back to a DataFrame if needed
        #distances_close_to_edges = pd.DataFrame(distance_matrix)
        #distances_close_to_edges = distance_matrix#np.array(distances_close_to_edges)
        #delta_entropy_edges = np.array(delta_entropy_edges)
        #neighborhood_similarity_edges = np.array(neighborhood_similarity_edges)

        # coords
        coords = nodes[["x", "y"]].to_numpy()

        # process neighborhood densities
        #density_types = ["Cell_density"]
        #entropy_types = ["Node_Entropy"]

        #densities = nodes[density_types].to_numpy()
        #edge_density = np.zeros((col_row_len, col_row_len))
        #edge_densities = np.empty((0, col_row_len, col_row_len))

        
        # for i in range(len(density_types)):
        #     for _, row in edges.iterrows():
        #         source = int(row['source'])
        #         target = int(row['target'])

        #         edge_density[source][target] = float(densities[:, i][target]) - float(densities[:, i][source])
        #         edge_density[target][source] = float(densities[:, i][source]) - float(densities[:, i][target])

        #     edge_densities = np.append(edge_densities, edge_density.reshape(-1, col_row_len, col_row_len), axis=0)

        #print('*************')
        #print('Edge_density Shape : ' + str(edge_densities.shape))
        
        distances_close_to_edges = distances_close_to_edges.reshape(-1, col_row_len, col_row_len)
        #delta_entropy_edges = delta_entropy_edges.reshape(-1, col_row_len, col_row_len)
        #neighborhood_similarity_edges = neighborhood_similarity_edges.reshape(-1, col_row_len, col_row_len)

        #print('Edge_entropy Shape : ' + str(delta_entropy_edges.shape))
        #print('Edge_distance Shape : ' + str(distances_close_to_edges.shape))
        #print('Neighborhood Similarity Shape : ' + str(neighborhood_similarity_edges.shape))
        #edge_features = np.concatenate((edge_densities, delta_entropy_edges, neighborhood_similarity_edges, distances_close_to_edges), axis=0)
         
        #edge_features = distances_close_to_edges.reshape(-1, col_row_len, col_row_len)
        #print(edge_features)
        # self.edge_features = utils.normalize_edge_feature_doubly_stochastic(edge_features) ### not to be used
        self.edge_features = utils.normalize_edge_features_rows(distances_close_to_edges) ### Use it to normalise the edge features
        #self.edge_features = edge_features  ### Use only if not using the normalization feature above


        ## To DO 

        # Change utils.normalize_edge_features_rows to log function to the base e

        #####

        self.channel = distances_close_to_edges.shape[0]

        self.dist = self.edge_features#utils.normalize_edge_features_rows(distances_close_to_edges.reshape(-1, col_row_len, col_row_len))


        # all_labels_cell_types
        #nodes["gt"].replace({'AMBIGUOUS': 0, 'nonTIL_stromal': 1, 'other': 2, 'sTIL': 3, 'tumor': 4}, inplace=True) # hover-net
                     

        # nuclei features
        #nuclei_feat = nodes[["area", "perim"]].to_numpy()

        #all_labels_cell_types = nodes["gt"].to_numpy()
        all_labels_cell_types = nodes["class"].to_numpy()

        # One-hot encoding for class types
        class_columns = ['amb', 'nonTil_stroma', 'other', 'sTIL', 'tumor']

        # Create dummy variables
        nodes_with_types_zero_one = pd.get_dummies(nodes['gt'], prefix='', prefix_sep='')

        # Reindex the DataFrame to include all columns in 'class_columns', filling missing ones with 0s
        nodes_with_types_zero_one = nodes_with_types_zero_one.reindex(columns=class_columns, fill_value=0)
        # Assigning y_nodes based on 'mask' values
        y_nodes = np.where(nodes['mask'].isin([1, 19, 20]), 1, 0)

        y_class_columns = ['tumor', 'stroma', 'inflammatory', 'other'] # Necrosis aggregated with other
        # Tumor 1 19 20
        # Stroma 2
        # Inflammatory 3 10 11
        # Necrosis 4
        # Other 5 6 7 8 9 12 13 14 15 16 17 18 21
        y_mapping = {0: 'other', 1: 'tumor', 2: 'stroma', 3: 'inflammatory', 
                     4: 'other', 5: 'other', 6: 'other', 7: 'other', 8: 'other', 
                     9: 'other', 10: 'inflammatory', 11: 'inflammatory', 12: 'other', 
                     13: 'other', 14: 'other', 15: 'other', 16: 'other', 17: 'other', 
                     18: 'other', 19: 'tumor', 20: 'tumor', 21: 'other'
                }
        nodes['y_mask'] = nodes['mask'].map(y_mapping)
        y_one_hot_encoding = pd.get_dummies(nodes['y_mask'], prefix='', prefix_sep='')
        y_one_hot_encoding = y_one_hot_encoding.reindex(columns=y_class_columns, fill_value=0)
        # count number of nodes in each class
        print('Number of nodes in each class:')
        print(y_one_hot_encoding.sum(axis=0))

        
        # x_nodes as 'id' column from nodes DataFrame
        x_nodes = nodes['id'].to_numpy()

        # Count positives and negatives Tumor/Non-Tumor
        y_pos, y_neg = np.count_nonzero(y_nodes == 1), np.count_nonzero(y_nodes == 0)

        # Extract cell type scores
        cell_types_scores = nodes_with_types_zero_one.to_numpy()
        #print(cell_types_scores.shape)

        # adjacency_matrix_close_to_edges
        adjacency_matrix_close_to_edges = np.copy(distances_close_to_edges)
        adjacency_matrix_close_to_edges[adjacency_matrix_close_to_edges != 0] = 1
        self.adj = adjacency_matrix_close_to_edges
        # edge_list_close_to_edge
        #edge_list_close_to_edge = edges[["source", "target"]]
        #edge_list_close_to_edge = edge_list_close_to_edge.to_numpy()

        # edge_list_crossing_edges
        # edge_list_crossing_edges = edges_crossing.to_numpy()

        #self.am_close_to_edges_including_distances = distances_close_to_edges
        self.classes = all_labels_cell_types
        self.class_scores = cell_types_scores
        self.coords = coords

        print('Finished reading data.')

        print('Setting up graph.') 
        self.nodes_count = len(coords) # Count vertices
        self.edges_count = len(edges) # Count edges

        self.features = torch.from_numpy(cell_types_scores).float()  # Cell features with just one-hot encoding 
           
        
        print('self.features.shape:', self.features.shape)
        # [2] end

        print('Finished setting up graph.')

        print('Setting up examples.')


        # # Assuming y_nodes contain labels for each node (1 for positive, 0 for negative)
        # positive_node_indices = [i for i, label in enumerate(y_nodes) if label == 1]
        # negative_node_indices = [i for i, label in enumerate(y_nodes) if label == 0]

        # # Generate negative and positive examples
        # positive_examples = []
        # negative_examples = []
        # _choice = np.random.choice

        # if self.mode != 'train':  # Validation/Test Mode
        #     print("self.mode != 'train'")
        #     # In this mode, you might want to handle positive and negative examples differently
        #     # Adjust the following logic as per your validation/test requirements
        #     positive_examples = positive_examples + positive_node_indices
        #     negative_examples = negative_examples + negative_node_indices
        # else:  # Training Mode
        #     # For training, you might want to balance the number of positive and negative examples
        #     num_examples = min(len(positive_node_indices), len(negative_node_indices))

        #     positive_examples = _choice(positive_node_indices, num_examples, replace=False)
        #     negative_examples = _choice(negative_node_indices, num_examples, replace=False)

        # # Convert to numpy arrays
        # positive_examples = np.array(positive_examples, dtype=np.int64)
        # negative_examples = np.array(negative_examples, dtype=np.int64)

        # print('y_pos, y_neg', y_pos, y_neg)
        # SMTH not working with sampling
        x = x_nodes#np.vstack((positive_examples, negative_examples))
        y = y_one_hot_encoding.to_numpy()#np.concatenate((np.ones(positive_examples.shape[0]),np.zeros(negative_examples.shape[0])))#, np.full(y_unk.shape, -1) ))
  
        #print(y)
        perm = np.random.permutation(x.shape[0])
        x, y = x[perm], y[perm]  # ERROR HERE -> IndexError: too many indices for array: array is 1-dimensional,
        # but 2 were indexed
        x, y = torch.from_numpy(x).long(), torch.from_numpy(y).long()
        self.x, self.y = x, y

        print('Finished setting up examples.')

        print('Dataset properties:')
        print('Mode: {}'.format(self.mode))
        print('Number of vertices: {}'.format(self.nodes_count))
        print('Number of edges: {}'.format(self.edges_count))
        
        # print('Number of positive/negative nodes: {}/{}'.format(positive_examples.shape[0],
        #                                                             negative_examples.shape[0]))
        print('Number of tumor/non-tumor: {}/{}'.format(y_pos, y_neg))
        print('Number of examples/datapoints: {}'.format(self.x.shape[0]))

        print('--------------------------------')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_coords_and_class(self):
        return self.coords, self.classes

    # def _form_computation_graph(self, idx):
    #     """
    #     Parameters
    #     ----------
    #     idx : int or list
    #         Indices of the node for which the forward pass needs to be computed.
    #     Returns
    #     -------
    #     node_layers : list of numpy array
    #         node_layers[i] is an array of the nodes in the ith layer of the
    #         computation graph.
    #     mappings : list of dictionary
    #         mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
    #         in node_layers[i] to its position in node_layers[i]. For example,
    #         if node_layers[i] = [2,5], then mappings[i][2] = 0 and
    #         mappings[i][5] = 1.
    #     """
    #     _list, _set = list, set
    #     if type(idx) is int:
    #         node_layers = [np.array([idx], dtype=np.int64)]
    #     elif type(idx) is list:
    #         node_layers = [np.array(idx, dtype=np.int64)]

    #     for _ in range(self.num_layers):
    #         prev = node_layers[-1]
    #         arr = [node for node in prev]
    #         arr.extend([e for node in arr for e in self.node_neighbors[node]])  # add neighbors to graph
    #         arr = np.array(_list(_set(arr)), dtype=np.int64)
    #         node_layers.append(arr)
    #     node_layers.reverse()

    #     mappings = [{j: i for (i, j) in enumerate(arr)} for arr in node_layers]

    #     return node_layers, mappings

    def collate_wrapper(self, batch):
        """
        Parameters
        ----------
        batch : list
            A list of examples from this dataset. An example is (node, label).
        Returns
        -------
        adj : torch.Tensor
            adjacency matrix of entire graph
        features : torch.FloatTensor
            A (n' x input_dim) tensor of input node features.
        edge_features : torch.FloatTensor
            A 3d tensor of edge features.
        edges : numpy array
            The edges in the batch.
        labels : torch.LongTensor
            Labels (1 or 0) for the edges in the batch.
        dist : torch.Tensor
            A distance matrix
        """ 
        adj = torch.from_numpy(self.adj).float()

        features = self.features
        edge_features = torch.from_numpy(self.edge_features).float()
        #edges = np.array([sample[0].numpy() for sample in batch])
        #labels = torch.FloatTensor([sample[1] for sample in batch])
        nodes = np.array([sample[0].numpy() for sample in batch])
        labels = torch.stack([sample[1] for sample in batch]).float()
            


        dist = torch.from_numpy(self.dist)

        return adj, features, edge_features, nodes, labels, dist
        #return features, edge_features, nodes, labels, dist

    def get_dims(self):
        print("self.features.shape: {}".format(self.features.shape))
        print("input_dims (input dimension) -> self.features.shape[1] = {}".format(self.features.shape[1]))
        return self.features.shape[1], 1

    def get_channel(self):
        return self.channel

    def parse_points(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        lines = [line[:-1].split(',') for line in lines]  # Remove \n from line
        return lines
##############################################################################################################



def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    if D != 0:
        x = Dx / D
        return x
    else:
        return False

