import random
import numpy as np
import networkx as nx
from copy import deepcopy
from torch_geometric.data import DataLoader , Data

def read_file():
    """读取数据文件

    Returns:
        networkx , np.array(num_nodes): G , labels , 图和标签
    """
    path = r"data/node_classification/usa-airports/"
    edges = []
    labels = []
    node_id_mapping = dict()
    with open(path + "labels.txt") as f:
        for id , line in enumerate(f.readlines()):
            old_id , label = line.strip().split()
            labels.append(int(label))
            node_id_mapping[old_id] = id
    with open(path + "edges.txt") as f:
        for line in f.readlines():
            u , v = line.strip().split()
            edges.append((node_id_mapping[u] , node_id_mapping[v]))
    G = nx.Graph(edges)
    G_degree = np.array([G.degree[i] for i in range(G.number_of_nodes())])
    attributes = np.expand_dims(np.log(G_degree + 1) , 1)
    G.graph["attributes"] = attributes
    labels = np.array(labels)
    # print("labels shape" , len(labels.shape))
    return G , labels

def get_data(G , labels):
    G = deepcopy(G)
    set_indices = list(range(len(labels)))
    random.shuffle(set_indices)
    labels = labels[set_indices]
    set_indices = np.expand_dims(set_indices , 1) # shape num_nodes * 1
    # train_mask , val_test_mask = 

