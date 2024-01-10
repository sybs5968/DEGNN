import utils
import torch
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
labels , node_id_mapping = utils.read_label(r"data/node_classification/brazil-airports/" , "node_classification")
edges = utils.read_edges(r"data/node_classification/brazil-airports/" , node_id_mapping) # [(u1 , v1) , (u2 , v2)] 
G = nx.Graph(edges)
attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)
attributes += np.expand_dims(np.log(utils.get_degrees(G)+1), 1).astype(np.float32)
G.graph['attributes'] = attributes
labels = np.array(labels) if labels is not None else None
# print(labels.shape)
train_indices, test_indices = train_test_split(list(range(131)), test_size=30, stratify=labels)
edge_index = torch.tensor(list(G.edges)).long().t().contiguous()
# print(edge_index.shape)

# print(edge_index[[1 , 0] , ].shape)

edge_index = torch.cat([edge_index, edge_index[[1, 0], ]], dim=-1)
print(np.array(1))
print(len(np.array([1])))
# print(edge_index.shape)
sp_length = utils.get_features_sp_sample(G , np.array([1]) , 3)
print(sp_length.shape)
onehot_encoding = np.eye(5, dtype=np.float64)  # [n_features, n_features]
features_sp = onehot_encoding[sp_length].sum(axis=1)
print(features_sp)
