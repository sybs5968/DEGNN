import torch
import random
import numpy as np
from torch import nn
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
import torch_geometric.utils as tgu
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader , Data
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, TAGConv, GATConv

file_path = r"data/node_classification/brazil-airports/"
test_ratio = 0.1
default_lr = 1e-4
default_wd = 0
default_epochs = 1000