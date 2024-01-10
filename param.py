import torch
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
import torch_geometric.utils as tgu
from torch_geometric.data import DataLoader , Data
from sklearn.model_selection import train_test_split

file_path = r"data/node_classification/usa-airports/"
test_ratio = 0.1