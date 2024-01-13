import os
from param import *

def read_file(args):
    """读取数据文件

    Returns:
        networkx , np.array(num_nodes): G , labels , 图和标签
    """
    edges , node_id_mapping = [] , dict()
    if args.dataset in ["brazil-airports"]:
        task = "node_classification"
    elif args.dataset in ["celegans_small"]:
        task = "link_prediction"
    file_path = "data/" + task + '/' + args.dataset + '/'
    if task == "node_classification":
        labels = []
        with open(file_path + "labels.txt") as f:
            for id , line in enumerate(f.readlines()):
                old_id , label = line.strip().split()
                labels.append(int(label))
                node_id_mapping[old_id] = id
    else:
        labels = None
        with open(file_path + "edges.txt") as f:
            nodes = []
            for line in f.readlines():
                nodes.extend(line.strip().split()[:2])
            nodes = sorted(list(set(nodes)))
            node_id_mapping = {old_id : new_id for new_id , old_id in enumerate(nodes)}

    with open(file_path + "edges.txt") as f:
        for line in f.readlines():
            u , v = line.strip().split()
            edges.append([node_id_mapping[u] , node_id_mapping[v]])

    G = nx.Graph(edges)
    G_degree = np.array([G.degree[i] for i in range(G.number_of_nodes())] , dtype=np.float32)
    attributes = np.expand_dims(np.log(G_degree + 1) , 1).astype(np.float32)
    G.graph["attributes"] = attributes
    labels = np.array(labels)
    return G , labels , task

def get_features_sp(G , set_index , max_sp = 3):
    """获取最短路特征

    Args:
        G (networkx): 图
        set_index (list): 起点(们)
        max_sp (int, optional): 最大长度. Defaults to 3.

    Returns:
        np.array(num_nodes * dim(max_sp + 2)): 返回每个点最短路的独热编码
    """

    # 最大的长度是3，所以有0，1，2，3四个，第五个猜测表示-1即没有联通
    dim = max_sp + 2

    sp_length = np.ones((G.number_of_nodes() , len(set_index)) , dtype=np.int32) * -1

    for i , node in enumerate(set_index):
        for u , dis in nx.shortest_path_length(G , source=node).items():
            sp_length[u , i] = dis
    
    sp_length = np.minimum(sp_length, max_sp)
    onehot_encoding = np.eye(dim, dtype=np.float64)
    features_sp = onehot_encoding[sp_length].sum(axis=1)
    return features_sp

def get_data_sample(G , set_index , hop_num , label):
    set_index = list(set_index)
    if len(set_index) > 1:
        G = G.copy()
        G.remove_edges_from(combinations(set_index , 2))

    edge_index = torch.tensor(list(G.edges)).long().t().contiguous()
    edge_index = torch.cat([edge_index , edge_index[[1 , 0] , :]] , dim = -1)

    subgrap_node_old_index , new_edge_index , new_set_index , edge_mask = tgu.k_hop_subgraph(
    torch.tensor(set_index).long() , hop_num , edge_index , num_nodes = G.number_of_nodes() , relabel_nodes=True)
    num_nodes = subgrap_node_old_index.shape[0]
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32) , create_using=type(G))
    new_G.add_nodes_from(np.arange(num_nodes , dtype=np.int32))

    x_list = []
    attributes = G.graph["attributes"]
    if attributes is not None:
        new_attributes = torch.tensor(attributes , dtype=torch.float32)[subgrap_node_old_index]
        if new_attributes.dim() < 2:
            new_attributes.unsqueeze_(1)
        x_list.append(new_attributes) # col = 1
    features_sp = torch.from_numpy(get_features_sp(new_G , np.array(new_set_index))).float()
    x_list.append(features_sp) # col = 5
    x = torch.cat(x_list , dim=-1)
    y = torch.tensor([label], dtype=torch.long) if label is not None else torch.tensor([0], dtype=torch.long)
    new_set_index = new_set_index.long().unsqueeze(0)
    return Data(x=x , edge_index=new_edge_index , y=y , set_indices=new_set_index)

def extract_subgaphs(G , labels , set_indices , prop_depth = 1 , layers = 2):
    """获取子图列表

    Args:
        G (networkx): 图
        labels (np.array(num_nodes * 1)): 节点特征
        set_index (np.array(num_nodes * 1)): 乱序映射
        prop_depth (int, optional): 深度. Defaults to 1.
        layers (int, optional): 层数. Defaults to 2.
    Returns:
        list[Data]: 返回子图列表
    """

    data_list = []
    hop_num = prop_depth * layers + 1
    for i in tqdm(range(set_indices.shape[0])):
        data = get_data_sample(G , set_indices[i] , hop_num , labels[i] if labels is not None else None)
        data_list.append(data)
    return data_list

def load_datasets(data_list , train_mask , val_mask , test_mask , batch_size = 64):
    """加载数据

    Args:
        data_list (list[Data]): 子图列表
        train_mask (np.array(num_nodes)): 训练集mask
        val_test_mask (np.array(num_nodes)): 验证集和测试集mask
    
    Returns:
        train_loader , val_loader , test_loader
    """
    train_set = [data_list[i] for i,x in enumerate(train_mask) if x]
    val_set   = [data_list[i] for i,x in enumerate(val_mask)   if x]
    test_set  = [data_list[i] for i,x in enumerate(test_mask)  if x]
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader   = DataLoader(val_set  , batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader  = DataLoader(test_set , batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader , val_loader , test_loader

def get_data(G , labels , task , args , size = 1.0):
    """获取数据集加载器

    Args:
        G (networkx): 图
        labels (np.array(num_nodes)): 标签
        size (float, optional): 要使用数据的百分比. Defaults to 1.0.

    Returns:
        _type_: _description_
    """

    G = deepcopy(G)
    if labels:

        set_indices = np.random.choice(G.number_of_nodes() , G.number_of_nodes() , replace=False)
        labels = labels[set_indices]
        set_indices = np.expand_dims(set_indices , dim = 1)

        data_list = extract_subgaphs(G , labels , set_indices)

        train_indices , val_test_indices = train_test_split(list(range(set_indices.shape[0])) , test_size=args.test_ratio * 2 , stratify=labels)
        val_test_labels = np.array([data_list[i].y for i in val_test_indices])
        val_indices , test_indices = train_test_split(val_test_indices , test_size=0.5 , stratify=val_test_labels)
        train_mask = get_mask(train_indices , set_indices.shape[0])
        val_mask   = get_mask(val_indices   , set_indices.shape[0])
        test_mask  = get_mask(test_indices  , set_indices.shape[0])

    else:
        G = G.to_undirected()
        pos_edges , neg_edges = [] , []
        if task == "link_prediction":
            set_size = 2
            pos_edges = np.array(list(G.edges) , dtype=np.int32)
        elif task == "triplet_prediction":
            set_size = 3
            tmp = set(frozenset([node1, node2, node3]) for node1 in G for node2 , node3 in combinations(G.neighbors(node1) , 2) if G.has_edge(node2 , node3))
            pos_edges = [list(x) for x in tmp]
        while len(neg_edges) < pos_edges.shape[0]:
            tmp = [int(random.random() * G.number_of_nodes()) for _ in range(set_size)]
            for node1 , node2 in combinations(tmp , 2):
                if not G.has_edge(node1 , node2):
                    neg_edges.append(tmp)
                    break
        neg_edges = np.array(neg_edges , dtype=np.int32)

        number_of_posedges = pos_edges.shape[0]
        pos_test_size = int(number_of_posedges * args.test_ratio)
        set_indices = np.concatenate([pos_edges , neg_edges] , axis=0)

        test_pos_indices = random.sample(range(number_of_posedges) , pos_test_size)

        test_neg_indices = list(range(number_of_posedges , number_of_posedges + pos_test_size))

        test_mask = get_mask(test_pos_indices + test_neg_indices , number_of_posedges * 2)

        train_mask = np.ones_like(test_mask) - test_mask

        labels = np.concatenate([np.ones((number_of_posedges)) , np.zeros((number_of_posedges))]).astype(np.int32)
        G.remove_edges_from([node_pair for set_index in list(set_indices[test_pos_indices]) for node_pair in combinations(set_index , 2)])

        permutation = np.random.permutation(2 * number_of_posedges)
        labels = labels[permutation]
        set_indices = set_indices[permutation]
        test_mask = test_mask[permutation]
        train_mask = train_mask[permutation]
        data_list = extract_subgaphs(G , labels , set_indices)
        val_test_indices = np.arange(len(data_list))[test_mask.astype(bool)]
        val_test_labels = np.array([data_list[i].y for i in val_test_indices])
        val_indices , test_indices = train_test_split(val_test_indices , test_size=0.5 , stratify=val_test_labels)
        val_mask = get_mask(val_indices , number_of_posedges * 2)
        test_mask = get_mask(test_indices , number_of_posedges * 2)


    train_loader , val_loader , test_loader = load_datasets(data_list , train_mask , val_mask , test_mask)
    return train_loader , val_loader , test_loader , len(np.unique(labels))

def get_mask(indices , length):
    mask = np.zeros((length) , dtype=np.int8)
    mask[indices] = 1
    return mask


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)