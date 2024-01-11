from param import *

def read_file():
    """读取数据文件

    Returns:
        networkx , np.array(num_nodes): G , labels , 图和标签
    """
    edges = []
    labels = []
    node_id_mapping = dict()
    with open(file_path + "labels.txt") as f:
        for id , line in enumerate(f.readlines()):
            old_id , label = line.strip().split()
            labels.append(int(label))
            node_id_mapping[old_id] = id
    with open(file_path + "edges.txt") as f:
        for line in f.readlines():
            u , v = line.strip().split()
            edges.append((node_id_mapping[u] , node_id_mapping[v]))
    G = nx.Graph(edges)
    G_degree = np.array([G.degree[i] for i in range(G.number_of_nodes())])
    attributes = np.expand_dims(np.log(G_degree + 1) , 1)
    G.graph["attributes"] = attributes
    labels = np.array(labels)
    return G , labels

def get_data_sample(G , set_index , hop_num , label , feature_flags = (True , False) , max_sprw = (3 , 0)):
    edge_index = torch.tensor(list(G.edges)).long().t().contiguous()
    edge_index = torch.cat(edge_index , edge_index[[1 , 0] , :] , dim = -1)
    subgrap_node_old_index , new_edge_index , new_set_index , edge_mask = tgu.k_hop_subgraph(
        torch.tensor(set_index).long() , hop_num , edge_index , num_nodes = G.number_of_nodes() , relabel_nodes=True)
    num_nodes = subgrap_node_old_index.shape[0]
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32) , create_using=type(G))
    new_G.add_nodes_from(np.arange(num_nodes , dtype=np.int32))

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
    # print(onehot_encoding[sp_length].shape , sp_length.shape)
    features_sp = onehot_encoding[sp_length].sum(axis=1)
    return features_sp


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
    num_nodes = set_indices.shape[0]
    for i in tqdm(range(int(num_nodes))):
        
        edge_index = torch.tensor(list(G.edges)).long().t().contiguous()
        edge_index = torch.cat([edge_index , edge_index[[1 , 0] , :]] , dim = -1)
        subgrap_node_old_index , new_edge_index , new_set_index , edge_mask = tgu.k_hop_subgraph(
        torch.tensor([i]).long() , hop_num , edge_index , num_nodes = G.number_of_nodes() , relabel_nodes=True)
        num_nodes = subgrap_node_old_index.shape[0]
        new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32) , create_using=type(G))
        new_G.add_nodes_from(np.arange(num_nodes , dtype=np.int32))

        x_list = []
        attributes = G.graph["attributes"]
        if attributes is not None:
            new_attributes = torch.tensor(attributes , dtype=torch.float32)[subgrap_node_old_index]
            if new_attributes.dim() < 2:
                new_attributes.unsqueeze_(1)
            x_list.append(new_attributes)
        features_sp = torch.from_numpy(get_features_sp(new_G , np.array(new_set_index))).float()
        x_list.append(features_sp)

        x = torch.cat(x_list , dim=-1)
        if labels is not None:
            y = torch.tensor([labels[i]] , dtype=torch.long)
        else:
            y = torch.tensor([0] , dtype=torch.long)
        new_set_index = new_set_index.long().unsqueeze(0)
        data_list.append(Data(x=x , edge_index=new_edge_index , y=y , set_indices=new_edge_index))
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
    train_set = [data_list[x] for x in train_mask if x]
    val_set   = [data_list[x] for x in val_mask   if x]
    test_set  = [data_list[x] for x in test_mask  if x]
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader , val_loader , test_loader

def get_data(G , labels , size = 1.0):
    G = deepcopy(G)
    set_indices = list(range(len(labels)))
    random.shuffle(set_indices)
    labels = labels[set_indices]
    set_indices = np.expand_dims(set_indices , 1) # shape num_nodes * 1
    
    data_list = extract_subgaphs(G , labels , set_indices)

    train_indices , val_test_indices = train_test_split(list(range(set_indices.shape[0])) , test_size=test_ratio*2 , stratify=labels)
    val_test_labels = np.array([data_list[i].y for i in val_test_indices])
    # np.array([data.y for data in data_list[val_test_indices]]) 
    val_indices , test_indices = train_test_split(val_test_indices , test_size=0.5 , stratify=val_test_labels)
    train_mask , val_mask , test_mask = [np.zeros(set_indices.shape[0] , dtype=np.uint8) for i in range(3)]
    train_mask[train_indices] = 1
    val_mask[val_indices] = 1
    test_mask[test_indices] = 1
    train_loader , val_loader , test_loader = load_datasets(data_list , train_mask , val_mask , test_mask)
    return train_loader , val_loader , test_loader , len(np.unique(labels))

# def get_model(in_features , out_feature , layers = 1 , prop_depth = 1 , )