from param import *
layer_dict = {"DE-GNN": GATConv , "GIN" : GINConv , "GCN" : GCNConv , "GraphSAGE" :SAGEConv , "GAT" : GATConv}
class GNNModel(nn.Module):
   def __init__(self , in_features , hidden_features ,  out_features , model_name , layers = 1 , prop_depth = 1 , dropout = 0.0):
      super(GNNModel , self).__init__()
      self.in_features , self.hidden_features , self.out_features , self.model_name , self.layers , self.prop_depth = in_features , hidden_features , out_features , model_name , layers , prop_depth
      self.act = nn.ReLU()
      self.dropout = nn.Dropout(p=dropout)
      self.layers = nn.ModuleList()
      self.layers.append(GATConv(in_channels=in_features , out_channels=hidden_features , K = prop_depth))
      self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for i in range(layers)])
      self.merger = nn.Linear(3 * hidden_features , hidden_features)
      self.feed_forward = FeedForwardNetwork(hidden_features, out_features)

   def forward(self , batch):
      x = batch.x
      edge_index = batch.edge_index
      for i , layer in enumerate(self.layers):
         x = layer(x , edge_index)
         x = self.act(x)
         x = self.dropout(x)
         x = self.layer_norms[i](x)
      x = self.get_minibatch_embeddings(x , batch)
      x = self.feed_forward(x)
      return x

   def get_minibatch_embeddings(self, x, batch):
        set_indices , batch , num_graphs = batch.set_indices, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs)[batch].sum(dim=0)
        # 计算每个图中的节点个数，num_nodes shape = [1 * num_graphs]
        zero = torch.tensor([0], dtype=torch.long)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
        # 把num_nodes的最后一项去掉，然后最前面加个零
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
      #   assert(index_bases.size(0) == set_indices.size(0))
        set_indices_batch = index_bases + set_indices
        # print('set_indices shape', set_indices.shape, 'index_bases shape', index_bases.shape, 'x shape:', x.shape)
        x = x[set_indices_batch]  # shape [B, set_size, F]
        x = self.pool(x)
        return x
   def pool(self, x):
      if x.size(1) == 1:
         return torch.squeeze(x, dim=1)
      # use mean/diff/max to pool each set's representations
      x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
      for i, j in combinations(range(x.size(1)), 2):
         x_diff += torch.abs(x[:, i, :]-x[:, j, :])
      x_mean = x.mean(dim=1)
      x_max = x.max(dim=1)[0]
      x = self.merger(torch.cat([x_diff, x_mean, x_max], dim=-1))
      return x

class FeedForwardNetwork(nn.Module):
   def __init__(self , in_features , out_features , act=nn.ReLU() , dropout=0.0):
      super(FeedForwardNetwork, self).__init__()
      self.act = act
      self.dropout = nn.Dropout(dropout)
      self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
      self.layer2 = nn.Linear(in_features, out_features)

   def forward(self, inputs):
      x = self.layer1(inputs)
      x = self.layer2(x)
      return x