import torch.nn as nn
import torch


class GCN_layer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout=0.2, batch_norm=False, residual_sum = False):
        super(GCN_layer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.linear = nn.Linear(in_feats, out_feats)
        self.residual_sum = residual_sum
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
        if residual_sum:
            if in_feats!=out_feats:
                self.residual_layer = nn.Linear(in_feats, out_feats)

    def forward(self, graph, node_feats):
        h = self.linear(node_feats)
        src, dst = graph.edge_index
        aggregated = torch.zeros_like(h)
        if src.numel() > 0:
            aggregated.index_add_(0, dst, h[src])
        # The established contract adds one self-message per node without removing
        # pre-existing self edges. Existing self edges are already aggregated above.
        h = aggregated + h
        if self.batch_norm:
            h = self.bn(h)
        h = self.activation(h)
        h = self.dropout(h)
        if self.residual_sum:
            if node_feats.shape[1]!=h.shape[1]:
                node_feats = self.residual_layer(node_feats)
            h = h + node_feats
        return h
    
class GCNs(nn.Module):
    def __init__(self,  in_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        super(GCNs, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i==0:
                _in_feats = in_feats
            else:
                _in_feats = hidden_feats
            if i==n_layers-1:
                _out_feats = out_feats
            else:
                _out_feats = hidden_feats
            self.layers.append(GCN_layer(_in_feats, _out_feats, activation, dropout, batch_norm, residual_sum))
        
    def forward(self, graph, node_feats):
        for layer in self.layers:
            node_feats = layer(graph, node_feats)
        return node_feats


def graph_sum_pool(graph, node_feats):
    """Sum node features per PyG graph using the established readout contract."""
    if hasattr(graph, "batch") and graph.batch is not None:
        batch = graph.batch
        num_graphs = int(graph.num_graphs)
    else:
        batch = torch.zeros(node_feats.shape[0], dtype=torch.long, device=node_feats.device)
        num_graphs = 1
    pooled = node_feats.new_zeros((num_graphs, node_feats.shape[-1]))
    if batch.numel() > 0:
        pooled.index_add_(0, batch, node_feats)
    return pooled
