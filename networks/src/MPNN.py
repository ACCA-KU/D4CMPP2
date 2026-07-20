import torch.nn as nn
import torch


class MPNN_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, activation, dropout=0.2, batch_norm=False, residual_sum = False):
        super(MPNN_layer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.linear = nn.Linear(in_node_feats*2+in_edge_feats, out_feats)
        self.residual_sum = residual_sum
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
        if residual_sum:
            if in_node_feats!=out_feats:
                self.residual_layer = nn.Linear(in_node_feats, out_feats)

    def forward(self, graph, node_feats, edge_feats):
        src, dst = graph.edge_index
        messages = self.linear(torch.cat([node_feats[src], node_feats[dst], edge_feats], dim=1))
        h = messages.new_zeros((node_feats.shape[0], messages.shape[1]))
        h.index_add_(0, dst, messages)
        degree = messages.new_zeros((node_feats.shape[0], 1))
        degree.index_add_(0, dst, messages.new_ones((messages.shape[0], 1)))
        h = h / degree.clamp_min(1)
        if self.batch_norm:
            h = self.bn(h)
        h = self.activation(h)
        h = self.dropout(h)
        if self.residual_sum:
            if node_feats.shape[1]!=h.shape[1]:
                node_feats = self.residual_layer(node_feats)
            h = h + node_feats
        return h
    
class MPNNs(nn.Module):
    def __init__(self,  in_node_feats, in_edge_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        super(MPNNs, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i==0:
                _in_feats = in_node_feats
            else:
                _in_feats = hidden_feats
            if i==n_layers-1:
                _out_feats = out_feats
            else:
                _out_feats = hidden_feats
            self.layers.append(MPNN_layer(_in_feats, in_edge_feats, _out_feats, activation, dropout, batch_norm, residual_sum))
        
    def forward(self, graph, node_feats, edge_feats):
        for layer in self.layers:
            node_feats = layer(graph, node_feats, edge_feats)
        return node_feats
