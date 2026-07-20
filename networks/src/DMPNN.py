
"""
This codes are modified from the project "GC-GNN" (https://github.com/gsi-lab/GC-GNN)
The original codes are under the MIT License. (https://github.com/gsi-lab/GC-GNN/blob/main/networks/DMPNN.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DMPNNLayer(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats, activation, dropout=0.2):
        super(DMPNNLayer, self).__init__()
        self.activation = activation

        self.W_m = nn.Linear(in_feats+edge_feats, out_feats, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    
    def forward(self, graph, node_feats, edge_feats, direct_feats=None, backward_feats=None):
        src, dst = graph.edge_index
        if backward_feats is not None and direct_feats is not None:
            direct = self.W_m(torch.cat([edge_feats, direct_feats], 1))
            backward = self.W_m(torch.cat([edge_feats, backward_feats], 1))
        else:
            direct = self.W_m(torch.cat([edge_feats, node_feats[src]], 1))
            backward = self.W_m(torch.cat([edge_feats, node_feats[dst]], 1))
        full_feats = direct.new_zeros((node_feats.shape[0], direct.shape[1]))
        full_feats.index_add_(0, dst, direct)
        new_edge_feats = full_feats[src] - backward
        new_backward_feats = full_feats[dst] - direct
        new_node_feats = full_feats
        if self.activation is not None:
            new_node_feats = self.activation(new_node_feats)
        new_node_feats = self.dropout_layer(new_node_feats)
        return new_node_feats, new_edge_feats, new_backward_feats
