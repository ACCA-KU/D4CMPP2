import torch.nn as nn
import torch

class GAT_layer(nn.Module):
    def __init__(self, in_node_feats, hidden_feats, out_feats, activation, dropout=0.2, batch_norm=False, residual_sum = False):
        super(GAT_layer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.attention_W = nn.Linear(in_node_feats*2, hidden_feats)
        self.attention_a = nn.Linear(hidden_feats,1, bias=False)
        self.linear = nn.Linear(in_node_feats, out_feats)

        self.batch_norm = batch_norm
        self.residual_sum = residual_sum
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
        if residual_sum:
            if in_node_feats!=out_feats:
                self.residual_layer = nn.Linear(in_node_feats, out_feats)

    def forward(self, graph, node_feats):
        src, dst = graph.edge_index
        logits = self.attention_a(
            nn.LeakyReLU()(self.attention_W(torch.cat([node_feats[src], node_feats[dst]], dim=1)))
        )
        max_per_dst = logits.new_full((node_feats.shape[0], logits.shape[1]), -torch.inf)
        max_per_dst.scatter_reduce_(
            0, dst[:, None].expand_as(logits), logits, reduce='amax', include_self=True
        )
        exponentials = torch.exp(logits - max_per_dst[dst])
        denominator = logits.new_zeros((node_feats.shape[0], logits.shape[1]))
        denominator.index_add_(0, dst, exponentials)
        scores = exponentials / denominator[dst]

        messages = scores * self.linear(node_feats[src])
        h = messages.new_zeros((node_feats.shape[0], messages.shape[1]))
        h.index_add_(0, dst, messages)
        h = nn.LeakyReLU()(h)
        if self.batch_norm:
            h = self.bn(h)
        h = self.activation(h)
        h = self.dropout(h)
        if self.residual_sum:
            if node_feats.shape[1]!=h.shape[1]:
                node_feats = self.residual_layer(node_feats)
            h = h + node_feats
        return h
    
class GATs(nn.Module):
    def __init__(self,  in_node_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        super(GATs, self).__init__()

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
            self.layers.append(GAT_layer(_in_feats, hidden_feats, _out_feats, activation, dropout, batch_norm, residual_sum))
        
    def forward(self, graph, node_feats):
        for layer in self.layers:
            node_feats = layer(graph, node_feats)
        return node_feats
