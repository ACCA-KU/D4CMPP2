import torch.nn as nn


class distGCN_layer(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats, activation, alpha=0.1):
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats)
        self.edge_linear = nn.Linear(edge_feats, out_feats)
        self.alpha = alpha

    def forward(self, graph, node_feats, edge_feats):
        if len(edge_feats) == 0:
            return node_feats
        src, dst = graph.edge_index
        transformed_nodes = self.linear(node_feats)
        transformed_edges = self.edge_linear(edge_feats)
        messages = transformed_nodes[src] * transformed_edges
        aggregated = messages.new_zeros((node_feats.shape[0], messages.shape[1]))
        aggregated.index_add_(0, dst, messages)
        return node_feats + self.alpha * self.activation(aggregated)
