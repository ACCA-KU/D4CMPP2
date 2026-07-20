
"""
This codes are modified from the project "GC-GNN" (https://github.com/gsi-lab/GC-GNN)
The original codes are under the MIT License. (https://github.com/gsi-lab/GC-GNN/blob/main/networks/AttentiveFP.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from D4CMPP2.networks.src.GCN import graph_sum_pool


def _graph_softmax(values, batch, num_graphs):
    maximum = values.new_full((num_graphs, values.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, batch[:, None].expand_as(values), values, reduce='amax', include_self=True)
    exponentials = torch.exp(values - maximum[batch])
    denominator = values.new_zeros((num_graphs, values.shape[1]))
    denominator.index_add_(0, batch, exponentials)
    return exponentials / denominator[batch]

class Atom_AFPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim, bias=True),
            nn.Dropout(config['dropout']),
            nn.LeakyReLU()
        )
        self.cal_alignment = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, 1, bias=True),
        )
        self.attend = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Dropout(config['dropout']),
        )
        self.GRUCell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_edge_lin[0].reset_parameters()
        self.cal_alignment[0].reset_parameters()
        self.attend[0].reset_parameters()
        self.GRUCell.reset_parameters()

    def forward(self, graph, node, edge):
        src, dst = graph.edge_index
        neighbor_message = self.embedding_edge_lin(torch.cat([node[src], edge], dim=-1))
        # Retained for state-dict and numerical compatibility although the legacy path did not apply the score.
        self.cal_alignment(torch.cat([neighbor_message, node[dst]], dim=-1))
        messages = self.dropout(self.attend(neighbor_message))
        context = messages.new_zeros(node.shape)
        context.index_add_(0, dst, messages)
        new_node = node + context
        return new_node

class Mol_AFPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.GRUCell = nn.GRUCell(config['hidden_dim'], config['hidden_dim'])

        self.cal_alignment = nn.Sequential(
            nn.Linear(config['hidden_dim'] + config['hidden_dim'], 1, bias=True),
            nn.LeakyReLU()
        )
        self.attend = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=True),
            nn.Dropout(config['dropout']),
        )
        self.dropout = nn.Dropout(config['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        self.GRUCell.reset_parameters()
        self.cal_alignment[0].reset_parameters()
        self.attend[0].reset_parameters()

    def forward(self, graph, super_node, node):
        super_node = F.leaky_relu(super_node)
        batch = getattr(graph, 'batch', node.new_zeros(node.shape[0], dtype=torch.long))
        num_graphs = int(getattr(graph, 'num_graphs', 1))
        score = self.cal_alignment(torch.cat([node, super_node[batch]], dim=1))
        attention_weight = _graph_softmax(score, batch, num_graphs)
        hidden_node = self.attend(self.dropout(node)) * attention_weight
        super_context = hidden_node.new_zeros((num_graphs, hidden_node.shape[1]))
        super_context.index_add_(0, batch, hidden_node)
        super_context = F.elu(super_context)
        
        super_node = F.relu(self.GRUCell(super_node, super_context))
        return super_node, attention_weight


class AttentiveFP(nn.Module):
    # Generate Context of each nodes
    def __init__(self, config):
        super().__init__()
        conv_layers = config.get('conv_layers', 4)
        T = config.get("T", conv_layers)
        self.PassingDepth = nn.ModuleList([Atom_AFPLayer(config) for _ in range(conv_layers)])
        self.MultiTimeSteps = nn.ModuleList([Mol_AFPLayer(config) for d in range(T)])
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.MultiTimeSteps:
            l.reset_parameters()

    def forward(self, graph, node, edge, **kwargs):
        for i in range(len(self.PassingDepth)):
            node = self.PassingDepth[i](graph, node, edge)
        if kwargs.get('only_atom', False):
            return node, None
        super_node = graph_sum_pool(graph, node)
        for i in range(len(self.MultiTimeSteps)):
            super_node, att_w = self.MultiTimeSteps[i](graph, super_node, node)
        return super_node, att_w


