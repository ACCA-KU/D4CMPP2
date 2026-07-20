
from D4CMPP2.networks.src.GCN import GCNs, graph_sum_pool
from D4CMPP2.networks.src.Linear import Linears
import torch.nn as nn
import torch

class SolventLayer(nn.Module):
    def __init__(self, config):
        super(SolventLayer, self).__init__()
        
        hidden_dim = config["hidden_dim"]
        linear_layers = config["linear_layers"]
        solvent_hidden_dim = config.get("solvent_hidden_dim", 64)
        solvent_conv_layers = config.get("solvent_conv_layers", 4)
        solvent_linear_layers = config.get("solvent_linear_layers", 2)
        solvent_dropout = config.get("solvent_dropout", 0.2)
        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)
        self.node_embedding_solv = nn.Linear(
            config["node_dim"], solvent_hidden_dim
        )

        self.GCNs_solv = GCNs(
            solvent_hidden_dim,
            solvent_hidden_dim,
            solvent_hidden_dim,
            nn.ReLU(),
            solvent_conv_layers,
            solvent_dropout,
            False,
            True,
        )
        self.Linears2 = Linears(
            solvent_hidden_dim,
            solvent_hidden_dim,
            nn.ReLU(),
            solvent_linear_layers,
            solvent_dropout,
            False,
            False,
        )
        self.Linears3 = Linears(
            hidden_dim + solvent_hidden_dim,
            config["target_dim"],
            nn.ReLU(),
            linear_layers,
            config.get("dropout", 0.2),
            False,
            False,
            True,
        )

    def forward(self, hidden_feats,solv_graph,solv_node_feats,**kwargs):
        h = hidden_feats

        h_solv = self.node_embedding_solv(solv_node_feats)
        h_solv = self.GCNs_solv(solv_graph, h_solv)
        h_solv = graph_sum_pool(solv_graph, h_solv)

        h_solv = self.Linears2(h_solv)
        h = torch.cat([h,h_solv],axis=1)
        h = self.Linears3(h)
        return h
