import torch.nn as nn
from D4CMPP2.networks.base import (
    InputContract,
    MolecularNetwork,
    STANDARD_SOLVENT_HYPERPARAMETERS,
    STANDARD_SOLVENT_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.GAT import GATs
from D4CMPP2.networks.src.GCN import graph_sum_pool
from D4CMPP2.networks.src.SolventLayer import SolventLayer

class SolventGAT(MolecularNetwork):
    """Graph-attention compound encoder coupled to a solvent branch."""

    model_name = "gat_solvent"
    input_contract = InputContract(
        required=(
            "compound_graphs",
            "compound_node_feature",
            "solvent_graphs",
            "solvent_node_feature",
        )
    )
    hyperparameters = STANDARD_SOLVENT_HYPERPARAMETERS
    default_optimization_space = STANDARD_SOLVENT_OPTIMIZATION_SPACE

    def __init__(self, config):
        super().__init__(config)
        
        hidden_dim = self.config["hidden_dim"]
        gcn_layers = self.config["conv_layers"]
        dropout = self.config["dropout"]


        self.node_embedding = nn.Linear(self.config["node_dim"], hidden_dim)

        self.GATs = GATs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), gcn_layers, dropout, False, True)
        self.SolventLayer = SolventLayer(self.config)

    def forward(self, **kwargs):
        graph = kwargs.get('compound_graphs', kwargs.get('graph'))
        node_feats = kwargs.get('compound_node_feature', kwargs.get('node_feats'))
        solv_graph = kwargs.get('solvent_graphs', kwargs.get('solv_graph'))
        solv_node_feats = kwargs.get('solvent_node_feature', kwargs.get('solv_node_feats'))
        values = {
            "compound_graphs": graph,
            "compound_node_feature": node_feats,
            "solvent_graphs": solv_graph,
            "solvent_node_feature": solv_node_feats,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(
                f"SolventGAT input is missing required fields {missing!r}."
            )

        h = self.node_embedding(node_feats)
        h = self.GATs(graph, h)
        h = graph_sum_pool(graph, h)
        h = self.SolventLayer(h,solv_graph,solv_node_feats)
        return h

network = SolventGAT
