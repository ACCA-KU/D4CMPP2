import torch.nn as nn
import torch
from D4CMPP2.networks.base import (
    Hyperparameter,
    InputContract,
    MolecularNetwork,
    STANDARD_SOLVENT_HYPERPARAMETERS,
    STANDARD_SOLVENT_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.GCN import GCNs, graph_sum_pool
from D4CMPP2.networks.src.Linear import Linears

class SolventGCN(MolecularNetwork):
    """Two-branch GCN for compound-solvent property prediction."""

    model_name = "gcn_solvent"
    input_contract = InputContract(
        required=(
            "compound_graphs",
            "compound_node_feature",
            "solvent_graphs",
            "solvent_node_feature",
        )
    )
    hyperparameters = {
        **STANDARD_SOLVENT_HYPERPARAMETERS,
        "fusion_dim": Hyperparameter(
            "int", default=16, low=1, search_low=8, search_high=128,
            step=8, grid=(16, 32, 64),
            description="Width of the fused compound-solvent representation.",
        ),
    }
    default_optimization_space = (
        *STANDARD_SOLVENT_OPTIMIZATION_SPACE,
        "fusion_dim",
    )

    def __init__(self, config):
        super().__init__(config)
        
        hidden_dim = self.config["hidden_dim"]
        conv_layers = self.config["conv_layers"]
        linear_layers = self.config["linear_layers"]
        dropout = self.config["dropout"]
        solvent_hidden_dim = self.config["solvent_hidden_dim"]
        solvent_conv_layers = self.config["solvent_conv_layers"]
        solvent_linear_layers = self.config["solvent_linear_layers"]
        solvent_dropout = self.config["solvent_dropout"]
        fusion_dim = self.config["fusion_dim"]
        self.node_embedding = nn.Linear(self.config["node_dim"], hidden_dim)
        self.node_embedding_solv = nn.Linear(
            self.config["node_dim"], solvent_hidden_dim
        )

        self.GCNs = GCNs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), conv_layers, dropout, False, True)
        self.GCNs_solv = GCNs(solvent_hidden_dim, solvent_hidden_dim, solvent_hidden_dim, nn.ReLU(), solvent_conv_layers, solvent_dropout, False, True)
        self.Linears1 = Linears(hidden_dim,hidden_dim, nn.ReLU(), 2, dropout, False, False)
        self.Linears2 = Linears(solvent_hidden_dim,solvent_hidden_dim, nn.ReLU(), solvent_linear_layers, solvent_dropout, False, False)
        self.Linears3 = Linears(hidden_dim+solvent_hidden_dim,fusion_dim, nn.ReLU(), linear_layers, dropout, False, False)

        self.output = nn.Linear(fusion_dim, self.config["target_dim"])

    def forward(self, **kwargs):
        batch = {
            "compound_graphs": kwargs.get(
                "compound_graphs", kwargs.get("graph")
            ),
            "compound_node_feature": kwargs.get(
                "compound_node_feature", kwargs.get("node_feats")
            ),
            "solvent_graphs": kwargs.get(
                "solvent_graphs", kwargs.get("solv_graph")
            ),
            "solvent_node_feature": kwargs.get(
                "solvent_node_feature", kwargs.get("solv_node_feats")
            ),
        }
        missing = [name for name, value in batch.items() if value is None]
        if missing:
            raise ValueError(
                f"SolventGCN input is missing required fields {missing!r}."
            )
        graph = batch["compound_graphs"]
        node_feats = batch["compound_node_feature"]
        solv_graph = batch["solvent_graphs"]
        solv_node_feats = batch["solvent_node_feature"]

        h = self.node_embedding(node_feats)
        h = self.GCNs(graph, h)
        h = graph_sum_pool(graph, h)

        h_solv = self.node_embedding_solv(solv_node_feats)
        h_solv = self.GCNs_solv(solv_graph, h_solv)
        h_solv = graph_sum_pool(solv_graph, h_solv)

        h = self.Linears1(h)
        h_solv = self.Linears2(h_solv)
        h = torch.cat([h,h_solv],axis=1)
        h = self.Linears3(h)
        h = self.output(h)
        return h

network = SolventGCN
