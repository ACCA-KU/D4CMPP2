import torch.nn as nn

from D4CMPP2.networks.base import (
    InputContract,
    MolecularNetwork,
    STANDARD_SOLVENT_HYPERPARAMETERS,
    STANDARD_SOLVENT_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.AFP import  AttentiveFP
from D4CMPP2.networks.src.SolventLayer import SolventLayer

class SolventAFP(MolecularNetwork):
    """Attentive fingerprint compound encoder with a solvent branch."""

    model_name = "afp_solvent"
    required_config = ("node_dim", "edge_dim", "target_dim")
    input_contract = InputContract(
        required=(
            "compound_graphs",
            "compound_node_feature",
            "compound_edge_feature",
            "solvent_graphs",
            "solvent_node_feature",
        )
    )
    hyperparameters = STANDARD_SOLVENT_HYPERPARAMETERS
    default_optimization_space = STANDARD_SOLVENT_OPTIMIZATION_SPACE

    def __init__(self, config):
        super().__init__(config)

        hidden_dim = self.config["hidden_dim"]

        self.embedding_node_lin = nn.Linear(self.config["node_dim"], hidden_dim, bias=True)
        self.embedding_edge_lin = nn.Linear(self.config["edge_dim"], hidden_dim, bias=True)

        self.AttentiveFP = AttentiveFP(self.config)
        self.Linears = SolventLayer(self.config) 


    def forward(self, **kwargs):
        graph = kwargs.get('compound_graphs', kwargs.get('graph'))
        node_feats = kwargs.get('compound_node_feature', kwargs.get('node_feats'))
        edge_feats = kwargs.get('compound_edge_feature', kwargs.get('edge_feats'))
        solv_graph = kwargs.get('solvent_graphs', kwargs.get('solv_graph'))
        solv_node_feats = kwargs.get('solvent_node_feature', kwargs.get('solv_node_feats'))
        missing = [
            name
            for name, value in {
                "compound_graphs": graph,
                "compound_node_feature": node_feats,
                "compound_edge_feature": edge_feats,
                "solvent_graphs": solv_graph,
                "solvent_node_feature": solv_node_feats,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                f"SolventAFP input is missing required fields {missing!r}."
            )

        node = self.embedding_node_lin(node_feats)
        edge = self.embedding_edge_lin(edge_feats)

        super_node, att_w = self.AttentiveFP(graph, node, edge)

        output = self.Linears(super_node, solv_graph, solv_node_feats)
        return output


network = SolventAFP
