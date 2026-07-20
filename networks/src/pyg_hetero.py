from types import SimpleNamespace

import torch


def relation_graph(graph, node_type, relation):
    """Expose one homogeneous HeteroData relation to the existing PyG layers."""
    store = graph[node_type]
    batch = getattr(store, 'batch', None)
    if batch is None:
        batch = torch.zeros(store.num_nodes, dtype=torch.long, device=store.x.device)
    return SimpleNamespace(
        edge_index=graph[node_type, relation, node_type].edge_index,
        batch=batch,
        num_graphs=int(getattr(graph, 'num_graphs', 1)),
    )


def relation_sum(graph, source_type, relation, destination_type, source_features):
    """Sum source features at each destination node for one typed relation."""
    src, dst = graph[source_type, relation, destination_type].edge_index
    output = source_features.new_zeros((graph[destination_type].num_nodes, source_features.shape[1]))
    output.index_add_(0, dst, source_features[src])
    return output
