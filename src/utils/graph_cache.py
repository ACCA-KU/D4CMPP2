"""Versioned graph-cache identity, validation, and atomic persistence."""

import hashlib
import importlib.metadata
import json
import os
import re
import uuid
from pathlib import Path

import torch
from torch_geometric.data import Data, HeteroData


GRAPH_CACHE_SCHEMA_VERSION = 2
GRAPH_FEATURE_CONTRACT_VERSION = 1
ISA_RELATIONS = (
    ("r_nd", "r2r", "r_nd"),
    ("r_nd", "r2i", "i_nd"),
    ("i_nd", "i2i", "i_nd"),
    ("i_nd", "i2d", "d_nd"),
    ("d_nd", "d2d", "d_nd"),
    ("d_nd", "d2r", "r_nd"),
)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_sha256(path):
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_smiles_sha256(smiles):
    encoded = json.dumps([str(value) for value in smiles], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def safe_cache_component(value):
    name = Path(os.fspath(value).replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "data"


def build_graph_recipe(manager, column):
    """Return deterministic identity metadata for one molecule-column cache."""
    is_isa = str(manager.graph_type).startswith("img")
    functional_group = None
    if is_isa:
        candidate = Path(manager.config.get("MODEL_PATH", "")) / "functional_group.csv"
        functional_group = candidate if candidate.is_file() else manager.config.get("FRAG_REF")
    recipe = {
        "graph_backend": "pyg",
        "graph_schema_version": GRAPH_CACHE_SCHEMA_VERSION,
        "feature_contract_version": GRAPH_FEATURE_CONTRACT_VERSION,
        "graph_type": manager.graph_type,
        "molecule_column": column,
        "explicit_h": column in manager.explicit_h_columns,
        "row_count": len(manager._molecule_smiles[column]),
        "ordered_smiles_sha256": _ordered_smiles_sha256(manager._molecule_smiles[column]),
        "node_dim": manager.gg.node_dim,
        "edge_dim": manager.gg.edge_dim,
        "max_dist": manager.config.get("max_dist", 4) if is_isa else None,
        "functional_group_sha256": _file_sha256(functional_group) if is_isa else None,
        "versions": {
            "d4cmpp2": _package_version("D4CMPP2"),
            "rdkit": _package_version("rdkit"),
            "torch": torch.__version__,
            "torch_geometric": _package_version("torch-geometric"),
        },
    }
    identity = dict(recipe)
    identity.pop("versions")
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    recipe["fingerprint"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return recipe


def cache_path(manager, column, recipe):
    parts = [
        safe_cache_component(manager.data),
        safe_cache_component(column),
        safe_cache_component(manager.graph_type),
    ]
    if column in manager.explicit_h_columns:
        parts.append("explicitH")
    parts.extend(["pyg", "v2", recipe["fingerprint"][:16]])
    return Path(manager.config["GRAPH_DIR"]) / ("_".join(parts) + ".pt")


def legacy_cache_paths(manager, column):
    base = f"{manager.data}_{column}_{manager.graph_type}"
    if column in manager.explicit_h_columns:
        base += "_explicitH"
    root = Path(manager.config["GRAPH_DIR"])
    return root / f"{base}_pyg_v1.pt", root / f"{base}.bin"


def atomic_save_graph_cache(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, staging)
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def _validate_tensor(tensor, name, expected_rows=None, expected_columns=None, floating=None):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if expected_rows is not None and tensor.shape[0] != expected_rows:
        raise ValueError(f"{name} rows={tensor.shape[0]}, expected {expected_rows}.")
    if expected_columns is not None and (tensor.ndim != 2 or tensor.shape[1] != expected_columns):
        raise ValueError(f"{name} shape={tuple(tensor.shape)}, expected (*, {expected_columns}).")
    if floating is True and not tensor.is_floating_point():
        raise ValueError(f"{name} dtype={tensor.dtype}, expected a floating dtype.")
    if tensor.is_floating_point() and tensor.numel() and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinite values.")


def _validate_edge_alignment(edge_index, edge_attr, name):
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"{name} edge feature rows={edge_attr.shape[0]}, "
            f"but edge_index contains {edge_index.shape[1]} edges."
        )


def validate_graph(graph, recipe, index):
    prefix = f"graphs[{index}]"
    if str(recipe["graph_type"]).startswith("img"):
        if not isinstance(graph, HeteroData):
            raise ValueError(f"{prefix} must be HeteroData, got {type(graph).__name__}.")
        missing_nodes = [node for node in ("r_nd", "i_nd", "d_nd") if node not in graph.node_types]
        missing_relations = [relation for relation in ISA_RELATIONS if relation not in graph.edge_types]
        if missing_nodes or missing_relations:
            raise ValueError(
                f"{prefix} is missing node types {missing_nodes} or relations {missing_relations}."
            )
        _validate_tensor(graph["r_nd"].x, f"{prefix}.r_nd.x", expected_columns=recipe["node_dim"], floating=True)
        for relation in ISA_RELATIONS:
            _validate_tensor(graph[relation].edge_index, f"{prefix}.{relation}.edge_index", expected_rows=2)
        r2r = graph["r_nd", "r2r", "r_nd"]
        _validate_tensor(
            r2r.edge_attr,
            f"{prefix}.r2r.edge_attr",
            expected_columns=recipe["edge_dim"],
            floating=True,
        )
        _validate_edge_alignment(r2r.edge_index, r2r.edge_attr, f"{prefix}.r2r")
        d2d = graph["d_nd", "d2d", "d_nd"]
        _validate_tensor(
            d2d.edge_attr,
            f"{prefix}.d2d.edge_attr",
            expected_columns=recipe["max_dist"],
            floating=True,
        )
        _validate_edge_alignment(d2d.edge_index, d2d.edge_attr, f"{prefix}.d2d")
        return
    if not isinstance(graph, Data) or isinstance(graph, HeteroData):
        raise ValueError(f"{prefix} must be Data, got {type(graph).__name__}.")
    _validate_tensor(graph.x, f"{prefix}.x", expected_columns=recipe["node_dim"], floating=True)
    _validate_tensor(graph.edge_index, f"{prefix}.edge_index", expected_rows=2)
    if graph.edge_index.dtype != torch.long:
        raise ValueError(f"{prefix}.edge_index dtype={graph.edge_index.dtype}, expected torch.int64.")
    _validate_tensor(graph.edge_attr, f"{prefix}.edge_attr", expected_columns=recipe["edge_dim"], floating=True)
    _validate_edge_alignment(graph.edge_index, graph.edge_attr, prefix)


def validate_payload(
    payload,
    expected_recipe,
    expected_smiles,
    path,
    *,
    validate_graph_tensors=True,
):
    if not isinstance(payload, dict):
        raise ValueError(f"Graph cache {str(path)!r} must contain a dictionary payload.")
    actual_recipe = payload.get("recipe")
    if not isinstance(actual_recipe, dict):
        raise ValueError(f"Graph cache {str(path)!r} has no v2 recipe metadata.")
    for field, expected in expected_recipe.items():
        if field == "versions":
            continue
        actual = actual_recipe.get(field)
        if actual != expected:
            raise ValueError(
                f"Graph cache {str(path)!r} recipe mismatch for {field!r}: "
                f"actual={actual!r}, expected={expected!r}. Regenerate this cache from the source CSV."
            )
    smiles = payload.get("smiles")
    if [str(value) for value in smiles or []] != [str(value) for value in expected_smiles]:
        raise ValueError(
            f"Graph cache {str(path)!r} ordered SMILES do not match the source CSV. "
            "Regenerate this cache from the current data."
        )
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != len(expected_smiles):
        raise ValueError(
            f"Graph cache {str(path)!r} graph count is "
            f"{len(graphs) if isinstance(graphs, list) else type(graphs).__name__}, "
            f"expected {len(expected_smiles)}. Regenerate this cache."
        )
    if validate_graph_tensors:
        for index, graph in enumerate(graphs):
            validate_graph(graph, expected_recipe, index)
    return graphs, list(payload.get("graph_errors") or [])
