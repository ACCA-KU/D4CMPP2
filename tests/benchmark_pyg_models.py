"""Small, deterministic forward/backward timing probe for all supported PyG IDs."""

import importlib
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from D4CMPP2.src.DataManager.GraphGenerator.ISAGraphGenerator import ISAGraphGenerator
from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator


MODELS = {
    "GCN": "GCN_model", "GCNwS": "GCNwithSolv_model",
    "MPNN": "MPNN_model", "MPNNwS": "MPNNwithSolv_model",
    "DMPNN": "DMPNN_model", "DMPNNwS": "DMPNNwithSolv_model",
    "AFP": "AFP_model", "AFPwS": "AFPwithSolv_model",
    "GAT": "GAT_model", "GATwS": "GATwithSolv_model",
    "GC": "GC_model", "ISAT": "ISAT_model", "ISATPN": "ISATPN_model",
}


def fixtures(device):
    molecule = MolGraphGenerator()
    compound = Batch.from_data_list([molecule.get_graph(s) for s in ("CCOCC", "CCNCC")]).to(device)
    solvent = Batch.from_data_list([molecule.get_graph(s) for s in ("O", "CO")]).to(device)
    common = {
        "compound_graphs": compound, "compound_node_feature": compound.x,
        "compound_edge_feature": compound.edge_attr, "solvent_graphs": solvent,
        "solvent_node_feature": solvent.x,
    }
    isa_generator = ISAGraphGenerator(
        frag_ref=str(ROOT / "src" / "utils" / "functional_group.csv"), sculptor_index=(6, 2, 0)
    )
    isa = Batch.from_data_list([isa_generator.get_graph(s, max_dist=4) for s in ("CCOCC", "CCNCC")]).to(device)
    isa_kwargs = {
        "compound_graphs": isa, "compound_r_node": isa["r_nd"].x,
        "compound_r2r_edge": isa["r_nd", "r2r", "r_nd"].edge_attr,
        "compound_i_node": isa["i_nd"].x, "compound_d_node": isa["d_nd"].x,
        "compound_d2d_edge": isa["d_nd", "d2d", "d_nd"].edge_attr,
    }
    config = {"node_dim": molecule.node_dim, "edge_dim": molecule.edge_dim, "target_dim": 1,
              "hidden_dim": 8, "conv_layers": 1, "linear_layers": 1, "dropout": 0.0}
    isa_config = dict(config, node_dim=isa_generator.node_dim, edge_dim=isa_generator.edge_dim)
    return common, isa_kwargs, config, isa_config


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available")
    common, isa_kwargs, config, isa_config = fixtures(device)
    results = {}
    for model_id, module_name in MODELS.items():
        torch.manual_seed(42)
        model = importlib.import_module(f"D4CMPP2.networks.{module_name}").network(
            dict(isa_config if model_id in {"GC", "ISAT", "ISATPN"} else config)
        ).to(device).eval()
        kwargs = isa_kwargs if model_id in {"GC", "ISAT", "ISATPN"} else dict(common)
        if not model_id.endswith("wS"):
            kwargs = {key: value for key, value in kwargs.items() if not key.startswith("solvent_")}
        durations = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(8):
            model.zero_grad(set_to_none=True)
            synchronize(device)
            start = time.perf_counter()
            output = model(**kwargs)
            output.sum().backward()
            synchronize(device)
            if iteration >= 2:
                durations.append((time.perf_counter() - start) * 1000)
        results[model_id] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "median_forward_backward_ms": round(statistics.median(durations), 3),
        }
        if device.type == "cuda":
            results[model_id]["peak_cuda_memory_mb"] = round(
                torch.cuda.max_memory_allocated(device) / (1024 * 1024), 3
            )
    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": importlib.import_module("torch_geometric").__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        environment["device_name"] = torch.cuda.get_device_name(device)
    print(json.dumps({"environment": environment, "batch_size": 2, "models": results}, indent=2))


if __name__ == "__main__":
    main()
