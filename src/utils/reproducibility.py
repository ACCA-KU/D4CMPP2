"""Opt-in training seed and deterministic-backend configuration."""

import os
import random

import numpy as np
import torch


def capture_backend_determinism():
    cudnn = getattr(torch.backends, "cudnn", None)
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": getattr(cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(cudnn, "benchmark", None),
    }


def restore_backend_determinism(state):
    torch.use_deterministic_algorithms(state["deterministic_algorithms"])
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        if state["cudnn_deterministic"] is not None:
            cudnn.deterministic = state["cudnn_deterministic"]
        if state["cudnn_benchmark"] is not None:
            cudnn.benchmark = state["cudnn_benchmark"]


def configure_reproducibility(config, resume=False):
    """Apply requested policy and return effective state for reporting."""
    seed = config.get("random_seed")
    deterministic = config.get("deterministic_algorithms", False)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if seed is not None and not resume:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return effective_reproducibility_state(config, resume=resume)


def effective_reproducibility_state(config, resume=False):
    cudnn = getattr(torch.backends, "cudnn", None)
    return {
        "random_seed": config.get("random_seed"),
        "split_random_seed": config.get("split_random_seed", 42),
        "deterministic_algorithms_requested": config.get("deterministic_algorithms", False),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": getattr(cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(cudnn, "benchmark", None),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "rng_source": "checkpoint" if resume else "configured_seed" if config.get("random_seed") is not None else "ambient",
    }


def seed_data_loader_worker(worker_id):
    """Seed Python and NumPy from the worker seed assigned by PyTorch."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
