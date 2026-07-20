"""Early, dependency-light validation for training configuration values."""

import difflib
import importlib
import re
import warnings


def _dependency_error(message):
    """Keep dependency-light source tests while categorizing package imports."""

    try:
        from D4CMPP2.exceptions import DependencyError
    except ImportError:
        return RuntimeError(message)
    return DependencyError(message)


SCALERS = ("standard", "minmax", "normalizer", "robust", "identity")
REGISTRY_FIELDS = (
    "network",
    "data_manager_module",
    "data_manager_class",
    "network_manager_module",
    "network_manager_class",
    "train_manager_module",
    "train_manager_class",
)


def _type_name(value):
    return type(value).__name__


def validate_entry_args(kwargs):
    """Validate arguments whose contract is known before config/path loading."""
    modes = [
        key for key in ("LOAD_PATH", "RESUME_PATH", "TRANSFER_PATH")
        if kwargs.get(key) is not None
    ]
    if len(modes) > 1:
        raise ValueError(
            f"{modes} are mutually exclusive. Use LOAD_PATH for weight-only continuation, "
            "RESUME_PATH for full-state resume, or TRANSFER_PATH for parameter transfer."
        )
    if modes:
        key = modes[0]
        path = kwargs[key]
        if not isinstance(path, str) or not path.strip():
            raise TypeError(
                f"{key} must be a non-empty string, got {path!r} "
                f"({_type_name(path)}). Provide a saved model folder or checkpoint path."
            )
        if key in {"LOAD_PATH", "RESUME_PATH"}:
            return

    data = kwargs.get("data")
    if not isinstance(data, str) or not data.strip():
        raise TypeError(
            f"data must be a non-empty CSV name or path string, got {data!r} "
            f"({_type_name(data)}). Example: data='Aqsoldb.csv'."
        )

    target = kwargs.get("target")
    if not isinstance(target, list) or not target or not all(
        isinstance(column, str) and column.strip() for column in target
    ):
        raise TypeError(
            f"target must be a non-empty list of column-name strings, got {target!r} "
            f"({_type_name(target)}). Example: target=['Solubility']."
        )

    network = kwargs.get("network")
    if not isinstance(network, str) or not network.strip():
        raise TypeError(
            f"network must be a non-empty network ID string, got {network!r} "
            f"({_type_name(network)}). Example: network='GCN'."
        )


def validate_network_entry(network_id, entry, available_ids):
    """Validate one selected network registry entry before manager loading."""
    if entry is None:
        candidates = difflib.get_close_matches(network_id, available_ids, n=3, cutoff=0.5)
        message = (
            f"Network ID {network_id!r} was not found. "
            f"Available network IDs: {', '.join(available_ids)}."
        )
        if candidates:
            message += f" Did you mean {', '.join(repr(item) for item in candidates)}?"
        raise ValueError(message)

    missing = [
        field
        for field in REGISTRY_FIELDS
        if not isinstance(entry.get(field), str) or not entry[field].strip()
    ]
    if missing:
        raise ValueError(
            f"Registry entry for network {network_id!r} is missing required non-empty "
            f"keys: {missing}. Check the selected network reference YAML file."
        )


def validate_training_config(config, optimizer_names=None):
    """Validate merged training values without mutating the configuration."""
    scaler = config.get("scaler")
    if scaler not in SCALERS:
        raise ValueError(
            f"scaler must be one of {list(SCALERS)}, got {scaler!r}. "
            "Choose a supported target scaling method."
        )

    optimizer = config.get("optimizer")
    if not isinstance(optimizer, str) or not optimizer.strip():
        raise TypeError(
            f"optimizer must be a torch.optim class name string, got {optimizer!r} "
            f"({_type_name(optimizer)}). Example: optimizer='Adam'."
        )
    if optimizer_names is not None and optimizer not in optimizer_names:
        candidates = difflib.get_close_matches(optimizer, sorted(optimizer_names), n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(repr(item) for item in candidates)}?" if candidates else ""
        raise ValueError(
            f"Optimizer {optimizer!r} is not available in torch.optim.{hint} "
            "Use a torch.optim optimizer class name such as 'Adam'."
        )

    _positive_int(config, "max_epoch")
    _positive_int(config, "batch_size")
    _non_negative_int(config, "lr_patience")
    _non_negative_int(config, "early_stopping_patience")
    _positive_number(config, "learning_rate")
    _non_negative_number(config, "weight_decay")
    _non_negative_number(config, "min_lr")

    learning_rate = config["learning_rate"]
    min_lr = config["min_lr"]
    if min_lr > learning_rate:
        raise ValueError(
            f"min_lr ({min_lr!r}) must not exceed learning_rate ({learning_rate!r}). "
            "Lower min_lr or increase learning_rate."
        )

    pin_memory = config.get("pin_memory")
    if not isinstance(pin_memory, bool):
        raise TypeError(
            f"pin_memory must be bool, got {pin_memory!r} ({_type_name(pin_memory)}). "
            "Use pin_memory=True or pin_memory=False."
        )

    legacy_silent_errors = config.get("legacy_silent_errors", False)
    if not isinstance(legacy_silent_errors, bool):
        raise TypeError(
            f"legacy_silent_errors must be bool, got {legacy_silent_errors!r} "
            f"({_type_name(legacy_silent_errors)}). Use True only for temporary legacy compatibility."
        )

    verbose = config.get("verbose", True)
    if not isinstance(verbose, bool):
        raise TypeError(
            f"verbose must be bool, got {verbose!r} ({_type_name(verbose)}). "
            "Use verbose=True to show status output or verbose=False to hide it."
        )

    device = config.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
        raise ValueError(
            f"device must be 'cpu', 'cuda', or 'cuda:<non-negative index>', got {device!r}. "
            "Example: device='cpu' or device='cuda:0'."
        )

    scaler_scope = config.get("target_scaler_fit_scope", "train")
    if scaler_scope not in {"train", "all"}:
        raise ValueError(
            f"target_scaler_fit_scope must be 'train' or 'all', got {scaler_scope!r}. "
            "Use 'train' to avoid validation/test leakage or 'all' only for legacy numerical compatibility."
        )

    cache_policy = config.get("graph_cache_policy", "v2")
    if cache_policy not in {"v2", "legacy", "regenerate"}:
        raise ValueError(
            f"graph_cache_policy must be 'v2', 'legacy', or 'regenerate', got {cache_policy!r}. "
            "Use 'v2' for verified caches, 'legacy' only for explicit schema-v1 compatibility, "
            "or 'regenerate' to rebuild an invalid v2 cache."
        )

    split_strategy = config.get("split_strategy", "auto")
    if split_strategy not in {"auto", "random", "predefined", "scaffold"}:
        raise ValueError(
            f"split_strategy must be 'auto', 'random', 'predefined', or "
            f"'scaffold', got {split_strategy!r}."
        )
    scaffold_column = config.get("scaffold_column")
    if scaffold_column is not None and (
        not isinstance(scaffold_column, str) or not scaffold_column.strip()
    ):
        raise TypeError(
            f"scaffold_column must be a non-empty molecule-column string or None, "
            f"got {scaffold_column!r}."
        )
    include_chirality = config.get("scaffold_include_chirality", False)
    if not isinstance(include_chirality, bool):
        raise TypeError(
            f"scaffold_include_chirality must be bool, got "
            f"{include_chirality!r} ({_type_name(include_chirality)})."
        )

    random_seed = config.get("random_seed")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0
    ):
        raise ValueError(
            f"random_seed must be a non-negative integer or None, got {random_seed!r}. "
            "Example: random_seed=42."
        )
    deterministic = config.get("deterministic_algorithms", False)
    if not isinstance(deterministic, bool):
        raise TypeError(
            f"deterministic_algorithms must be bool, got {deterministic!r} "
            f"({_type_name(deterministic)})."
        )
    if deterministic and random_seed is None:
        raise ValueError(
            "deterministic_algorithms=True requires random_seed to be set. "
            "Example: random_seed=42, deterministic_algorithms=True."
        )

    lr_dict = config.get("lr_dict", {})
    if not isinstance(lr_dict, dict):
        raise TypeError(
            f"lr_dict must be a mapping of layer-name components to learning rates, "
            f"got {lr_dict!r} ({_type_name(lr_dict)})."
        )
    invalid_lr_entries = {
        key: value for key, value in lr_dict.items()
        if (
            not isinstance(key, str)
            or not key.strip()
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        )
    }
    if invalid_lr_entries:
        raise ValueError(
            "lr_dict keys must be non-empty layer-name strings and values must be "
            f"positive learning rates; invalid entries: {invalid_lr_entries!r}."
        )

    _validate_sculptor_config(config)


def validate_runtime_environment(config, backend="pyg", torch_module=None, importer=None):
    """Validate device availability and graph backend imports before model creation."""
    importer = importer or importlib.import_module
    if torch_module is None:
        try:
            torch_module = importer("torch")
        except (ImportError, OSError) as exc:
            raise _dependency_error(
                "PyTorch could not be imported, so training cannot start. "
                "Install a PyTorch build compatible with your Python and requested CPU/CUDA environment. "
                f"Original import error: {exc}"
            ) from exc

    device = config["device"]
    if device.startswith("cuda"):
        cuda = torch_module.cuda
        if not cuda.is_available():
            raise _dependency_error(
                f"Requested device {device!r}, but CUDA is not available to PyTorch "
                f"{getattr(torch_module, '__version__', 'unknown')!r}. "
                "Check the NVIDIA driver and CUDA-enabled PyTorch installation, or explicitly set device='cpu'. "
                "D4CMPP2 does not switch devices automatically."
            )
        index = 0 if device == "cuda" else int(device.split(":", 1)[1])
        count = cuda.device_count()
        if index >= count:
            raise _dependency_error(
                f"Requested device {device!r}, but PyTorch reports {count} CUDA device(s) "
                f"with valid indices 0..{count - 1}. Choose an available index or set device='cpu'."
            )
    elif config.get("pin_memory", False):
        warnings.warn(
            "pin_memory=True was requested with device='cpu'. Pinned host memory normally benefits "
            "CUDA transfers and may add overhead for CPU-only training; consider pin_memory=False.",
            UserWarning,
            stacklevel=2,
        )

    if backend != "pyg":
        raise ValueError(f"Unknown graph backend {backend!r}; this version supports only 'pyg'.")
    module_name = "torch_geometric"
    try:
        backend_module = importer(module_name)
    except (ImportError, OSError) as exc:
        torch_version = getattr(torch_module, "__version__", "unknown")
        raise _dependency_error(
            f"Graph backend {backend!r} could not be imported with PyTorch {torch_version!r}. "
            f"Install compatible torch/{module_name} builds for the same Python and CPU/CUDA environment. "
            f"Original import error: {exc}"
        ) from exc
    return {
        "torch": str(getattr(torch_module, "__version__", "unknown")),
        backend: str(getattr(backend_module, "__version__", "unknown")),
    }


def validate_sculptor_index_argument(value):
    """Validate the legacy tuple form before it is expanded into config keys."""
    if not isinstance(value, tuple):
        return
    if len(value) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ValueError(
            f"sculptor_index must be a tuple of three non-negative integers, got {value!r}. "
            "Example: sculptor_index=(6, 2, 0)."
        )


def _positive_int(config, key):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r} ({_type_name(value)}).")


def _non_negative_int(config, key):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer, got {value!r} ({_type_name(value)}).")


def _positive_number(config, key):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} must be a positive number, got {value!r} ({_type_name(value)}).")


def _non_negative_number(config, key):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{key} must be a non-negative number, got {value!r} ({_type_name(value)}).")


def _validate_sculptor_config(config):
    isa_manager = config.get("data_manager_module") == "ISADataManager"
    if not isa_manager:
        return

    keys = ("sculptor_s", "sculptor_c", "sculptor_a")
    missing = [key for key in keys if key not in config]
    if missing:
        original = config.get("sculptor_index")
        if isinstance(original, list):
            raise TypeError(
                f"sculptor_index must be a tuple of three non-negative integers, got list {original!r}. "
                "Example: sculptor_index=(6, 2, 0)."
            )
        raise ValueError(
            f"ISA network {config.get('network')!r} requires sculptor_index=(split, combine, absorb); "
            f"missing normalized keys: {missing}. Example: sculptor_index=(6, 2, 0)."
        )

    invalid = {key: config[key] for key in keys if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0}
    if invalid:
        raise ValueError(
            f"ISA sculptor values must be non-negative integers, got {invalid!r}. "
            "Example: sculptor_index=(6, 2, 0)."
        )
