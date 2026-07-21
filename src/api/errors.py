"""Canonical public D4CMPP2 exception categories."""


class D4CMPPError(Exception):
    """Base class for categorized package failures."""


class ConfigError(ValueError, D4CMPPError):
    """Invalid or incompatible configuration."""


class DataError(ValueError, D4CMPPError):
    """Invalid tabular or aligned input data."""


class GraphError(ValueError, D4CMPPError):
    """Invalid graph data, graph schema, or graph generation state."""


class ModelError(RuntimeError, D4CMPPError):
    """Model construction, execution, or state-loading failure."""


class DependencyError(RuntimeError, D4CMPPError):
    """Missing or incompatible runtime dependency."""


class CheckpointError(D4CMPPError):
    """Base category for checkpoint failures."""


class CheckpointNotFoundError(FileNotFoundError, CheckpointError):
    """A requested checkpoint does not exist."""


class CheckpointFormatError(ValueError, CheckpointError):
    """A checkpoint payload or schema is invalid."""


class CheckpointLoadError(RuntimeError, CheckpointError):
    """A checkpoint exists but could not be deserialized."""


class CheckpointIOError(OSError, CheckpointError):
    """A checkpoint could not be committed to storage."""


class ModuleLoadError(ImportError, D4CMPPError):
    """A custom module exists but failed to import or initialize."""


class ManagerNotFoundError(FileNotFoundError, D4CMPPError):
    """A configured manager module or class could not be located."""
