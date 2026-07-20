import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Union

from D4CMPP2.exceptions import ManagerNotFoundError, ModuleLoadError

Config = Mapping[str, Any]
ManagerFactory = Callable[[str, str], Any]


def load_train_manager(config: Config) -> Any:
    """Construct the configured built-in or external training manager."""

    return _load_manager(
        "TrainManager_PATH",
        "train_manager_module",
        "train_manager_class",
        load_default_train_manager,
    )(config)


def load_data_manager(config: Config) -> Any:
    """Construct the configured built-in or external data manager."""

    return _load_manager(
        "DataManager_PATH",
        "data_manager_module",
        "data_manager_class",
        load_default_data_manager,
    )(config)


def load_network_manager(config: Config) -> Any:
    """Construct the configured built-in or external network manager."""

    return _load_manager(
        "NetworkManager_PATH",
        "network_manager_module",
        "network_manager_class",
        load_default_network_manager,
    )(config)


def _load_manager(
    path_key: str,
    module_key: str,
    class_key: str,
    default_manager: ManagerFactory,
) -> Callable[[Config], Any]:
    def load_manager(config: Config) -> Any:
        module_name = config.get(module_key)
        class_name = config.get(class_key)
        if not isinstance(module_name, str) or not isinstance(class_name, str):
            raise ManagerNotFoundError(
                f"Manager config requires string keys {module_key!r} and "
                f"{class_key!r}; got module={module_name!r}, class={class_name!r}."
            )
        if path_key in config:
            path = config[path_key]
            module_path = os.path.join(path, module_name + ".py")
            if not os.path.isfile(module_path):
                raise ManagerNotFoundError(
                    f"Custom manager module {module_path!r} was not found. "
                    f"Check {path_key}, {module_key}, and {class_key}."
                )
            return load_module(path, module_name, class_name)
        try:
            return default_manager(module_name, class_name)
        except (AttributeError, KeyError) as exc:
            raise ManagerNotFoundError(
                f"Built-in manager {module_name}.{class_name} was not found. "
                f"Check {module_key} and {class_key} in the selected registry."
            ) from exc

    return load_manager


def load_default_train_manager(module_name: str, class_name: str) -> Any:
    """Resolve a training-manager class from the installed package."""

    return _load_default_manager("TrainManager", module_name, class_name)


def load_default_data_manager(module_name: str, class_name: str) -> Any:
    """Resolve a data-manager class from the installed package."""

    return _load_default_manager("DataManager", module_name, class_name)


def load_default_network_manager(module_name: str, class_name: str) -> Any:
    """Resolve a network-manager class from the installed package."""

    return _load_default_manager("NetworkManager", module_name, class_name)


def _load_default_manager(family: str, module_name: str, class_name: str) -> Any:
    module = importlib.import_module(f"D4CMPP2.src.{family}.{module_name}")
    return getattr(module, class_name)


def load_module(
    path: Union[str, os.PathLike[str]],
    module: str,
    class_name: str,
) -> Any:
    """Load one custom manager class without permanently changing ``sys.path``."""

    directory = str(Path(path).resolve())
    module_path = os.path.join(directory, module + ".py")
    identity = hashlib.sha256(
        os.path.normcase(module_path).encode("utf-8")
    ).hexdigest()[:16]
    internal_name = f"_d4cmpp2_custom_{module}_{identity}"
    inserted_path = directory not in sys.path
    try:
        if inserted_path:
            sys.path.insert(0, directory)
        spec2 = importlib.util.spec_from_file_location(internal_name, module_path)
        if spec2 is None or spec2.loader is None:
            raise ImportError("Python could not create a module specification.")
        module2 = importlib.util.module_from_spec(spec2)
        sys.modules[internal_name] = module2
        spec2.loader.exec_module(module2)
    except (ImportError, OSError, SyntaxError) as exc:
        sys.modules.pop(internal_name, None)
        raise ModuleLoadError(
            f"Custom module {module_path!r} could not be loaded: {exc}"
        ) from exc
    except BaseException:
        sys.modules.pop(internal_name, None)
        raise
    finally:
        if inserted_path:
            try:
                sys.path.remove(directory)
            except ValueError:
                pass
    try:
        return getattr(module2, class_name)
    except AttributeError as exc:
        raise ManagerNotFoundError(
            f"Custom module {module_path!r} does not define class {class_name!r}."
        ) from exc
