"""Typed network registry used by training and optimization."""

from dataclasses import dataclass

from D4CMPP2.networks.base import MolecularNetwork


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    network: type[MolecularNetwork]
    aliases: tuple[str, ...] = ()
    data_contract: str = "molecule"

    def __post_init__(self):
        if not isinstance(self.network, type) or not issubclass(
            self.network, MolecularNetwork
        ):
            raise TypeError(
                f"Registered network for {self.name!r} must inherit MolecularNetwork."
            )
        if "<locals>" in self.network.__qualname__:
            raise ValueError(
                f"Network {self.network.__qualname__!r} must be defined at module "
                "scope so saved runs can import it."
            )
        if self.network.model_name != self.name:
            raise ValueError(
                f"Registry name {self.name!r} does not match "
                f"{self.network.__name__}.model_name={self.network.model_name!r}."
            )
        self.network.optimization_space()
        if self.data_contract not in {"molecule", "solvent", "isa"}:
            raise ValueError(
                f"Model {self.name!r} has unknown data_contract "
                f"{self.data_contract!r}. Expected 'molecule', 'solvent', or 'isa'."
            )

    def training_config(self):
        managers = {
            "molecule": ("MolDataManager", "MolDataManager", "NetworkManager", "NetworkManager", "TrainManager", "Trainer"),
            "solvent": ("MolDataManager", "MolDataManager_withSolv", "NetworkManager", "NetworkManager", "TrainManager", "Trainer"),
            "isa": ("ISADataManager", "ISADataManager", "ISANetworkManager", "ISANetworkManager", "ISATrainManager", "ISATrainer"),
        }
        data_module, data_class, network_module, network_class, train_module, train_class = managers[self.data_contract]
        return {
            "name": self.name,
            "network": self.network.__module__.rsplit(".", 1)[-1],
            "network_id": self.name,
            "data_manager_module": data_module,
            "data_manager_class": data_class,
            "network_manager_module": network_module,
            "network_manager_class": network_class,
            "train_manager_module": train_module,
            "train_manager_class": train_class,
            "description": self.network.__doc__ or self.name,
            "version": "2.0",
        }


_MODELS = {}


def register_model(definition):
    names = (definition.name, *definition.aliases)
    conflicts = [name for name in names if name in _MODELS]
    if conflicts:
        raise ValueError(f"Model names are already registered: {conflicts!r}.")
    for name in names:
        _MODELS[name] = definition
    return definition


def register_network(network, *, aliases=(), data_contract="molecule"):
    """Register a MolecularNetwork subclass for train and optimize entry points."""
    return register_model(
        ModelDefinition(
            network.model_name,
            network,
            aliases=tuple(aliases),
            data_contract=data_contract,
        )
    )


def get_model(name):
    try:
        return _MODELS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {name!r}. Registered names: {sorted(_MODELS)!r}."
        ) from exc


def registered_models():
    return {
        definition.name: definition
        for definition in _MODELS.values()
    }


def _register_builtins():
    from D4CMPP2.networks.AFP_model import AFP
    from D4CMPP2.networks.AFPwithSolv_model import SolventAFP
    from D4CMPP2.networks.DMPNN_model import DMPNN
    from D4CMPP2.networks.DMPNNwithSolv_model import SolventDMPNN
    from D4CMPP2.networks.GAT_model import GAT
    from D4CMPP2.networks.GATwithSolv_model import SolventGAT
    from D4CMPP2.networks.GCN_model import GCN
    from D4CMPP2.networks.GCNwithSolv_model import SolventGCN
    from D4CMPP2.networks.GC_model import GroupContributionNetwork
    from D4CMPP2.networks.ISAT_model import ISAT
    from D4CMPP2.networks.ISATPN_model import ISATPN
    from D4CMPP2.networks.MPNN_model import MPNN
    from D4CMPP2.networks.MPNNwithSolv_model import SolventMPNN

    register_model(ModelDefinition("afp", AFP, aliases=("AFP", "AFP_model")))
    register_model(
        ModelDefinition(
            "afp_solvent",
            SolventAFP,
            aliases=("AFPwS", "AFPwithSolv_model"),
            data_contract="solvent",
        )
    )
    register_model(
        ModelDefinition("dmpnn", DMPNN, aliases=("DMPNN", "DMPNN_model"))
    )
    register_model(
        ModelDefinition(
            "dmpnn_solvent",
            SolventDMPNN,
            aliases=("DMPNNwS", "DMPNNwithSolv_model"),
            data_contract="solvent",
        )
    )
    register_model(ModelDefinition("gcn", GCN, aliases=("GCN", "GCN_model")))
    register_model(
        ModelDefinition(
            "gcn_solvent",
            SolventGCN,
            aliases=("GCNwS", "GCNwithSolv_model"),
            data_contract="solvent",
        )
    )
    register_model(
        ModelDefinition(
            "group_contribution",
            GroupContributionNetwork,
            aliases=("GC", "GC_model"),
            data_contract="isa",
        )
    )
    register_model(
        ModelDefinition(
            "isat", ISAT, aliases=("ISAT", "ISAT_model"), data_contract="isa"
        )
    )
    register_model(
        ModelDefinition(
            "isatpn",
            ISATPN,
            aliases=("ISATPN", "ISATPM", "ISATPM_model"),
            data_contract="isa",
        )
    )
    register_model(ModelDefinition("gat", GAT, aliases=("GAT", "GAT_model")))
    register_model(
        ModelDefinition(
            "gat_solvent",
            SolventGAT,
            aliases=("GATwS", "GATwithSolv_model"),
            data_contract="solvent",
        )
    )
    register_model(ModelDefinition("mpnn", MPNN, aliases=("MPNN", "MPNN_model")))
    register_model(
        ModelDefinition(
            "mpnn_solvent",
            SolventMPNN,
            aliases=("MPNNwS", "MPNNwithSolv_model"),
            data_contract="solvent",
        )
    )


_register_builtins()
