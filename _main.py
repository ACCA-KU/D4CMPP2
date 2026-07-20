import yaml
import os
import traceback
import importlib
import warnings
from pathlib import Path
from D4CMPP2.src.utils import argparser
from D4CMPP2.src.utils import PATH
from D4CMPP2.src.PostProcessor import PostProcessor
from D4CMPP2.src.utils import module_loader, supportfile_saver
from D4CMPP2.src.utils.output import get_output
from D4CMPP2.src.utils.run_manifest import RunManifest
from D4CMPP2.src.utils.config_resolution import (
    ConfigResolution,
    mark_derived,
    merge_config_layers,
    overlay_config_layer,
    split_runtime_config,
)
from D4CMPP2.src.DataManager.contracts import validate_data_manager_contract
from D4CMPP2.src.utils.reproducibility import (
    capture_backend_determinism,
    configure_reproducibility,
    restore_backend_determinism,
)
from D4CMPP2.src.utils.config_validation import (
    validate_entry_args,
    validate_network_entry,
    validate_runtime_environment,
    validate_sculptor_index_argument,
    validate_training_config,
)

config0 = {
    "data": None,
    "target": None,
    "network": None,

    "scaler": "standard",
    "optimizer": "Adam",

    "max_epoch": 2000,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "lr_patience": 80,
    "early_stopping_patience": 200,
    'min_lr': 1e-5,

    "device": "cuda:0",
    "pin_memory": False,
    "target_scaler_fit_scope": "train",
    "legacy_silent_errors": False,
    "random_seed": None,
    "deterministic_algorithms": False,
    "verbose": True,

    'hidden_dim': None,
    'conv_layers':None,
    'linear_layers': None,
    'dropout': None,

    'solv_hidden_dim': None,
    'solv_conv_layers': None,
    'solv_linear_layers': None,
}


def train(**kwargs):
    """
    Train the network with the given configuration.

    Args for the training:
        (Required args for the scratch training)
        data        : str
                      the name or path of the data file as a csv file. you can omit the ".csv" extension.
        target      : list[str]
                      the name of the target column.
        network     : str
                      the ID of the network to use. Refer to the networks.yaml file for the available networks ID.
        molecule_columns: list[str]. Default= ['compound'] or ['compound', 'solvent']
                        the name of the columns to be used as the molecule columns. Please check the network sorce code for the required molecule columns for each network.
        numeric_input_columns: list[str]. Default= []
                        the name of the columns to be used as numeric input features. Please check the network sorce code for the required numeric input columns for each network.
        
        (Reqired args for continuing the training)
        LOAD_PATH   : str
                      the path of the directory that contains the model to continue the training.

        (Required args for the transfer learning)
        TRANSFER_PATH: str
                      the path of the directory that contains the model to transfer the learning.
        lr_dict     : dict (optional)
                      the dictionary for the learning rate of specific layers. e.g. {'GCNs': 0.0001}
                      you can find the layer names in "model_summary.txt" in the model directory.

                                    
        (Optional args)
        explicit_h_columns: list[str]. Default= []
                        the name of the columns to be used as explicit hydrogen features for the nodes.
        scaler      : str. Default= "standard",
                      ['standard', 'minmax', 'normalizer', 'robust', 'identity']
        optimizer   : str. Default= "Adam",
                      ['Adam', 'SGD', 'RMSprop', ... supported by torch.optim]
        max_epoch   : int. Defualt= 2000,
        batch_size  : int. Default= 256,
        learning_rate: float. Defualt= 0.001,
        weight_decay: float. Default= 0.0005,
        lr_patience : int. Defualt= 80,
                      the number of epochs with no improvement after which learning rate will be reduced.
        early_stopping_patience: 200,
                      the number of epochs with no improvement after which training will be stopped
        min_lr      : float. Defult= 1e-5,
                      the minimum learning rate
        device      : str. Defualt='cuda:0'
                      ['cpu', 'cuda:0', 'cuda:1', ...]
        pin_memory  : bool. Default= False
                      If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        split_random_seed: int. Default= 42,
        save_prediction: bool. Default= True,
        callbacks   : sequence of callback objects. Default= None
                      Runtime-only observers for training events. Callback objects
                      are not written to config.yaml or checkpoints.
        verbose     : bool. Default=True
                      Show informational messages, result tables, and progress bars.
                      Warnings and errors remain visible when False.
            
        DATA_PATH   : str. Default= None
                      the path of directory that contains the data file. If None, it will walk the subpath.
        NET_REFER   : str. Default= "{src}/network_refer.yaml"
                      the path to the network reference file. 
        MODEL_DIR   : str. Default= "./_Models"
                      the path to the directory to save the models.
        MODEL_PATH  : str. Default= None
                      the path to the directory to save the model. If None, it will be created in the MODEL_DIR.
        GRAPH_DIR  : str. Default= "./_Graphs"
                      the path to the directory to save the graphs.
        FRAG_REF    : str. Default= "{src}/utils/functional_group.csv"
                      the path to the reference file for the functional groups.

    Args for the network:
    (The following args will be varied according to the network chosen. Refer to the source code of the network for more details.
    In general, below are the common args for the network)
        hidden_dim      : int. Default= 64
                          the dimension of the hidden layers in the graph convolutional layers
        conv_layers     : int. Default= 6
                          the number of graph convolutional layers
        linear_layers   : int. Default= 3
                          the number of linear layers after the graph convolutional layers
        dropout         : float. Default= 0.1
    
    --------------------------------------------
    Example:

        train(data='data.csv', target=['target1','target2'], network='GCN',)

        train(LOAD_PATH='GCN_model_test_Abs_20240101_000000')

        train(TRANSFER_PATH='GCN_model_test_Abs_20240101_000000', data='data.csv', target=['target1','target2'], lr_dict={'GCNs': 0.0001})

        train(data='data.csv', target=['target1','target2'], network='GCN', hidden_dim=32, conv_layers=4, linear_layers=2, dropout=0.1)

    
    """
    callbacks = kwargs.pop("callbacks", None)
    if kwargs.pop('use_argparser', False):
        parsed = {key: value for key, value in vars(argparser.parse_args()).items() if value is not None}
        kwargs.update(parsed)
    kwargs = check_args(**kwargs)
    resolution = resolve_config(**kwargs)
    config = resolution.to_dict()
    if not config.get('loaded',False) and not config.get('full_resume', False):
        supportfile_saver.save_additional_files(config)
    return run(
        config,
        callbacks=callbacks,
        config_provenance=dict(resolution.provenance),
    )

def check_args(**kwargs):
    "check the arguments for the training"
    validate_entry_args(kwargs)
    if kwargs.get('LOAD_PATH',None) is not None or kwargs.get('RESUME_PATH',None) is not None:
        return kwargs
    kwargs['target_dim'] = len(kwargs['target'])
    if kwargs['data'][-4:] == '.csv':
        kwargs['data'] = kwargs['data'][:-4]
    return kwargs


def set_config(**kwargs):
    """Return the historical mutable dictionary config contract."""

    return resolve_config(**kwargs).to_dict()


def resolve_config(**kwargs):
    """Resolve an immutable config snapshot while preserving legacy precedence."""

    config, provenance = merge_config_layers((
        ("defaults", config0),
        ("api_or_cli", kwargs),
    ))
    config = {k: v for k, v in config.items() if v is not None}
    provenance = {key: provenance[key] for key in config}

    netrefer_loaded = False
    if config.get('RESUME_PATH', None) is not None:
        resume_overrides = {key: value for key, value in kwargs.items() if value is not None}
        requested = Path(config['RESUME_PATH'])
        if requested.is_file() or requested.suffix == ".ckpt":
            checkpoint = requested
            model_path = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
            if not model_path.is_dir():
                raise FileNotFoundError(
                    f"Model folder {str(model_path)!r} for RESUME_PATH {str(checkpoint)!r} was not found."
                )
            config['MODEL_PATH'] = str(model_path)
            config['RESUME_PATH'] = str(checkpoint)
        else:
            config['MODEL_PATH'] = PATH.find_model_path(config['RESUME_PATH'], config)
            config['RESUME_PATH'] = config['MODEL_PATH']
        resolved_model_path = config['MODEL_PATH']
        resolved_resume_path = config['RESUME_PATH']
        with open(os.path.join(resolved_model_path, 'config.yaml'), "r") as file:
            saved_config = yaml.safe_load(file)
        for policy_key, legacy_default in (
            ("random_seed", None),
            ("deterministic_algorithms", False),
        ):
            if (
                policy_key in resume_overrides
                and resume_overrides[policy_key] != saved_config.get(policy_key, legacy_default)
            ):
                raise ValueError(
                    f"RESUME_PATH cannot change {policy_key}: "
                    f"checkpoint run config={saved_config.get(policy_key, legacy_default)!r}, "
                    f"requested={resume_overrides[policy_key]!r}. "
                    "Full resume uses the saved RNG policy and checkpoint RNG state."
                )
        config, provenance = merge_config_layers((
            ("saved_resume", saved_config),
            ("resume_override", resume_overrides),
        ))
        config['MODEL_PATH'] = resolved_model_path
        provenance['MODEL_PATH'] = "derived_resume_path"
        config['RESUME_PATH'] = resolved_resume_path
        provenance['RESUME_PATH'] = "derived_resume_path"
        config['full_resume'] = True
        provenance['full_resume'] = "derived_mode"
        netrefer_loaded = True
    # overwrite the config with the loaded config if the LOAD_PATH is given
    if config.get('LOAD_PATH',None) is not None:
        config['LOAD_PATH'] = PATH.find_model_path(config['LOAD_PATH'] , config)
        config['MODEL_PATH'] = config.pop('LOAD_PATH')
        config['loaded'] = True
        with open(config['MODEL_PATH']+'/config.yaml', "r") as file:
            _config = yaml.safe_load(file)# _config = yaml.load(open(os.path.join(config['LOAD_PATH'],'config.yaml'), 'r'), Loader=yaml.FullLoader)
        legacy_scaler_scope = (
            'target_scaler_fit_scope' not in _config
            and 'target_scaler_fit_scope' not in kwargs
        )
        config, provenance = merge_config_layers((
            ("saved_load", _config),
            ("legacy_load_defaults_or_override", config),
        ))
        if legacy_scaler_scope:
            config['target_scaler_fit_scope'] = 'all'
            provenance['target_scaler_fit_scope'] = "legacy_compatibility"
            warnings.warn(
                "The loaded model config predates target_scaler_fit_scope; continuing with "
                "target_scaler_fit_scope='all' to preserve historical full-data scaling. "
                "Set target_scaler_fit_scope='train' explicitly to migrate to leakage-free scaling.",
                UserWarning,
                stacklevel=2,
            )
        netrefer_loaded = True

    PATH.init_path(config)

    if config.get('TRANSFER_PATH',None) is not None:
        config['TRANSFER_PATH'] = PATH.find_model_path(config['TRANSFER_PATH'],config)

        # if the network is not given, load the network from the transfer path
        if config.get('network', None):
            net_config = load_NET_REFER(config)
            config, provenance = overlay_config_layer(
                config,
                provenance,
                net_config,
                source="registry",
            )

        # load the config from the transfer path
        with open(config['TRANSFER_PATH']+'/config.yaml', "r") as file:
            _config = yaml.safe_load(file)
            _config.pop('MODEL_PATH',None)
        saved_values, saved_provenance = merge_config_layers((
            ("saved_transfer", _config),
        ))
        config, provenance = overlay_config_layer(
            saved_values,
            saved_provenance,
            config,
            layer_provenance=provenance,
        )
        config['MODEL_PATH'] = PATH.get_model_path(config)
        provenance['MODEL_PATH'] = "derived_transfer_path"
        netrefer_loaded = True
        
    if not netrefer_loaded:
        net_config = load_NET_REFER(config)
        config, provenance = overlay_config_layer(
            config,
            provenance,
            net_config,
            source="registry",
        )
        config['MODEL_PATH'] = PATH.get_model_path(config)
        provenance['MODEL_PATH'] = "derived_model_path"

    if config.get('sculptor_index',None) is not None and type(config['sculptor_index']) is tuple:
        validate_sculptor_index_argument(config['sculptor_index'])
        config['sculptor_s'] = config['sculptor_index'][0]
        config['sculptor_c'] = config['sculptor_index'][1]
        config['sculptor_a'] = config['sculptor_index'][2]
        sculptor_source = provenance.get('sculptor_index', "derived")
        provenance['sculptor_s'] = sculptor_source
        provenance['sculptor_c'] = sculptor_source
        provenance['sculptor_a'] = sculptor_source
        config.pop('sculptor_index')
        provenance.pop('sculptor_index', None)
    import torch
    optimizer_names = {
        name for name in dir(torch.optim)
        if isinstance(getattr(torch.optim, name), type)
    }
    validate_training_config(config, optimizer_names=optimizer_names)
    validate_runtime_environment(config, backend="pyg", torch_module=torch)
    PATH.check_path(config)
    mark_derived(provenance, config)
    return ConfigResolution.from_working(config, provenance)

def load_NET_REFER(config):
    from D4CMPP2.networks.registry import get_model, registered_models

    with open(PATH.get_network_refer(config), "r") as file:
        network_refer = yaml.safe_load(file)
    definition = None
    try:
        definition = get_model(config["network"])
    except ValueError:
        pass
    net_config = network_refer.get(config['network'],None)
    if net_config is None:
        if definition is not None:
            net_config = definition.training_config()
        else:
            validate_network_entry(
                config["network"],
                None,
                [*network_refer, *registered_models()],
            )
    validate_network_entry(config['network'], net_config, [
        *network_refer,
        *registered_models(),
    ])
    net_config = dict(net_config)
    net_config["network_id"] = (
        definition.name if definition is not None else config["network"]
    )
    return net_config


def run(config, callbacks=None, config_provenance=None):
    "the main function to run the training"
    config, runtime = split_runtime_config(config)
    output = get_output(config)
    tf_path = runtime.transfer_path
    resume_path = runtime.resume_path
    is_loaded = runtime.loaded
    is_resumed = runtime.full_resume
    mode = "resume" if is_resumed else "transfer" if tf_path else "legacy_continue" if is_loaded else "scratch"
    previous_backend_determinism = capture_backend_determinism()
    config["effective_reproducibility"] = configure_reproducibility(
        config,
        resume=is_resumed,
    )
    if tf_path:
        output.info(
            f"[Training] Starting transfer learning. Source model: {tf_path!r}; "
            f"output directory: {config['MODEL_PATH']!r}."
        )
    elif is_resumed:
        output.info(
            "[Training] Resuming optimizer and scheduler state. "
            f"Model directory: {config['MODEL_PATH']!r}."
        )
    elif is_loaded:
        output.info(
            "[Training] Continuing from saved model weights. "
            f"Model directory: {config['MODEL_PATH']!r}."
        )
    else:
        output.info(
            "[Training] Starting a new run. "
            f"Output directory: {config['MODEL_PATH']!r}."
        )

    manifest = RunManifest(config, mode)
    if config_provenance is not None:
        manifest.update(config_provenance=config_provenance)
    config["run_id"] = manifest.run_id
    try:
        dm = module_loader.load_data_manager(config)(config)
        data_contract = validate_data_manager_contract(dm, config)
        if data_contract is not None:
            manifest.update(
                data_manager_contract={
                    "manager": data_contract.manager,
                    "graph_type": data_contract.graph_type,
                    "dataset": data_contract.dataset,
                    "unwrapper": data_contract.unwrapper,
                    "feature_dimensions": dict(data_contract.feature_dimensions),
                    "batch": {
                        "name": data_contract.batch.name,
                        "required_keys": data_contract.batch.required_keys,
                        "optional_keys": data_contract.batch.optional_keys,
                        "molecule_suffixes": data_contract.batch.molecule_suffixes,
                        "numeric_suffix": data_contract.batch.numeric_suffix,
                    },
                }
            )
        network_manager_kwargs = {
            "tf_path": tf_path,
            "unwrapper": dm.unwrapper,
        }
        if resume_path is not None:
            network_manager_kwargs["resume_path"] = resume_path
        nm = module_loader.load_network_manager(config)(
            config,
            **network_manager_kwargs,
        )
        if tf_path and hasattr(nm, "transfer_report"):
            manifest.update(transfer=nm.transfer_report)
        if is_resumed:
            manifest.update(parent_run_id=getattr(nm, "parent_run_id", None))
        tm = module_loader.load_train_manager(config)(config)
        if callbacks is not None:
            set_callbacks = getattr(tm, "set_callbacks", None)
            if not callable(set_callbacks):
                raise TypeError(
                    f"Train manager {type(tm).__name__!r} does not support callbacks. "
                    "Implement set_callbacks(callbacks), or omit the callbacks argument."
                )
            set_callbacks(callbacks)
        dm.init_data()
        supportfile_saver.save_config(config)
        train_loaders, val_loaders, test_loaders = dm.get_Dataloaders()
        manifest.update(
            split={
                "train": len(dm.train_dataset),
                "val": len(dm.val_dataset),
                "test": len(dm.test_dataset),
            }
        )
        tm.train(nm, train_loaders, val_loaders)
        if tm.train_error is not None:
            raise tm.train_error
        pp = PostProcessor(config)
        pp.postprocess(dm, nm, tm, train_loaders, val_loaders, test_loaders)
    except KeyboardInterrupt as exc:
        manifest.finish("interrupted", error=exc)
        restore_backend_determinism(previous_backend_determinism)
        raise
    except Exception as exc:
        manifest.finish("failed", error=exc)
        restore_backend_determinism(previous_backend_determinism)
        if config.get('legacy_silent_errors', False):
            output.error("[Error] Training failed in legacy silent-error mode.")
            output.error(traceback.format_exc())
            return None
        raise
    manifest.finish(
        "completed",
        completed_epoch=nm.completed_epoch,
        best_epoch=nm.best_epoch,
        best_metric=nm.best_loss,
        final_learning_rate=nm.get_lr(),
        checkpoints={
            "latest": os.path.join(config["MODEL_PATH"], "checkpoints", "latest.ckpt"),
            "best": os.path.join(config["MODEL_PATH"], "checkpoints", "best.ckpt"),
            "final": os.path.join(config["MODEL_PATH"], "final.pth"),
        },
    )
    restore_backend_determinism(previous_backend_determinism)
    return config['MODEL_PATH']

if __name__ == "__main__":
    train(use_argparser=True)
