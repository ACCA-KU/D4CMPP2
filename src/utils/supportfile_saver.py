import pandas as pd
import yaml
import hashlib
import inspect
import json
from . import PATH

def save_funtionalgroup_csv(file_path,save_path):
    """
    save the functional group csv file
    """
    df = pd.read_csv(file_path)
    df.to_csv(save_path+'/functional_group.csv',index=False)

def save_network_refer(file_path,save_path):
    """
    save the network refer yaml file
    """
    with open(file_path, "r") as file:
        network_refer = yaml.safe_load(file)
    with open(save_path+'/network_refer.yaml', "w") as file:
        yaml.dump(network_refer, file)

def save_network_module(file_path,save_path):
    """
    save the network module
    """
    with open(file_path, "r") as file:
        network_module = file.read()
    with open(save_path+'/network.py', "w") as file:
        file.write(network_module)

def save_registered_network(network_class, save_path):
    source_path = inspect.getsourcefile(network_class)
    if source_path is None:
        raise ValueError(
            f"Cannot save network {network_class.__qualname__!r}: its source file "
            "cannot be located. Define custom networks in an importable .py module."
        )
    with open(source_path, "r", encoding="utf-8") as stream:
        source = stream.read()
    class_name = network_class.__qualname__
    if "." in class_name:
        raise ValueError(
            f"Cannot save nested network class {class_name!r}; define it at module scope."
        )
    with open(save_path + '/network.py', "w", encoding="utf-8") as file:
        file.write(source)
        file.write(f"\n\n# Saved D4CMPP2 entry point\nnetwork = {class_name}\n")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    with open(save_path + "/network_identity.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "module": network_class.__module__,
                "class": class_name,
                "source_sha256": digest,
                "snapshot": "network.py",
            },
            file,
            indent=2,
            sort_keys=True,
        )

def save_config(config):
    """
    save the configuration file
    """
    with open(config['MODEL_PATH']+'/config.yaml', "w") as file:
        yaml.dump(config, file)

def save_additional_files(config):
    """
    save the additional files
    """
    if "sculptor_s" in config:
        FRAG_REF = PATH.get_frag_ref_path(config)
        save_funtionalgroup_csv(FRAG_REF,config['MODEL_PATH'])

    NET_REF = PATH.get_network_refer(config)
    save_network_refer(NET_REF,config['MODEL_PATH'])

    try:
        from D4CMPP2.networks.registry import get_model

        definition = get_model(config.get("network_id", config["network"]))
    except ValueError:
        definition = None
    if definition is None:
        NET_MODULE = PATH.get_NET_DIR(config)+'/'+config['network']+'.py'
        save_network_module(NET_MODULE,config['MODEL_PATH'])
    else:
        save_registered_network(definition.network, config['MODEL_PATH'])
    
    save_config(config)
