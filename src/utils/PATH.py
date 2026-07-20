import os
import sys
import time
from pathlib import Path

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
main_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
def _cwd():
    return Path.cwd()

def _is_path_like(value):
    text = os.fspath(value)
    return Path(text).is_absolute() or "/" in text or "\\" in text

def _matches_below(root, name, directories):
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    predicate = Path.is_dir if directories else Path.is_file
    return sorted(
        (candidate for candidate in root_path.rglob(name) if predicate(candidate)),
        key=lambda candidate: str(candidate),
    )

def init_path(config):
    "Initialize the path for the training"
    if not "DATA_PATH" in config:
        init_data_path(config)
    if not "MODEL_DIR" in config:
        init_MODEL_DIR(config)
    init_GRAPH_DIR(config)
    # init_network_refer(config)
    # init_frag_ref_path(config)
    # init_NET_DIR(config)

def init_data_path(config):
    DATA_PATH = None
    data_value = os.fspath(config.get('data', None))
    data_file = data_value if data_value.lower().endswith('.csv') else data_value + ".csv"
    requested = Path(data_file)
    package_default = Path(base_dir) / "_Data" / requested.name
    if requested.is_file():
        DATA_PATH = str(requested)
    elif _is_path_like(data_file):
        raise FileNotFoundError(
            f"Data path {data_file!r} is path-like but is not an existing CSV file. "
            "Check the exact path; path-like inputs are not reinterpreted as basenames."
        )
    elif package_default.is_file():
        DATA_PATH = str(package_default)
    else:
        matches = _matches_below(_cwd(), requested.name, directories=False)
        if len(matches) == 1:
            DATA_PATH = str(matches[0])
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"Data file {data_file!r} is ambiguous below working directory {str(_cwd())!r}. "
                f"Candidates: {[str(match) for match in matches]}. Provide an exact path."
            )
    if DATA_PATH is None:
        searched = [data_file, str(package_default), f"recursively below {_cwd()}"]
        raise FileNotFoundError(
            f"Data file {data_file!r} was not found. Searched: {searched}. "
            "Check the data argument or provide DATA_PATH with an existing CSV file."
        )
    config["DATA_PATH"] = DATA_PATH
    return DATA_PATH
    
def init_GRAPH_DIR(config):
    if 'GRAPH_DIR' in config:
        GRAPH_DIR = config['GRAPH_DIR']
    elif os.path.exists("./_Graphs"):
        GRAPH_DIR = "./_Graphs"
    else:
        os.makedirs("./_Graphs")
        GRAPH_DIR = "./_Graphs"
    config["GRAPH_DIR"] = GRAPH_DIR
    return GRAPH_DIR
    
def init_MODEL_DIR(config):
    if 'MODEL_DIR' in config:
        MODEL_DIR = config['MODEL_DIR']
    elif os.path.exists("./_Models"):
        MODEL_DIR = "./_Models"
    else:
        os.makedirs("./_Models")
        MODEL_DIR = "./_Models"
    config["MODEL_DIR"] = MODEL_DIR
    return MODEL_DIR
    
def get_network_refer(config):
    NET_REFER = None
    if 'NET_REFER' in config:
        NET_REFER = config.pop('NET_REFER')
    elif os.path.exists(base_dir+"/network_refer.yaml"):
        NET_REFER = base_dir+"/network_refer.yaml"
    
    if NET_REFER is None or not os.path.isfile(NET_REFER):
        raise FileNotFoundError(
            f"NET_REFER {NET_REFER!r} is not an existing network reference file. "
            f"Checked explicit NET_REFER and package default {os.path.join(base_dir, 'network_refer.yaml')!r}. "
            "Provide NET_REFER with an existing YAML file."
        )
    return NET_REFER

def get_frag_ref_path(config):
    FRAG_REF = None
    if 'FRAG_REF' in config:
        FRAG_REF = config.pop('FRAG_REF')
    elif os.path.exists(base_dir+"/src/utils/functional_group.csv"):
        FRAG_REF = base_dir+"/src/utils/functional_group.csv"
    
    if FRAG_REF is None or not os.path.isfile(FRAG_REF):
        raise FileNotFoundError(
            f"FRAG_REF {FRAG_REF!r} is not an existing functional-group reference file. "
            f"Checked explicit FRAG_REF and package default {os.path.join(base_dir, 'src', 'utils', 'functional_group.csv')!r}. "
            "Provide FRAG_REF with an existing CSV file."
        )
    return FRAG_REF
    
def get_NET_DIR(config):
    if 'NET_DIR' in config:
        NET_DIR = config.pop('NET_DIR')
    elif (_cwd() / "networks").exists():
        NET_DIR = str(_cwd() / "networks")
    elif os.path.exists(base_dir+"/networks"):
        NET_DIR = base_dir+"/networks"
    else:
        NET_DIR = None
    if NET_DIR is None or not os.path.isdir(NET_DIR):
        raise FileNotFoundError(
            f"NET_DIR {NET_DIR!r} is not an existing network source directory. "
            f"Checked explicit NET_DIR, {str(_cwd() / 'networks')!r}, and "
            f"{os.path.join(base_dir, 'networks')!r}. Provide NET_DIR with an existing directory."
        )
    return NET_DIR
                          
    

def check_path(config):
    "Check the path of validity for the training"
    if not os.path.isfile(config['DATA_PATH']):
        raise FileNotFoundError(
            f"DATA_PATH {config['DATA_PATH']!r} is not an existing file. "
            "Provide the CSV file path via data or DATA_PATH."
        )
    if not os.path.isdir(config['GRAPH_DIR']):
        raise FileNotFoundError(
            f"GRAPH_DIR {config['GRAPH_DIR']!r} is not an existing directory. "
            "Create it or omit GRAPH_DIR to use ./_Graphs."
        )
    if not os.path.isdir(config['MODEL_DIR']):
        raise FileNotFoundError(
            f"MODEL_DIR {config['MODEL_DIR']!r} is not an existing directory. "
            "Create it or omit MODEL_DIR to use ./_Models."
        )


def find_model_path(model_name,config=None):
    requested = Path(os.fspath(model_name))
    if requested.is_dir():
        return str(requested)
    if _is_path_like(model_name):
        raise FileNotFoundError(
            f"Model path {os.fspath(model_name)!r} is path-like but is not an existing directory. "
            "Check the exact path; path-like inputs are not reinterpreted as basenames."
        )
    if config is None:
        MODEL_DIR = "."
    else:
        if 'MODEL_DIR' in config:
            MODEL_DIR = config['MODEL_DIR']
        else:
            MODEL_DIR = init_MODEL_DIR(config)
    matches = _matches_below(MODEL_DIR, requested.name, directories=True)
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Model folder {os.fspath(model_name)!r} is ambiguous below MODEL_DIR {MODEL_DIR!r}. "
            f"Candidates: {[str(match) for match in matches]}. Provide an exact model path or narrow MODEL_DIR."
        )
    raise FileNotFoundError(
        f"Model folder {model_name!r} was not found below MODEL_DIR {MODEL_DIR!r}. "
        f"Resolved search root: {os.path.abspath(MODEL_DIR)!r}. Check the model folder name or MODEL_DIR. "
        "Provide an exact model path or check the model folder name and MODEL_DIR."
    )

def _model_data_label(data):
    """Return a stable dataset label for an automatically generated model folder."""
    data_text = os.fspath(data).replace("\\", "/")
    label = Path(data_text).name
    if label.lower().endswith(".csv"):
        label = label[:-4]
    if not label:
        raise ValueError(
            f"Cannot derive a model folder label from data path {data!r}. "
            "Provide a CSV file name or dataset name."
        )
    return label


def get_model_path(config, make_dir=True):
    "Provide the new model path for the training"
    if 'MODEL_DIR' in config:
        MODEL_DIR = config['MODEL_DIR']
    else:
        MODEL_DIR = init_MODEL_DIR(config)

    if config.get('MODEL_PATH',None) is not None:
        requested = os.fspath(config['MODEL_PATH'])
        path = requested if _is_path_like(requested) else os.path.join(MODEL_DIR, requested)
    else:
        data_label = _model_data_label(config['data'])
        if config.get('TRANSFER_PATH',None) is not None:
            tf_name = Path(os.fspath(config['TRANSFER_PATH']).replace('\\', '/')).name
            tf_parts = tf_name.split('_')
            if len(tf_parts) > 2:
                tf_name = "_".join(tf_parts[:-2])
            path = os.path.join(MODEL_DIR,tf_name+"~"+config['network']+"_"+data_label+'_'+','.join(config['target']))
        else:
            path = os.path.join(MODEL_DIR,config['network']+"_"+data_label+'_'+','.join(config['target']))
        if 'sculptor_index' in config:
            path += "_"+''.join([str(i) for i in config['sculptor_index']])
        path += "_"+time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)
    config["MODEL_PATH"] = path
    return path

def get_xyz_dir_path(config):
    """Resolve the legacy conformer/XYZ cache directory."""
    if 'xyz_dir' in config:
        XYZ_DIR = config['xyz_dir']
    elif (_cwd() / "_XYZ").exists():
        XYZ_DIR = str(_cwd() / "_XYZ")
    else:
        os.makedirs(_cwd() / "_XYZ")
        XYZ_DIR = str(_cwd() / "_XYZ")
    if not os.path.exists(XYZ_DIR):
        raise FileNotFoundError(
            f"xyz_dir {XYZ_DIR!r} is not an existing XYZ directory. "
            "Create it or omit xyz_dir to use the working-directory _XYZ folder."
        )
    config["XYZ_PATH"] = XYZ_DIR
    return XYZ_DIR
