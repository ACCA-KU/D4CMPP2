import os
import pandas as pd
import numpy as np
from rdkit import Chem

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .GraphGenerator.MolGraphGenerator import MolGraphGenerator
from .Dataset.GraphDataset import GraphDataset, GraphDataset_withSolv

from D4CMPP2.src.utils.scaler import Scaler
from D4CMPP2.src.utils.csv_validation import (
    validate_aligned_lengths,
    validate_automatic_split_size,
    validate_csv_schema,
    validate_nonempty_targets,
    validate_numeric_columns,
    validate_set_labels,
)
from D4CMPP2.src.utils.data_quality import write_data_quality_report
from D4CMPP2.src.utils.graph_cache import (
    atomic_save_graph_cache,
    build_graph_recipe,
    cache_path,
    legacy_cache_paths,
    validate_graph,
    validate_payload,
)
from D4CMPP2.src.utils.reproducibility import seed_data_loader_worker
from D4CMPP2.src.utils.output import get_output
from D4CMPP2.src.utils.splitting import (
    scaffold_split_indices,
    write_split_report,
)

import hashlib
import warnings


class MolDataManager:
    """The class for the management of the molecular data, preparing the dataset, and dataloader
    
    This class is intended to provide the dataset and dataloader for the training, validation, and test.
    This class works mainly in two steps.

    1. call the init_data() to load the data from the csv file and prepare the graph.
        1-1. the load_data() will load the data from the csv file, saving the smiles and target values.
        1-2. the prepare_graph() will prepare the graph from the saved smiles, trying to load the graph first, then generate the graph if it does not exist.

    2. call the get_Dataloaders() to provide the dataloader for the training, validation, and test.
        2-1. the prepare_dataset() will drop the invalid data and initialize the dataset with generated graphs and target values.
        2-2. the split_data() will split the dataset into train, val, and test. if the set is given, the dataset will be split according to the set.
        2-3. the init_Dataloader() will initialize the dataloader based on the set and partial_train.    
    """
    feature_dimension_keys = ("node_dim", "edge_dim")

    def __init__(self, config):
        self.config = config
        self.output = get_output(config)
        self.data = config["data"]
        self.target = config["target"]
        self.scaler = Scaler(config.get("scaler","identity"))
        self.explicit_h_columns = config.get("explicit_h_columns",[])
        self.molecule_columns = config.get("molecule_columns", ['compound'])
        self.numeric_input_columns = config.get("numeric_input_columns", [])
        self.molecule_graphs={col:[] for col in self.molecule_columns}
        self._molecule_smiles = {}
        self.molecule_smiles = {}
        self.valid_smiles = {}
        self.original_row_indices = np.array([], dtype=int)
        self.graph_errors = []
        self.random_seed = config.get("split_random_seed",42)
        training_seed = config.get("random_seed")
        self._partial_train_rng = (
            np.random.default_rng(training_seed)
            if training_seed is not None else None
        )
        self.set = None
        self.import_others()
        config.update({"node_dim":self.gg.node_dim, "edge_dim":self.gg.edge_dim})

    def _output(self):
        """Return the configured adapter, including for legacy test fixtures."""

        return getattr(self, "output", get_output(getattr(self, "config", None)))

    # Import the graph generator and dataset
    def import_others(self):
        self.graph_type = 'mol'
        self.gg =MolGraphGenerator()
        self.dataset =GraphDataset
        self.unwrapper = self.dataset.unwrapper

    # Initialize the temporary data for the prediction
    def init_temp_data(self, *args, **kwargs):
        if len(args)+len(kwargs) != len(self.molecule_columns) + len(self.numeric_input_columns):
            raise ValueError(f"Please provide the smiles for {self.molecule_columns} and numeric input for {self.numeric_input_columns}.")
        if len(args) > 0:
            self._output().info(
                "[Data] Positional inference inputs follow this order: "
                f"molecule columns={self.molecule_columns!r}, "
                f"numeric columns={self.numeric_input_columns!r}."
            )
            if len(args) == len(self.molecule_columns) + len(self.numeric_input_columns):
                kwargs = {self.molecule_columns[i]: args[i] for i in range(len(self.molecule_columns))}
                kwargs.update({self.numeric_input_columns[i]: args[i + len(self.molecule_columns)] for i in range(len(self.numeric_input_columns))})
            elif len(args) == len(self.molecule_columns):
                kwargs.update({self.molecule_columns[i]: args[i] for i in range(len(self.molecule_columns))})
            elif len(args) == len(self.numeric_input_columns):
                kwargs.update({self.numeric_input_columns[i]: args[i] for i in range(len(self.numeric_input_columns))})
            else:
                raise ValueError(f"Please provide the smiles for {self.molecule_columns} and numeric input for {self.numeric_input_columns}.")
        if len(kwargs) >0:
            for k in kwargs.keys():
                if k not in self.molecule_columns and k not in self.numeric_input_columns:
                    raise ValueError(f"Unknown key {k}. Please provide the smiles for {self.molecule_columns} or numeric input for {self.numeric_input_columns}.")
            for k in self.molecule_columns:
                if k not in kwargs:
                    raise ValueError(f"Please provide the smiles for {k}.")
            for k in self.numeric_input_columns:
                if k not in kwargs:
                    raise ValueError(f"Please provide the numeric input for {k}.")
        for k in kwargs.keys():
            if type(kwargs[k]) is not list :
                kwargs[k] = [kwargs[k]]
        inputs = {col: kwargs[col] for col in self.molecule_columns if col in kwargs}

        self._molecule_smiles = {col: np.array(inputs[col]) for col in self.molecule_columns}
        self.molecule_smiles = {col: arr.copy() for col, arr in self._molecule_smiles.items()}
        self.valid_smiles = {col: np.array(inputs[col]) for col in self.molecule_columns}
        self.numeric_inputs = {col: np.array(kwargs[col]) for col in self.numeric_input_columns if col in kwargs}
        self.original_row_indices = np.arange(len(self._molecule_smiles[self.molecule_columns[0]]))

        self.molecule_graphs={col:[] for col in self.molecule_columns}
        self.target_value = np.zeros((len(self._molecule_smiles[self.molecule_columns[0]]), self.config["target_dim"]))
        self.gg.verbose = False
        for col in self.molecule_columns:
            self.generate_graph(col)

        # Keep all arrays aligned by applying one centralized mask after graph generation.
        self.drop_none_graph()

        result = self.valid_smiles.copy()
        result.update(self.numeric_inputs)
        return result

    # (main function) Initialize the data from the csv file based on the configuration
    def init_data(self):
        self.graph_errors = []
        self.load_data()
        self.prepare_graph()
        self.write_graph_error_report()
        self.write_data_quality_report()

    def load_csv(self):
        path = os.path.join(self.config['DATA_PATH'])
        encodings = ('utf-8', 'cp949', 'euc-kr')
        decode_errors = []
        for encoding in encodings:
            try:
                self.df = pd.read_csv(path, encoding=encoding)
                return
            except UnicodeDecodeError as exc:
                decode_errors.append(f"{encoding}: {exc}")
            except pd.errors.EmptyDataError as exc:
                raise ValueError(
                    f"CSV file {path!r} is empty or has no columns. "
                    "Add a header row and data rows before training."
                ) from exc
            except pd.errors.ParserError as exc:
                raise ValueError(
                    f"Could not parse CSV file {path!r} using encoding {encoding!r}: {exc}. "
                    "Check delimiters, quoting, and malformed rows."
                ) from exc
            except OSError as exc:
                raise OSError(
                    f"Could not read CSV file {path!r} using encoding {encoding!r}: {exc}. "
                    "Check that the path exists and is readable."
                ) from exc
        raise UnicodeError(
            f"Could not decode CSV file {path!r}. Tried encodings {list(encodings)}. "
            f"Decode errors: {decode_errors}. Save the file as UTF-8 or specify a supported Korean encoding."
        )

    # Load the data from the csv file, saving the smiles and target values
    def load_data(self):
        self.load_csv()
        self.write_data_quality_report()
        path = self.config['DATA_PATH']
        validate_csv_schema(
            path,
            self.df.columns,
            len(self.df),
            self.molecule_columns,
            self.target,
            self.numeric_input_columns,
        )

        converted_targets = self.df[self.target].apply(pd.to_numeric, errors='coerce')
        invalid_targets = {
            col: self.df.index[(converted_targets[col].isna()) & (self.df[col].notna())].tolist()
            for col in self.target
        }
        validate_numeric_columns(path, invalid_targets, "Target")
        validate_nonempty_targets(
            path,
            {col: bool(converted_targets[col].isna().all()) for col in self.target},
        )

        if self.numeric_input_columns:
            converted_numeric = self.df[self.numeric_input_columns].apply(pd.to_numeric, errors='coerce')
            invalid_numeric = {
                col: self.df.index[(converted_numeric[col].isna()) & (self.df[col].notna())].tolist()
                for col in self.numeric_input_columns
            }
            validate_numeric_columns(path, invalid_numeric, "Numeric input")

        if "set" in self.df.columns:
            validate_set_labels(path, self.df["set"].tolist())
        else:
            validate_automatic_split_size(path, len(self.df))

        for col in self.molecule_columns:
            self._molecule_smiles[col] = np.array(list(self.df[col]))
            self.molecule_smiles[col] = self._molecule_smiles[col].copy()
            self.valid_smiles[col] = self._molecule_smiles[col].copy()
        self.target_value = torch.tensor(converted_targets.values, dtype=torch.float32)
        if self.target_value.dim() == 1:
            self.target_value = self.target_value.unsqueeze(1)
        self.numeric_inputs = {
            col: np.asarray(converted_numeric[col].values, dtype=np.float32)
            for col in self.numeric_input_columns
        } if self.numeric_input_columns else {}
        self.config.update({"target_dim":self.target_value.shape[1]})
        self.set = self.df.get("set",None)
        self.original_row_indices = np.asarray(self.df.index)
        lengths = {f"molecule:{col}": len(values) for col, values in self._molecule_smiles.items()}
        lengths["target"] = len(self.target_value)
        lengths.update({f"numeric:{col}": len(self.df[col]) for col in self.numeric_input_columns})
        if self.set is not None:
            lengths["set"] = len(self.set)
        validate_aligned_lengths(path, lengths)

    def write_data_quality_report(self):
        if not self.config.get("data_quality_report", True):
            return None
        if not self.config.get("MODEL_PATH"):
            return None
        report_path, issues_path = write_data_quality_report(
            self.df,
            self.config,
            getattr(self, "graph_errors", None),
        )
        self._output().info(
            f"[Data] Quality report written to {str(report_path)!r}; "
            f"row-level issues: {str(issues_path)!r}."
        )
        return str(report_path), str(issues_path)

    # Prepare the graph from the smiles, trying to load the graph first, then generate the graph if it does not exist
    def prepare_graph(self):
        for col in self.molecule_columns:
            recipe = build_graph_recipe(self, col)
            path = cache_path(self, col, recipe)
            policy = self.config.get("graph_cache_policy", "v2")
            if policy not in ("v2", "legacy", "regenerate"):
                raise ValueError(
                    f"graph_cache_policy must be 'v2', 'legacy', or 'regenerate', got {policy!r}."
                )
            if path.is_file():
                try:
                    self.load_graphs(col)
                    continue
                except (OSError, RuntimeError, ValueError, TypeError) as exc:
                    if policy != "regenerate":
                        raise ValueError(
                            f"Graph cache {str(path)!r} is corrupt or incompatible: {exc} "
                            "Remove the file and rerun, or set graph_cache_policy='regenerate' "
                            "to rebuild it from the source CSV."
                        ) from exc
                    warnings.warn(
                        f"Regenerating corrupt or incompatible graph cache {str(path)!r}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            legacy_v1, legacy_dgl = legacy_cache_paths(self, col)
            if policy == "legacy" and legacy_v1.is_file():
                warnings.warn(
                    f"Loading legacy v1 graph cache {str(legacy_v1)!r}. Its feature recipe and "
                    "ISA rule identity cannot be verified; use the default v2 policy when possible.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._load_legacy_graphs(col, legacy_v1)
                continue
            legacy_found = [str(candidate) for candidate in (legacy_v1, legacy_dgl) if candidate.is_file()]
            if legacy_found:
                warnings.warn(
                    f"Legacy graph cache files were preserved but will not be loaded: {legacy_found}. "
                    f"Generating fingerprinted v2 cache {str(path)!r} from the source CSV.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.generate_graph(col)
            self.save_graphs(col)

    # Generate the graph from the smiles. the data failed to generate the graph will be saved as an empty graph
    def generate_graph(self,col,graphs=None):
        if graphs is None:
            # Always rebuild this column from scratch to avoid accidental append accumulation.
            self.molecule_graphs[col] = []
        if graphs is not None:
            for smi,g in zip(self._molecule_smiles[col],graphs):
                self.molecule_graphs[col].append(g)
                self.valid_smiles[col].append(smi)
            return
        if len(self._molecule_smiles[col]) < 10:
            iterator = self._molecule_smiles[col]
        else:
            iterator = self._output().progress(
                self._molecule_smiles[col],
                desc=f"Generating {col} graphs",
            )
        for position, smi in enumerate(iterator):
            graphgenerator = self.gg
            if col == 'solvent' and hasattr(self,'gg_solv'):
                graphgenerator = self.__getattribute__('gg_solv')
            else:
                graphgenerator = self.gg
            try:
                g=graphgenerator.get_graph(smi, explicit_h = col in self.explicit_h_columns, **self.config)
            except Exception as e:
                g = graphgenerator.get_empty_graph() # save the empty graph instead of the failed graph
                self._record_graph_error(position, smi, col, str(e))
            self.molecule_graphs[col].append(g)

    def _record_graph_error(self, position, smiles, molecule_column, reason):
        if not hasattr(self, 'graph_errors'):
            self.graph_errors = []
        if hasattr(self, 'original_row_indices') and len(self.original_row_indices) > position:
            row_index = self.original_row_indices[position]
            if hasattr(row_index, 'item'):
                row_index = row_index.item()
        else:
            row_index = position
        self.graph_errors.append({
            'smiles': smiles,
            'type': molecule_column,
            'reason': reason,
            'row_index': row_index,
        })

    def write_graph_error_report(self):
        if not self.graph_errors:
            return None
        report_path = os.path.join(self.config['MODEL_PATH'], 'graph_error.csv')
        report = pd.DataFrame(
            self.graph_errors,
            columns=['smiles', 'type', 'reason', 'row_index'],
        )
        report.to_csv(report_path, index=False)
        counts = report.groupby('type', sort=False).size().to_dict()
        unique_rows = report['row_index'].nunique()
        self._output().warning(
            f"[Data] Graph generation excluded {unique_rows} unique CSV rows "
            f"({len(report)} failures by molecule column: {counts}). "
            f"Details: {report_path!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return report_path
        
    def get_graphs_path(self,col):
        recipe = build_graph_recipe(self, col)
        return str(cache_path(self, col, recipe))

    # Save the generated graphs
    def save_graphs(self,col):
        recipe = build_graph_recipe(self, col)
        for index, graph in enumerate(self.molecule_graphs[col]):
            validate_graph(graph, recipe, index)
        atomic_save_graph_cache(
            {
                "graph_backend": "pyg",
                "graph_schema_version": 2,
                "recipe": recipe,
                "smiles": list(self._molecule_smiles[col]),
                "graphs": self.molecule_graphs[col],
                "graph_errors": [
                    error for error in self.graph_errors if error.get("type") == col
                ],
            },
            cache_path(self, col, recipe),
        )

    # Load the saved graphs
    def load_graphs(self,col):
        recipe = build_graph_recipe(self, col)
        path = cache_path(self, col, recipe)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        graphs, cached_errors = validate_payload(
            payload,
            recipe,
            self._molecule_smiles[col],
            path,
        )
        self.molecule_graphs[col] = graphs
        self.graph_errors.extend(cached_errors)
        for position, graph in enumerate(self.molecule_graphs[col]):
            if graph.num_nodes == 0 and not any(
                error.get("type") == col and error.get("row_index") == self.original_row_indices[position]
                for error in cached_errors
            ):
                self._record_graph_error(
                    position,
                    self._molecule_smiles[col][position],
                    col,
                    "Empty graph loaded from cache; the original generation error is unavailable.",
                )

    def _load_legacy_graphs(self, col, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("graph_backend") != "pyg"
            or payload.get("graph_schema_version") != 1
        ):
            raise ValueError(
                f"Legacy graph cache {str(path)!r} is not a PyG schema-v1 payload. "
                "Use the default v2 policy to regenerate it from the source CSV."
            )
        graphs = payload.get("graphs")
        smiles = payload.get("smiles")
        if not isinstance(graphs, list) or len(graphs) != len(self._molecule_smiles[col]):
            raise ValueError(
                f"Legacy graph cache {str(path)!r} has the wrong graph count. "
                "Regenerate a v2 cache from the source CSV."
            )
        if [str(value) for value in smiles or []] != [
            str(value) for value in self._molecule_smiles[col]
        ]:
            raise ValueError(
                f"Legacy graph cache {str(path)!r} ordered SMILES do not match the source CSV."
            )
        recipe = build_graph_recipe(self, col)
        for index, graph in enumerate(graphs):
            validate_graph(graph, recipe, index)
        self.molecule_graphs[col] = graphs

    # Drop the data with the empty graph
    def drop_none_graph(self):
        _masks = []
        for col in self.molecule_columns:
            _masks.append(np.array([g.num_nodes > 0 for g in self.molecule_graphs[col]]))
        _masks = np.stack(_masks)
        mask = np.all(_masks,axis=0)
        self.masking(mask)

    # Drop the data with the nan target
    def drop_nan_value(self):
        mask = np.array(~torch.isnan(torch.as_tensor(self.target_value)).all(dim=1))
        self.masking(mask)

    # Mask the data
    def masking(self,mask):
        mask = np.asarray(mask, dtype=bool)

        base_len = len(mask)
        for col in self.molecule_columns:
            if len(self._molecule_smiles[col]) != base_len:
                raise ValueError(
                    f"Length mismatch before masking for {col}: "
                    f"_molecule_smiles={len(self._molecule_smiles[col])}, mask={base_len}."
                )
            if len(self.molecule_graphs[col]) != base_len:
                raise ValueError(
                    f"Length mismatch before masking for {col}: "
                    f"molecule_graphs={len(self.molecule_graphs[col])}, mask={base_len}."
                )

        self.target_value = self.target_value[mask]
        if hasattr(self, 'original_row_indices'):
            if len(self.original_row_indices) != base_len:
                raise ValueError(
                    f"Length mismatch before masking for original_row_indices: "
                    f"rows={len(self.original_row_indices)}, mask={base_len}."
                )
            self.original_row_indices = self.original_row_indices[mask]
        if self.set is not None:
            self.set = self.set[mask]

        for col in self.molecule_columns:
            self._molecule_smiles[col] = self._molecule_smiles[col][mask]
            self.molecule_smiles[col] = self._molecule_smiles[col].copy()
            if col in self.valid_smiles:
                self.valid_smiles[col] = self.valid_smiles[col][mask]
            self.molecule_graphs[col] = [g for i,g in enumerate(self.molecule_graphs[col]) if mask[i]]

        if hasattr(self, 'numeric_inputs') and self.numeric_inputs is not None:
            for col in self.numeric_inputs:
                self.numeric_inputs[col] = self.numeric_inputs[col][mask]
        self._mask_additional_features(mask)

    def _mask_additional_features(self, mask):
        """Apply the central row mask to model-specific per-row features."""
        return None

    # drop the invalid data and initialize the dataset
    def prepare_dataset(self,temp=False):
        self.drop_none_graph()
        if not temp:
            self.drop_nan_value()
        self.whole_dataset = self.init_dataset()

    def init_dataset(self):
        if len(self.numeric_input_columns) > 0:
            if hasattr(self, 'numeric_inputs') and self.numeric_inputs is not None:
                if len(self.numeric_input_columns) != len(self.numeric_inputs):
                    raise ValueError(f"Please provide the numeric input for {self.numeric_input_columns}.")
                numeric_inputs = self.numeric_inputs
            else:
                raise ValueError(
                    f"Numeric inputs {self.numeric_input_columns!r} were not materialized before dataset creation. "
                    "Reload the CSV and verify the configured numeric columns."
                )
        else:
            numeric_inputs = None
        return self.dataset(graphs = self.molecule_graphs,
                            smiles = self._molecule_smiles,
                            target = self.target_value,
                            numeric_inputs = numeric_inputs,
                            row_indices = self.original_row_indices,)
        
    # Split the dataset into train, val, and test. if the set is given, the dataset will be split according to the set
    def split_data(self):
        import copy

        def _subset_dataset(indices):
            if hasattr(self.whole_dataset, 'get_subDataset'):
                return self.whole_dataset.get_subDataset(indices)
            dataset = copy.deepcopy(self.whole_dataset)
            if hasattr(dataset, 'subDataset'):
                dataset.subDataset(indices)
                return dataset
            raise AttributeError("Dataset does not support subsetting methods")

        requested_strategy = self.config.get("split_strategy", "auto")
        strategy = (
            "predefined"
            if requested_strategy == "auto" and self.set is not None
            else "random"
            if requested_strategy == "auto"
            else requested_strategy
        )
        if strategy == "random":
            validate_automatic_split_size(self.config['DATA_PATH'], len(self.whole_dataset))

            indices = np.arange(len(self.whole_dataset))
            train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=self.random_seed)
            train_idx, val_idx = train_test_split(train_idx, test_size=1/9, random_state=self.random_seed)
        elif strategy == "predefined":
            if self.set is None:
                raise ValueError(
                    "split_strategy='predefined' requires a CSV 'set' column with "
                    "train/val/test labels."
                )
            validate_set_labels(self.config['DATA_PATH'], list(self.set))
            train_idx = [i for i,s in enumerate(self.set) if s=="train"]
            val_idx = [i for i,s in enumerate(self.set) if s=="val"]
            test_idx = [i for i,s in enumerate(self.set) if s=="test"]
        elif strategy == "scaffold":
            scaffold_column = self.config.get(
                "scaffold_column", self.molecule_columns[0]
            )
            if scaffold_column not in self.molecule_columns:
                raise ValueError(
                    f"scaffold_column {scaffold_column!r} is not one of configured "
                    f"molecule_columns {self.molecule_columns!r}."
                )
            (train_idx, val_idx, test_idx), _ = scaffold_split_indices(
                self._molecule_smiles[scaffold_column],
                seed=self.random_seed,
                include_chirality=self.config.get(
                    "scaffold_include_chirality", False
                ),
            )
        else:
            raise ValueError(
                f"Unknown split_strategy {requested_strategy!r}. Use 'auto', "
                "'random', 'predefined', or 'scaffold'."
            )

        if (
            self.config.get("MODEL_PATH")
            and hasattr(self, "molecule_columns")
            and hasattr(self, "_molecule_smiles")
            and hasattr(self, "original_row_indices")
        ):
            scaffold_column = self.config.get(
                "scaffold_column", self.molecule_columns[0]
            )
            if scaffold_column not in self.molecule_columns:
                raise ValueError(
                    f"scaffold_column {scaffold_column!r} is not one of configured "
                    f"molecule_columns {self.molecule_columns!r}."
                )
            write_split_report(
                self.config,
                strategy,
                (train_idx, val_idx, test_idx),
                self.original_row_indices,
                self.target_value,
                self.target,
                self._molecule_smiles[scaffold_column],
                scaffold_column,
                include_chirality=self.config.get(
                    "scaffold_include_chirality", False
                ),
            )

        self._fit_and_transform_targets(train_idx)
        self.train_dataset = _subset_dataset(train_idx)
        self.val_dataset = _subset_dataset(val_idx)
        self.test_dataset = self.val_dataset if len(test_idx) == 0 else _subset_dataset(test_idx)
        self._output().info(
            "[Data] Dataset split sizes: "
            f"train={len(self.train_dataset)}, "
            f"validation={len(self.val_dataset)}, "
            f"test={len(self.test_dataset)}."
        )

    def _fit_and_transform_targets(self, train_idx):
        scope = self.config.get('target_scaler_fit_scope', 'train')
        fit_values = self.target_value if scope == 'all' else self.target_value[train_idx]
        if scope == 'all':
            warnings.warn(
                "target_scaler_fit_scope='all' fits target scaling on validation/test rows and may leak "
                "their statistics. Use 'train' for leakage-free training.",
                UserWarning,
                stacklevel=2,
            )
        self.scaler.fit(fit_values)
        transformed = self.scaler.transform(self.target_value)
        self.target_value = torch.as_tensor(transformed, dtype=torch.float32)
        self.whole_dataset.target = self.target_value
    
    # Initialize the dataloader
    def init_Dataloader(self,dataset=None,partial_train=1.0, shuffle=False):
        import copy

        def _subset_for_loader(src_dataset, indices):
            if hasattr(src_dataset, 'get_subDataset'):
                return src_dataset.get_subDataset(indices)
            subset = copy.deepcopy(src_dataset)
            if hasattr(subset, 'subDataset'):
                subset.subDataset(indices)
                return subset
            raise AttributeError("Dataset does not support subsetting methods")

        if dataset is None:
            dataset = self.whole_dataset
        if partial_train and partial_train<1.0 and partial_train>0:
            if self._partial_train_rng is None:
                subset_idx = np.random.choice(
                    len(self.train_dataset),
                    int(len(self.train_dataset) * partial_train),
                    replace=False,
                )
            else:
                subset_idx = self._partial_train_rng.choice(
                    len(self.train_dataset),
                    int(len(self.train_dataset) * partial_train),
                    replace=False,
                )
            dataset = _subset_for_loader(dataset, subset_idx)
            self._output().info(
                f"[Data] Partial training subset contains {len(dataset)} rows."
            )

        worker_init_fn = (
            seed_data_loader_worker
            if self.config.get("random_seed") is not None else None
        )
        return DataLoader(dataset, 
                        batch_size=self.config.get('batch_size',32), 
                        shuffle=shuffle,
                        collate_fn=self.dataset.collate, 
                        pin_memory=self.config.get('pin_memory',True),
                        num_workers=self.config.get('num_workers',0),
                        worker_init_fn=worker_init_fn)

    # (main function) Provide the dataloader for the training, validation, and test
    def get_Dataloaders(self, temp=False):
        self.prepare_dataset(temp=temp)
        if temp:
            loader = self.init_Dataloader(shuffle=False)
            return loader
        else:
            self.split_data()
            self.train_loader = self.init_Dataloader(self.train_dataset,partial_train=self.config.get('partial_train',1.0),shuffle=self.config.get('shuffle',True))
            self.val_loader = self.init_Dataloader(self.val_dataset)
            self.test_loader = self.init_Dataloader(self.test_dataset)

            return self.train_loader, self.val_loader, self.test_loader
  

class MolDataManager_withSolv(MolDataManager):
    """The class for the management of the molecular data with the solvent, preparing the dataset, and dataloader"""
    def __init__(self, config):
        super().__init__(config)
        self.molecule_columns.append('solvent')
        self.molecule_graphs={col:[] for col in self.molecule_columns}

    def import_others(self):
        self.graph_type = 'mol'
        self.gg =MolGraphGenerator()
        self.gg_solv =MolGraphGenerator()
        self.dataset =GraphDataset_withSolv
        self.unwrapper = self.dataset.unwrapper
        
    def init_temp_data(self,smiles,solvents,**kwargs):
        return super().init_temp_data(smiles, solvents, **kwargs)

    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.molecule_graphs['solvent'], self.target_value, self._molecule_smiles['compound'], self._molecule_smiles['solvent'])

def hash(s):
    md5_hash = hashlib.md5()
    md5_hash.update(s.encode('utf-8'))
    hex_digest = md5_hash.hexdigest()
    decimal_hash = int(hex_digest, 16) % (10**12+7)
    
    return decimal_hash
