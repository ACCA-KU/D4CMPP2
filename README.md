# D4CMPP2

D4CMPP2 is a PyTorch Geometric package for molecular-property regression. It
supports ordinary molecular graphs, compound/solvent pairs, and interpretable
ISA models through one training and inference API.

## Requirements and installation

Supported runtime:

- Python 3.10-3.12
- PyTorch `>=2.2,<3`
- PyTorch Geometric `>=2.8,<3`
- Linux and Windows
- CPU, or CUDA with a matching PyTorch build

Install the appropriate PyTorch build for the machine first, then install
D4CMPP2:

```sh
python -m pip install D4CMPP2
```

Notebook extras are optional:

```sh
python -m pip install "D4CMPP2[notebook]"
```

DGL is not installed or loaded. DGL graph caches cannot be renamed or reused
as PyG caches.

## Supported models

| Family | Network IDs |
|---|---|
| General | `GCN`, `MPNN`, `DMPNN`, `AFP`, `GAT` |
| With solvent | `GCNwS`, `MPNNwS`, `DMPNNwS`, `AFPwS`, `GATwS` |
| ISA | `GC`, `ISAT`, `ISATPN` |

`AGC`, `MegNet`, `MegNetwS`, `ALIGNN`, and `ALIGNNwS` are not part of the
PyG-only release.

## Input data

A general model needs a `compound` SMILES column and one or more numeric target
columns. A `wS` model additionally needs a `solvent` SMILES column:

```csv
compound,solvent,Solubility
CCO,O,-0.31
CCN,CO,-0.18
```

Useful optional columns and arguments:

- `set`: predefined `train`, `val`, and `test` labels.
- `numeric_input_columns`: additional finite numeric model inputs.
- `molecule_columns`: molecule columns in their model input order.
- `explicit_h_columns`: molecule columns that require explicit hydrogens.

Without a `set` column, the default split is a seeded 80/10/10 random split.
Invalid SMILES, missing columns, non-numeric inputs, all-NaN targets, and
misaligned arrays fail with the affected path, column, value, or row.

## Quickstart

The package includes a small `test.csv`. Run a two-epoch CPU smoke training:

```sh
d4cmpp2 --data test --target Abs --network GCN --device cpu --epoch 2
```

Equivalent Python:

```python
from D4CMPP2 import train

model_path = train(
    data="test",
    target=["Abs"],
    network="GCN",
    device="cpu",
    max_epoch=2,
    batch_size=4,
)
print(model_path)
```

Runnable examples for training, solvent and ISA models, saved-model modes,
inference, uncertainty, optimization, callbacks, custom networks, and the CLI
are indexed in [`examples/README.md`](examples/README.md).

For GPU training, use `device="cuda:0"` or `--device cuda:0`. D4CMPP2 validates
the requested index and does not silently fall back to CPU.

### Solvent and ISA

```python
from D4CMPP2 import train

solvent_model = train(
    data="solvent_data.csv",
    target=["Solubility"],
    network="GCNwS",
    molecule_columns=["compound", "solvent"],
    device="cpu",
)

isa_model = train(
    data="Aqsoldb",
    target=["Solubility"],
    network="ISATPN",
    sculptor_index=(6, 2, 0),
    device="cpu",
)
```

`sculptor_index=(split, combine, absorb)` controls ISA fragmentation. Existing
fragment rules and saved `functional_group.csv` files are part of the model
contract.

## Training behavior

The most important defaults are:

| Key | Default |
|---|---|
| `batch_size` | `256` |
| `max_epoch` | `2000` |
| `optimizer` | `"Adam"` |
| `learning_rate` | `0.001` |
| `weight_decay` | `0.0005` |
| `device` | `"cuda:0"` |
| `scaler` | `"standard"` |
| `verbose` | `True` |

Configuration is resolved into an isolated working copy. Historical precedence
is retained for numerical compatibility:

| Mode | Lower to higher precedence |
|---|---|
| Scratch | defaults, API/CLI, registry |
| Full resume | saved config, explicit resume overrides |
| `LOAD_PATH` continue | saved config, current defaults/API |
| Transfer | source config, current defaults/API, registry |
| Optimization trial | resolved base, trial values |

Unknown keys remain available to legacy configs and custom implementations.
Resolved provenance is written to the run manifest without changing
`config.yaml`.

### Splitting and reproducibility

`split_strategy` accepts:

- `"auto"`: use `set` when present, otherwise random 80/10/10.
- `"random"`: always use row-level random splitting.
- `"predefined"`: require the `set` column.
- `"scaffold"`: keep RDKit Murcko-scaffold groups within one split.

Normal runs write `split_report.json` and `split_assignments.csv`.

```python
model_path = train(
    data="Aqsoldb",
    target=["Solubility"],
    network="GCN",
    random_seed=42,
    split_random_seed=42,
    deterministic_algorithms=True,
    device="cpu",
)
```

`random_seed` controls Python, NumPy, torch/CUDA, shuffle, DataLoader workers,
dropout, and partial-training selection. `split_random_seed` controls only the
data split. Deterministic algorithms may reduce GPU performance or reject an
unsupported operation.

### Resume, continue, and transfer

The three saved-model modes are mutually exclusive:

- `RESUME_PATH` or `--resume`: exact full-state resume from
  `checkpoints/latest.ckpt`.
- `LOAD_PATH` or `--load`: load `final.pth` but reset optimizer, schedulers,
  epoch, and RNG.
- `TRANSFER_PATH` or `--transfer`: copy compatible same-name/same-shape
  parameters into a new model.

Full checkpoints preserve optimizer, both schedulers, epoch, best metric,
early-stopping state, and RNG state. Transfer writes `transfer_report.json`
with loaded and skipped parameters plus the source weight hash.

### Outputs and caches

Training returns the model directory, normally below `./_Models/`. Important
artifacts include:

- `config.yaml`, `network.py`, `final.pth`, `scaler.pkl`
- `network_identity.json`, `model_summary.txt`
- `checkpoints/latest.ckpt`, `checkpoints/best.ckpt`
- `runs/<run_id>/run_manifest.json`
- `result/prediction.csv`, metrics, curves, and plots
- data-quality and split reports

Graph caches are atomic, fingerprinted PyG schema-v2 files. Identity includes
ordered SMILES, feature dimensions, explicit-hydrogen settings, and ISA rules.
The cache policies are:

- `"v2"`: validate the current cache and fail clearly if it is incompatible.
- `"regenerate"`: rebuild an invalid v2 cache from the source CSV.
- `"legacy"`: explicitly load a compatible PyG v1 cache with a warning.

DGL caches are preserved but never loaded.

## Prediction and analysis

```python
from D4CMPP2 import Analyzer

analyzer = Analyzer(model_path, device="cpu", save_result=False)
print(analyzer.predict(["CCO", "CCN"]))
```

`Analyzer` reads the saved config and selects the general, solvent, or ISA
implementation. It requires `config.yaml`, `network.py`, and `final.pth`;
non-identity scaling also requires `scaler.pkl`, and ISA requires the saved
`functional_group.csv`. Load pickle-based artifacts only from trusted model
folders.

Use row-preserving inference when duplicates or invalid inputs matter:

```python
result = analyzer.predict_rows(
    compound=["CCO", "CCO", "not-a-smiles"],
)
print(result.to_dataframe())
```

Every row retains its original index, inputs, prediction, status, and error.
Solvent and numeric-input models require every saved input column with equal
length:

```python
result = analyzer.predict_rows(
    compound=["CCO", "CCN"],
    solvent=["O", "CO"],
    temperature=[298.0, 310.0],
)
```

CSV inference preserves source rows and commits the output atomically:

```python
output_path = analyzer.predict_csv(
    "molecules.csv",
    "molecules_prediction.csv",
    uncertainty_samples=30,
    uncertainty_seed=42,
)
```

`predict_uncertainty()` uses MC dropout and `Analyzer.predict_ensemble()` uses
compatible independently trained models. Their standard deviations are model
dispersion estimates, not calibrated prediction intervals.

ISA analyzers also provide `analyze_rows()` and `plot_analysis()`. Fragment
indices are validated against atom alignment before scores or features are
returned.

## Experiment comparison and optimization

```python
from D4CMPP2 import compare_experiments

leaderboard = compare_experiments(
    ["./_Models"],
    output_path="leaderboard.csv",
    metric="val_rmse",
)
```

Ranking is calculated separately per target. Legacy, failed, and interrupted
runs remain visible without borrowing another run's metrics.

For new hyperparameter searches, use the model-aware API:

```python
import D4CMPP2

result = D4CMPP2.optimize(
    data="Aqsoldb.csv",
    target=["Solubility"],
    network="GCN",
    HP=["hidden_dim", "dropout"],
    optimize_strategy="bayesian",
    n_trials=20,
    device="cpu",
)
print(result.best_params, result.best_model_path)
```

`HP=None` uses the model's default optimization space. A key list uses declared
model ranges; a dictionary supplies categorical values or numeric ranges.
Supported strategies are `"bayesian"` and `"grid"`. Atomic JSON/CSV summaries
and trial directories allow compatible searches to resume.

The legacy `grid_search()` API remains available with its historical `None`
return value and continue-on-trial-error behavior.

## Callbacks and custom networks

Observation-only training callbacks are runtime objects and are not saved:

```python
from D4CMPP2 import train
from D4CMPP2.src.TrainManager.callbacks import EventHistory

history = EventHistory()
model_path = train(
    data="test",
    target=["Abs"],
    network="GCN",
    device="cpu",
    callbacks=[history],
)
```

Callbacks receive immutable run/epoch/validation/checkpoint/error events. A
callback failure stops training and preserves the original failure contract.

Custom models subclass `D4CMPP2.MolecularNetwork`, declare `model_name`,
`input_contract`, hyperparameters, and optionally `compute_loss()`, then call:

```python
D4CMPP2.register_network(CustomGCN, data_contract="molecule")
```

Available data contracts are `"molecule"`, `"solvent"`, and `"isa"`. Define
the class at module scope in an importable file. Training copies that source to
the model directory as `network.py`; reload and transfer prefer the saved
snapshot. See `examples/custom_network.py` for a complete implementation.

## CLI

```sh
python -m D4CMPP2 --help
d4cmpp2 --help
```

Target columns are comma-separated. `--quiet` hides routine information and
progress but not warnings or errors. `--cuda 0` remains a deprecated alias for
`--device cuda:0`.

## Migration and compatibility

- Public `train`, `grid_search`, `Analyzer`, `Segmentator`, and `Data` names
  remain available.
- New runs fit the target scaler on training rows by default. Old configs
  without the policy retain historical full-data scaling with a warning.
- The loader prefers the saved model folder's `network.py`, config, weights,
  scaler, and ISA rules over obsolete absolute paths.
- The historical `ISATPM` saved name is supported through the `ISATPN`
  compatibility adapter.
- Keep DGL-era models and caches with their original environment for archival
  reproduction. Automatic conversion is not provided.
- Package version and saved-model/config contract version are separate.

Public API or saved-asset compatibility breaks require a major release.
Deprecations should include a warning and a migration path.

## Troubleshooting

- `torch_geometric` import failure: install torch and PyG builds compatible
  with the same Python and CPU/CUDA runtime.
- CUDA unavailable or index out of range: select an existing `cuda:N` or use
  `device="cpu"` explicitly.
- Missing target/SMILES column: compare the reported available columns with the
  configured target and molecule columns.
- Invalid SMILES: fix or remove the reported rows; related arrays share one
  alignment mask.
- Cache shape or recipe mismatch: use the source CSV with
  `graph_cache_policy="regenerate"` after reviewing the diagnostic.
- Model path missing or ambiguous: pass the exact saved-model directory.
- Old model fails to load: preserve the complete folder and original
  environment; report config, package versions, and the error without sharing
  private data.

## Repository maintenance

Implementation contracts, compatibility constraints, validation commands, and
change notes live in the nearest `AGENTS.md` file within each source area.
Read the workspace and nearest folder instructions before modifying code.
