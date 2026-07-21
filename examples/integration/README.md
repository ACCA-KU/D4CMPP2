# Heavy integration examples

These examples execute real public workflows rather than mocking training or
calling networks directly. They are intentionally excluded from the fast test
layer because they train and reload multiple models on CPU.

Run everything with self-cleaning temporary outputs:

```sh
python examples/integration/run_all.py
```

Keep generated models, transfer reports, and manifests for inspection:

```sh
python examples/integration/run_all.py --keep-output integration-output
```

Run one workflow:

```sh
python examples/integration/run_all.py --only transfer-learning
```

The workflows are:

- `solvent-analyzer`: all five solvent models, from one-epoch public training
  through saved artifacts and row-preserving Analyzer prediction.
- `transfer-learning`: representative GCN, GCNwS, and ISAT source/target runs;
  compatible backbone loading, resized-head reporting, saved-model reload, and
  Analyzer prediction.

The data is generated locally, no download is performed, and CPU is used by
default. The command exits nonzero if any workflow fails.
