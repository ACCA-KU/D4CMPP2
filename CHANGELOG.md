# Changelog

This project follows semantic versioning for the Python package. Saved-model
and config compatibility is documented separately because a package release
does not automatically invalidate existing model assets.

## 1.0.0

- Stabilized all solvent-aware model pipelines from training and persistence to
  saved-model Analyzer prediction while retaining legacy Dataset input aliases.
- Added public end-to-end integration examples for solvent Analyzer and transfer
  learning workflows, together with a single-command integration runner.
- Improved transfer-learning config resolution and reporting for compatible,
  skipped, and shape-mismatched parameters.
- Added visible CSV and graph-cache preparation status plus opt-in fast paths for
  repeatedly reviewed data and verified graph caches.
- Removed repeated Dataset tensor-copy warnings without changing target tensor
  dtype, detachment, or copy semantics.
- Moved the main training, optimization, CLI, and exception implementations into
  `D4CMPP2.src.api`, retaining thin aliases for historical public module paths.

## 0.4.0

- Replaced the primary DGL graph backend with PyTorch Geometric.
- Added validated CPU training, resume, transfer, optimization, Analyzer, ISA,
  batch-inference, and reproducibility workflows.
- Added package/CLI entry points, wheel packaging, CI, and clearer errors.
- Preserved legacy public names and saved-model adapters where documented.
- Prevented ISATPN single-fragment variance regularization from producing NaN
  while preserving the existing unbiased variance for larger score sets.
- Corrected ISA bare single-atom self-loop edge features and added cache
  validation for edge-index/feature-row alignment.
- Fixed solvent-model training validation by exposing canonical compound and
  solvent batch keys while retaining every legacy Dataset alias.
- Accepted the documented explicit `molecule_columns=["compound", "solvent"]`
  solvent configuration without duplicating the solvent CSV column.
- Saved the fitted target scaler even when `save_prediction=False`, so every
  completed non-identity model remains loadable by Analyzer.
- Restored solvent Analyzer prediction for both named `compound`/`solvent`
  inputs and the historical two-positional-input form.
- Preserved provenance for paths derived before a `TRANSFER_PATH` config merge,
  allowing public transfer runs to resolve saved and target data paths.
- Supports Python 3.10-3.12, PyTorch `>=2.2,<3`, and PyG `>=2.8,<3`.

See the migration and compatibility section in `README.md` before using
DGL-era caches or saved models.
