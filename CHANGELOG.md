# Changelog

- Remove repeated PyTorch tensor-copy warnings when Dataset targets are already
  tensors while preserving the prior detached, independent float32 copy semantics.
- Report CSV and graph-cache preparation progress, and add the opt-in
  `data_quality_report=False` and `validate_graph_cache=False` fast paths for
  previously reviewed data/caches while retaining required CSV and cache identity
  checks.
- Move large root API implementations into `D4CMPP2.src.api`; retain the five
  historical root modules as thin aliases so public imports, CLI, and module-level
  monkeypatch behavior remain compatible.

This project follows semantic versioning for the Python package. Saved-model
and config compatibility is documented separately because a package release
does not automatically invalidate existing model assets.

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
