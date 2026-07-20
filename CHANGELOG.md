# Changelog

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
- Supports Python 3.10-3.12, PyTorch `>=2.2,<3`, and PyG `>=2.8,<3`.

See the migration and compatibility section in `README.md` before using
DGL-era caches or saved models.
