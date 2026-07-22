# Public API implementation instructions

## Scope

This directory owns the canonical implementations of training, optimization,
legacy grid search, CLI dispatch, and public exception categories.

## Compatibility

- Keep `D4CMPP2.train`, `D4CMPP2.optimize`, and `D4CMPP2.grid_search` stable.
- Root `_main.py`, `cli.py`, `optimize.py`, `grid_search.py`, and `exceptions.py`
  are module-alias shims. Do not put implementation logic back into them.
- Preserve the old module aliases until a separately approved deprecation removes them.
- Internal package code should import canonical modules from `D4CMPP2.src.api`.

## Validation

- Exercise canonical imports and every compatibility module path.
- Verify module-level monkeypatches through old paths affect canonical globals.
- Run CLI, wheel/sdist, full PyG, and static checks after moving API code.

## Change history

- 2026-07-21: Moved root API implementations here while retaining root module aliases.
- 2026-07-21: Optimization calls without an explicit `optimization_path` create
  unique timestamped run directories. Resuming requires reusing an explicit path.
