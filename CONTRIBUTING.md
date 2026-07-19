# Contributing

Contributions should make the implementation easier to reproduce, evaluate,
extend, or use.

## Before opening a pull request

1. Search existing issues and pull requests.
2. Open an issue before a substantial behavioral or architectural change.
3. Keep each pull request focused on one change.
4. Add or update tests and documentation when behavior changes.
5. Report the exact checks run, environment, and any skipped tests.

Useful contributions include bug fixes, reproducible benchmark artifacts,
dataset/model adapters, full-catalog evaluation, profiling, and precise
documentation corrections. Trivial or generated activity intended only to pad
contributor counts will be closed.

## Setup and checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m recommender.cli compare --config configs/synthetic_quick.yaml
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

Benchmark changes must follow
[`docs/reproducibility.md`](docs/reproducibility.md).

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
