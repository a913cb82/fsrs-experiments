# FSRS Divergence Experiments

A simulator and analysis toolset for studying how fitted FSRS parameters diverge from ground truth under various conditions, such as different target retentions, burn-in periods, and retention schedules.

## Core Features

- **Population Analysis**: Visualizes **Aggregate Deck Forgetting Curves**, comparing "Actual Knowledge" (Nature) vs. "Predicted Knowledge" (Model) across your entire deck.
- **Seeded Simulations**: Initialize simulations using your real Anki review history (`collection.anki2`).
- **High Performance**: Native Rust backend support for ~20x faster parameter fitting.
- **Statistical Rigor**: Supports multiple repeats with averaging and standard deviation shading for stable results.
- **Advanced Metrics**: Calculates per-card RMSE and KL Divergence averaged across the population.

## Requirements

Python 3.10+ and dependencies in `pyproject.toml`:

```bash
pip install .
```

## Usage

### Plotting Divergence
The primary tool is `src/plot_fsrs_divergence.py`. It runs a cross-product of all provided arguments.

```bash
python src/plot_fsrs_divergence.py \
    --days 30 60 \
    --retentions 0.85 0.95 \
    --repeats 5 \
    --concurrency 8
```

### Advanced Options

- **Seed from Anki**: Load real review history and filter by deck configuration.
  ```bash
  --seed-history collection.anki2 --deck-config "Default"
  ```
- **Custom Ground Truth**: Set the "natural" parameters your brain is simulated to follow.
  ```bash
  --ground-truth "0.4,1.2,3.2,15.7,7.2,0.5,1.5,0.005,1.5,0.1,1.0,1.9,0.1,0.3,2.3,0.2,3.0,0.5,0.7,0.0,0.15"
  ```
- **Burn-in Period**: accumulating data with ground-truth scheduling before the model starts fitting.
  ```bash
  --burn-ins 30 60
  ```

## High-Performance Backend (Recommended)

This project supports a Rust-powered backend which is significantly faster than the default PyTorch optimizer.

1. **Install Rust**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. **Setup and Patch**:
   ```bash
   ./scripts/setup_rust_backend.sh
   ```
3. **Build Bindings**:
   ```bash
   cd fsrs-rs-python-repo && maturin develop --release
   ```

## Development

This project uses `ruff` for linting/formatting and `mypy` for type checking.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```