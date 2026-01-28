# FSRS Divergence & Optimal Retention Research

A research-oriented toolset for studying FSRS parameter stability and identifying mathematically optimal target retentions using real study history.

## Research Focus

This repository is primarily focused on interactive analysis via **Jupyter Notebooks**, supported by a high-performance simulation engine.

### Key Notebooks

- **[Optimal Retention Analysis](notebooks/Review_Time_Analysis.ipynb)**: 
  - Analyzes the relationship between study time, retrievability, and user ratings.
  - Fits predictive models (Lasso) to your real study data to estimate cost-per-review.
  - Identifies your **Mathematically Optimal Target Retention** by simulating 180 days of study under various strategies and correcting for the "volume" of unseen cards.
- **[Divergence Exploration](notebooks/Divergence_Exploration.ipynb)**: 
  - Explores how fitted FSRS parameters diverge from ground truth over time.
  - Studies the impact of different target retentions, burn-in periods, and variable retention schedules.

## Installation & Setup

High-performance Rust integration is essential for efficient parameter fitting during simulations.

### 1. Clone and Install Dependencies
```bash
git clone <this-repo-url>
cd fsrs_experiments
pip install -e .
```

### 2. Setup Rust Backend
You must have `rustc` and `cargo` installed.
```bash
# 1. Clone and patch the FSRS Rust repositories
./scripts/setup_rust_backend.sh

# 2. Build and install the optimized bindings
cd fsrs-rs-python-repo
pip install maturin
maturin develop --release
cd ..
```

## Usage

### Interactive Research
Start the research environment to run the analysis notebooks:
```bash
jupyter notebook
```

### Advanced CLI Usage
The primary CLI tool `src/plot_fsrs_divergence.py` can be used for large-scale batch processing and population-level forgetting curve visualization.

```bash
python src/plot_fsrs_divergence.py \
    --days 30 90 \
    --retentions 0.85 0.95 \
    --repeats 10 \
    --concurrency 8
```

## Core Features

- **Personalized Modeling**: Infers your specific study habits (rating probabilities and study cost) from your Anki history (`collection.anki2`).
- **Duration-Based Simulations**: Supports simulations constrained by daily study time (e.g., 30 minutes/day) with realistic review prioritization.
- **High Performance**: Native Rust backend support for ~20x faster parameter fitting.
- **Fair Baseline Normalization**: Corrects total recall metrics by accounting for the latent knowledge in unstudied "new" cards.

## Development

This project uses `ruff` for linting/formatting and `mypy` for type checking.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```