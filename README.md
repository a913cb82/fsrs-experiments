# FSRS Divergence Experiments

A simulator and analysis toolset for studying how fitted FSRS parameters diverge from ground truth under various conditions, such as different target retentions, burn-in periods, and retention schedules.

## Files

- `src/simulate_fsrs.py`: Core simulation engine.
- `src/plot_fsrs_divergence.py`: CLI tool to run and plot simulations.

## Requirements

Python 3.10+ and dependencies in `pyproject.toml`:

```bash
pip install .
```

## Usage

### Running a basic simulation
Run a simulation for a fixed number of days with a daily review limit.
```bash
fsrs-sim --days 365 --reviews 200 --retention 0.9
```

### Using a Burn-in Period
Run a simulation where the first 30 days use ground truth parameters for scheduling. After 30 days, FSRS parameters are fitted to the history and used for all subsequent scheduling.
```bash
fsrs-sim --days 365 --burn-in 30
```

### Plotting Divergence
Compare multiple configurations (different day limits, burn-in periods, or retention schedules) in a single plot. You can run multiple repeats per configuration to average the results.
```bash
fsrs-plot --days 100 200 --burn-ins 0 30 --retentions 0.85 0.95 --repeats 5
```
The resulting graph is saved as `forgetting_curve_divergence.png`. It shows the average forgetting curve across repeats and reports the average RMSE and KL divergence metrics.

## Development

This project uses `ruff` for linting and formatting, and `mypy` for type checking. Configuration for these tools can be found in `pyproject.toml`.

### Pre-commit Hooks

To ensure code quality, pre-commit hooks are configured. To set them up, run:

```bash
pip install pre-commit
pre-commit install
```

You can run the hooks manually on all files with:

```bash
pre-commit run --all-files
```
