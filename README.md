# FSRS Divergence Experiments

A simulator and analysis toolset for studying how fitted FSRS parameters diverge from ground truth under various conditions, such as different target retentions, burn-in periods, and retention schedules.

## Files

- `simulate_fsrs.py`: Core simulation engine. It mimics user behavior (recall/forget) using ground truth FSRS parameters and can optionally use fitted parameters for scheduling after a "burn-in" period.
- `plot_fsrs_divergence.py`: CLI tool to run multiple simulation configurations and plot their resulting forgetting curves against the ground truth curve to visualize divergence.

## Requirements

- Python 3.10+
- PyTorch
- Matplotlib
- Pandas
- Tqdm
- FSRS (`py-fsrs`)

## Usage

### Running a basic simulation
Run a simulation for a fixed number of days with a daily review limit.
```bash
python3 simulate_fsrs.py --days 365 --reviews 200 --retention 0.9
```

### Using a Burn-in Period
Run a simulation where the first 30 days use ground truth parameters for scheduling. After 30 days, FSRS parameters are fitted to the history and used for all subsequent scheduling.
```bash
python3 simulate_fsrs.py --days 365 --burn-in 30
```

### Plotting Divergence
Compare multiple configurations (different day limits, burn-in periods, or retention schedules) in a single plot.
```bash
python3 plot_fsrs_divergence.py --days 100 200 --burn-ins 0 30 --retention-schedules "5:0.7,1:0.9"
```
The resulting graph is saved as `forgetting_curve_divergence.png`.
