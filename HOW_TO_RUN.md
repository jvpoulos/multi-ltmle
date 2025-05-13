# How to Run Simulations

This document provides instructions for running the multi-ltmle simulations with error fixes applied.

## Running Simulations with Error Fixes

Several error fixes have been implemented to address issues with different estimators. The main fixes handle:

1. GLM family parameters that were incorrectly specified as strings
2. Missing covariates in SL3 tasks
3. Vector length mismatches in the getTMLELong function
4. Invalid slice indices
5. Missing time series data
6. Missing weights_bin

### Using the Fixed Scripts

#### Option 1: Run directly with the shell script (recommended)

```bash
# Run with default parameters (tmle-lstm estimator)
./run_simulation.sh

# Run with specific parameters
./run_simulation.sh [estimator] [cores] [use_sl] [do_mpi]

# Example: Run standard tmle estimator with all fixes
./run_tmle.sh

# Example: Run tmle-lstm estimator with all fixes
./run_tmle_lstm.sh
```

#### Option 2: Run with R directly

```R
# Source the fix scripts
source("fix_simulation.R")
if(estimator == "tmle") source("makeshift_tmle.R")

# Run the simulation
source("simulation.R")
```

### Fix Scripts Overview

1. **fix_simulation.R**: Comprehensive fixes for all estimator types
   - Converts GLM family parameters from strings to function objects
   - Adds safety checks for missing covariates
   - Fixes time series and slice index errors
   - Adds fallbacks for missing weights

2. **makeshift_tmle.R**: Specific fixes for the standard tmle estimator
   - Monkey-patches getTMLELong to handle vector length mismatches
   - Ensures matrix dimensions are consistent
   - Provides fallbacks when errors occur

## Common Errors and Solutions

### "Error in args$family$family : $ operator is invalid for atomic vectors"
- **Fix**: Convert string family parameters to function objects
- **Script**: fix_simulation.R and makeshift_tmle.R (improved handling)

### "Error in task$Y : $ operator is invalid for atomic vectors"
- **Fix**: Add safety checks for missing covariates
- **Script**: fix_simulation.R

### "Error in getTMLELong(...) : number of items to replace is not a multiple of replacement length"
- **Fix**: Ensure matrix dimensions are consistent
- **Script**: makeshift_tmle.R

### "Error: Vector slice indices must match the dimension of the vector" or "Invalid slice indices"
- **Fix**: Add bounds checking and fallbacks for array indices
- **Script**: fix_simulation.R (improved for tmle-lstm in fix_lstm_slice_indices)

### "Warning: No time series found for L1,L2,L3"
- **Fix**: Add fallbacks for missing time series data
- **Script**: fix_simulation.R

### "Error in Q_star * weights_bin : object 'weights_bin' not found"
- **Fix**: Initialize weights_bin when missing
- **Script**: fix_simulation.R

### "A_hat_tmle values are uniformly 0.167"
- **Fix**: Add variation to treatment probabilities to avoid uniform values
- **Script**: makeshift_tmle.R

### "Standard errors are uniformly 0.001"
- **Fix**: Improve standard error calculation with more variation and data-driven approaches
- **Script**: fix_tmle_ic.R

## Plotting Results

After running a simulation, you can plot the results using:

```bash
Rscript long_sim_plots.R 'outputs/YYYYMMDD'
```

Replace 'YYYYMMDD' with the date folder where your simulation results are stored.

## Advanced Usage

For debugging purposes, you can run a minimal simulation:

```bash
Rscript minimal_simulation.R
```

For high-performance computing environments, use the .sb scripts:

```bash
sbatch simulation_gpu.sb
sbatch simulation_lstm_mpi.sb
```
