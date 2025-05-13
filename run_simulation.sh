#!/bin/bash

# Simple wrapper script to run the simulation
estimator="${1:-tmle-lstm}"
cores="${2:-1}"
use_sl="${3:-TRUE}"
do_mpi="${4:-FALSE}"

# Run the simulation with all fixes applied through run_simulation.R
echo "Running simulation with: estimator=$estimator cores=$cores use_sl=$use_sl do_mpi=$do_mpi"
Rscript run_simulation.R "$estimator" "$cores" "$use_sl" "$do_mpi"

# Check for completion
if [ $? -eq 0 ]; then
  output_dir="./outputs/$(date +%Y%m%d)/"
  echo "Simulation completed successfully."
  echo "You can now plot the results with:"
  echo "Rscript long_sim_plots.R '$output_dir'"
else
  echo "Simulation failed. Check the error messages above."
fi