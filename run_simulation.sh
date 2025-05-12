#!/bin/bash

# Simple wrapper script to run the simulation
estimator="${1:-tmle-lstm}"
cores="${2:-1}"
use_sl="${3:-TRUE}"
do_mpi="${4:-FALSE}"

# Apply fixes first
echo "Applying fixes to simulation code..."
Rscript fix_simulation.R

# Create output directory
output_dir="./outputs/$(date +%Y%m%d)/"
mkdir -p "$output_dir"
echo "Created output directory: $output_dir"

# Run the simulation
echo "Running simulation with: estimator=$estimator cores=$cores use_sl=$use_sl do_mpi=$do_mpi"
Rscript simulation.R "$estimator" "$cores" "$use_sl" "$do_mpi"

# Check for completion
if [ $? -eq 0 ]; then
  echo "Simulation completed successfully."
  echo "You can now plot the results with:"
  echo "Rscript long_sim_plots.R '$output_dir'"
else
  echo "Simulation failed. Check the error messages above."
fi