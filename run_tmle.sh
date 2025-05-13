#!/bin/bash

# Script specifically for running the standard tmle estimator with all fixes
# This estimator had the most issues with vector length mismatches

# Parameters
estimator="tmle"  # Use standard tmle estimator
cores="1"         # Use single core
use_sl="TRUE"     # Use SuperLearner
do_mpi="FALSE"    # Don't use MPI

echo "Running standard TMLE simulation with all fixes applied..."
echo "Estimator: $estimator"
echo "Cores: $cores"
echo "Use SuperLearner: $use_sl"
echo "Use MPI: $do_mpi"

# Run the simulation using our updated run_simulation.R script
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