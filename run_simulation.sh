#!/bin/bash

# Default parameters
ESTIMATOR=${1:-"tmle-lstm"}
CORES=${2:-2}
USE_SL=${3:-"TRUE"}
DO_MPI=${4:-"FALSE"}

# Activate the Python environment
source ./myenv/bin/activate

# Run the simulation
echo "Running simulation with: estimator=$ESTIMATOR, cores=$CORES, use_SL=$USE_SL, do_MPI=$DO_MPI"
Rscript simulation.R $ESTIMATOR $CORES $USE_SL $DO_MPI