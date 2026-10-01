#!/bin/bash
# Usage: ./run_simulation.sh [estimator] [cores] [use_sl] [do_mpi] [R] [target_times] [output_dir]
#   estimator:    'tmle' (Super Learner / GLM path) or 'tmle-lstm'
#   cores:        'tmle': cores used within each replicate; 'tmle-lstm': replicates run in parallel
#   use_sl:       TRUE = Super Learner, FALSE = GLM / multinomial logistic regression
#   R:            number of simulation replicates (default 100); finished replicates in output_dir are skipped
#   target_times: 'all' (t = 1..36) or a comma-separated list, e.g. '6,12,18,24,30,36' ('tmle' only)
#   output_dir:   directory for per-replicate results (default ./outputs/YYYYMMDD)

ESTIMATOR=${1:-"tmle"}
CORES=${2:-2}
USE_SL=${3:-"TRUE"}
DO_MPI=${4:-"FALSE"}
R=${5:-100}
TARGET_TIMES=${6:-"all"}
OUTPUT_DIR=${7:-"./outputs/$(date +%Y%m%d)"}

# Increase stack size to prevent C stack overflow (ignored if not permitted)
ulimit -s unlimited 2>/dev/null || true

# The Python environment is only needed for the LSTM estimator
if [ "$ESTIMATOR" = "tmle-lstm" ]; then
  source ./myenv/bin/activate
fi

echo "Running simulation with: estimator=$ESTIMATOR, cores=$CORES, use_SL=$USE_SL, do_MPI=$DO_MPI, R=$R, target_times=$TARGET_TIMES, output_dir=$OUTPUT_DIR"
Rscript simulation.R "$ESTIMATOR" "$CORES" "$USE_SL" "$DO_MPI" "$R" "$TARGET_TIMES" "$OUTPUT_DIR"
