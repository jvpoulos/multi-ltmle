#!/bin/bash
# Usage: ./run_simulation.sh [estimator] [cores] [use_sl] [do_mpi] [R] [target_times] [output_dir] [replicates]
#   estimator:    'tmle' (Super Learner / GLM learners) or 'tmle-lstm' (LSTM learners); both use the same LTMLE core
#   cores:        cores used within each replicate (forked workers; for 'tmle-lstm' each runs single-threaded TensorFlow)
#   use_sl:       'tmle' only: TRUE = Super Learner, FALSE = GLM / multinomial logistic regression
#   R:            number of simulation replicates (default 100); finished replicates in output_dir are skipped
#   target_times: 'all' (t = 1..36) or a comma-separated list, e.g. '6,12,18,24,30,36'
#   output_dir:   directory for per-replicate results (default ./outputs/YYYYMMDD)
#   replicates:   optional subset of replicates, e.g. '1:25', to run disjoint ranges in parallel processes

ESTIMATOR=${1:-"tmle"}
CORES=${2:-2}
USE_SL=${3:-"TRUE"}
DO_MPI=${4:-"FALSE"}
R=${5:-100}
TARGET_TIMES=${6:-"all"}
OUTPUT_DIR=${7:-"./outputs/$(date +%Y%m%d)"}
REPLICATES=${8:-""}

# Increase stack size to prevent C stack overflow (ignored if not permitted)
ulimit -s unlimited 2>/dev/null || true

# The Python environment is only needed for the LSTM estimator
if [ "$ESTIMATOR" = "tmle-lstm" ]; then
  source ./myenv/bin/activate
fi

echo "Running simulation with: estimator=$ESTIMATOR, cores=$CORES, use_SL=$USE_SL, do_MPI=$DO_MPI, R=$R, target_times=$TARGET_TIMES, output_dir=$OUTPUT_DIR, replicates=${REPLICATES:-1:$R}"
Rscript simulation.R "$ESTIMATOR" "$CORES" "$USE_SL" "$DO_MPI" "$R" "$TARGET_TIMES" "$OUTPUT_DIR" "$REPLICATES"
