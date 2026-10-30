#!/bin/bash
# Runs the simulation study (simulation.R); see README.md.
# Usage: ./run_simulation.sh estimator cores use_sl [R] [target_times] [output_dir] [replicates]
#   estimator:    'tmle' (Super Learner / GLM learners) or 'tmle-lstm' (LSTM learners);
#                 both use the same LTMLE core
#   cores:        cores used within each replicate (forked workers; for 'tmle-lstm' each runs
#                 single-threaded TensorFlow)
#   use_sl:       'tmle' only: TRUE = Super Learner, FALSE = GLM / multinomial regression
#   R:            number of replicates (default 100); replicates already saved in output_dir
#                 are skipped, so an interrupted run resumes where it stopped
#   target_times: 'all' (t = 1..36) or a comma-separated list, e.g. '12,24,36'
#   output_dir:   directory for the per-replicate results (default ./outputs/YYYYMMDD)
#   replicates:   optional subset of replicates, e.g. '1:25' or '7', to run disjoint ranges in
#                 parallel processes or to re-run a single replicate (spot check)

ESTIMATOR=${1:-"tmle"}
CORES=${2:-2}
USE_SL=${3:-"TRUE"}
R=${4:-100}
TARGET_TIMES=${5:-"all"}
OUTPUT_DIR=${6:-"./outputs/$(date +%Y%m%d)"}
REPLICATES=${7:-""}

# Increase stack size to prevent C stack overflow (ignored if not permitted)
ulimit -s unlimited 2>/dev/null || true

# The Python environment is only needed for the LSTM estimator
if [ "$ESTIMATOR" = "tmle-lstm" ]; then
  source ./myenv/bin/activate
fi

echo "Running simulation with: estimator=$ESTIMATOR, cores=$CORES, use_SL=$USE_SL, R=$R," \
  "target_times=$TARGET_TIMES, output_dir=$OUTPUT_DIR, replicates=${REPLICATES:-1:$R}"
Rscript simulation.R "$ESTIMATOR" "$CORES" "$USE_SL" "$R" "$TARGET_TIMES" "$OUTPUT_DIR" \
  "$REPLICATES"
