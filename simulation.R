############################################################################################
# Longitudinal simulation study: LTMLE, IPTW and g-computation with multinomial or        #
# binomial treatment models, using Super Learner / GLM ("tmle") or LSTM ("tmle-lstm")      #
# learners. Driver script: run through run_simulation.sh (see README.md).                 #
#                                                                                          #
# Usage: Rscript simulation.R estimator cores use_SL [R] [target_times] [output_dir]       #
#                             [replicates]                                                 #
#   estimator     "tmle" (Super Learner or GLM learners) or "tmle-lstm" (LSTM learners)    #
#   cores         cores used within each replicate (forked workers)                       #
#   use_SL        "tmle" only: TRUE = Super Learner, FALSE = GLM                           #
#   R             number of replicates (default 100)                                       #
#   target_times  "all" (t = 1..36) or a comma-separated list, e.g. "12,24,36"             #
#   output_dir    directory for the per-replicate results (default ./outputs/YYYYMMDD)     #
#   replicates    optional subset of replicates, e.g. "1:25" or "7"                        #
# Each replicate r is saved as its own .rds file in output_dir by simLong(); replicates   #
# already saved there are skipped, so an interrupted run resumes where it stopped.        #
############################################################################################

args <- commandArgs(trailingOnly = TRUE)
estimator <- as.character(args[1])
if (!estimator %in% c("tmle", "tmle-lstm")) stop("estimator must be 'tmle' or 'tmle-lstm'")
cores <- as.numeric(args[2])
use.SL <- as.logical(args[3])
arg_or <- function(i, default) if (length(args) >= i && nzchar(args[i])) args[i] else default

# simulation settings
n <- 10000 # sample size
treatment.rule <- "all" # psi_t is estimated under the static, dynamic and stochastic rules
J <- 6 # number of treatments
t.end <- 36 # number of months after baseline (t = 0)
R <- as.numeric(arg_or(4, 100)) # number of replicates
target.times <- arg_or(5, "all") # months t at which psi_t is estimated
target.times <- if (target.times == "all") 1:t.end else as.numeric(strsplit(target.times, ",")[[1]])
replicates <- arg_or(7, paste0("1:", R)) # replicates to run, e.g. "1:25" or "26,27,28"
replicates <- if (grepl(":", replicates)) {
  do.call(seq, as.list(as.integer(strsplit(replicates, ":")[[1]])))
} else {
  as.integer(strsplit(replicates, ",")[[1]])
}
scale.continuous <- FALSE # continuous covariates are not standardized
gbound <- c(0.05, 1) # bound for the treatment density ratio (as 1/20) and censoring probabilities
ybound <- c(0.0001, 0.9999) # bounds for the outcome predictions
n.folds <- 3 # cross-validation folds of the Super Learner

output_dir <- sub("/$", "", arg_or(6, file.path("outputs", format(Sys.time(), "%Y%m%d"))))
output_dir <- paste0(output_dir, "/")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# packages (attached in this order)
for (pkg in c("simcausal", "purrr", "origami", "sl3", "nnet", "ranger", "glmnet", "MASS",
              "progressr", "data.table", "gtools", "dplyr", "readr", "tidyr")) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}
options(simcausal.verbose = FALSE, sl3.verbose = FALSE)

# function definitions
source(file.path("src", "simcausal_fns.R"))
source(file.path("src", "tmle_fns.R"))
source(file.path("src", "SL3_fns.R"))
source(file.path("src", "tmle_IC.R"))
source(file.path("src", "simulation_fns.R"))
if (estimator == "tmle-lstm") {
  # LSTM learners (src/lstm.R; Python side in src/*.py). Python is selected here but initialised
  # only inside the (forked) workers that fit the LSTMs.
  suppressPackageStartupMessages(library(reticulate))
  source(file.path("src", "lstm.R"))
  lstm_python(file.path(".", "myenv", "bin", "python"))
}
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))

# data-generating process and the cached large-n truth (computed once, stored in ./data)
source(file.path("src", "simcausal_dgp.R"))
Dset <- set.DAG(D, vecfun = c("StochasticFun"))
invisible(compute_truth(Dset, t.end))

# skip replicates already saved in output_dir (resume a long run)
done_files <- list.files(output_dir, pattern = paste0("longitudinal_simulation_results_estimator_",
                                                      estimator, "_treatment_rule_all_r_\\d+_"))
done_r <- as.numeric(sub(".*_r_(\\d+)_.*", "\\1", done_files))
to_run <- replicates[!replicates %in% done_r]

print(paste0("simulation setting: estimator = ", estimator, ", R = ", R, ", n = ", n,
             ", J = ", J, ", t.end = ", t.end, ", use.SL = ", use.SL, ", target times = ",
             paste(target.times, collapse = ","), ", output_dir = ", output_dir))

# run the replicates sequentially (each replicate parallelises internally over `cores`);
# a failed replicate is reported and the remaining replicates still run
for (r in to_run) {
  tryCatch(simLong(r = r, Dset = Dset, output_dir = output_dir, J = J, n = n, t.end = t.end,
                   gbound = gbound, ybound = ybound, n.folds = n.folds, cores = cores,
                   estimator = estimator, treatment.rule = treatment.rule, use.SL = use.SL,
                   scale.continuous = scale.continuous, target.times = target.times),
           error = function(e) message("Replicate ", r, " failed: ", conditionMessage(e)))
}
