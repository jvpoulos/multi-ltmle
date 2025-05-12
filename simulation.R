#\!/usr/bin/env Rscript

################################################################################
# Fix for the simulation.R script with proper error handling and variable scope
################################################################################

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  cat("Not enough arguments provided. Using defaults.\n")
  args <- c("tmle", "1", "TRUE", "FALSE")
}

# Extract arguments
estimator <- as.character(args[1])
cores <- as.numeric(args[2])
use.SL <- as.logical(args[3])
doMPI <- as.logical(args[4])

# Create output directory
simulation_version <- format(Sys.time(), "%Y%m%d")
output_dir <- paste0("./outputs/", simulation_version, "/")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

# Define filename for saving results
filename <- paste0(output_dir,
                   "longitudinal_simulation_results_",
                   "estimator_", estimator,
                   "_treatment.rule_all",
                   "_R_11",
                   "_n_10000",
                   "_J_6",
                   "_n.folds_3",
                   "_scale.continuous_FALSE",
                   "_use.SL_", use.SL, ".rds")

# Print simulation settings
cat("Simulation settings:\n")
cat("  estimator =", estimator, "\n")
cat("  cores =", cores, "\n")
cat("  use.SL =", use.SL, "\n")
cat("  doMPI =", doMPI, "\n")
cat("Results will be saved to:", filename, "\n")

# Now load the original simulation.R code and run the simLong function directly
tryCatch({
  # Source the simulation functions from the original file
  original_code_lines <- readLines("simulation.R.original")

  # Find the simLong function definition
  sim_function_start <- grep("^simLong <- function", original_code_lines)
  sim_function_end <- NULL

  # Find where the simLong function ends
  brace_count <- 0
  for(i in sim_function_start:length(original_code_lines)) {
    line <- original_code_lines[i]
    brace_count <- brace_count + sum(gregexpr("\\{", line)[[1]] > 0)
    brace_count <- brace_count - sum(gregexpr("\\}", line)[[1]] > 0)

    if(brace_count == 0 && i > sim_function_start) {
      sim_function_end <- i
      break
    }
  }

  if(!is.null(sim_function_end)) {
    # Extract the simLong function
    sim_function_code <- original_code_lines[sim_function_start:sim_function_end]
    sim_function_text <- paste(sim_function_code, collapse="\n")

    # Load required libraries
    library(simcausal)
    options(simcausal.verbose=FALSE)
    library(purrr)
    library(origami)
    library(sl3)
    options(sl3.verbose = FALSE)
    library(nnet)
    library(ranger)
    library(glmnet)
    library(MASS)
    library(progressr)
    library(data.table)
    library(gtools)
    library(dplyr)
    library(readr)
    library(tidyr)
    library(latex2exp)

    # Add needed source files
    source('./src/tmle_IC.R')
    source('./src/misc_fns.R')
    source('./src/simcausal_fns.R')
    source('./src/SL3_fns.R')

    if(estimator == "tmle") {
      source('./src/tmle_fns.R')
    } else if(estimator == "tmle-lstm") {
      source('./src/tmle_fns_lstm.R')
      source('./src/lstm.R')
      verify_reticulate()
      library(reticulate)
      use_python("/media/jason/Dropbox/github/multi-ltmle/myenv/bin/python", required = TRUE)
      library(tensorflow)
      library(keras)
    }

    # Define settings
    treatment.rule <- "all"
    n <- 10000
    J <- 6
    t.end <- 36
    R <- 11
    scale.continuous <- FALSE
    gbound <- c(0.05,1)
    ybound <- c(0.0001,0.9999)
    n.folds <- 3
    window_size <- 12
    debug <- FALSE

    # Evaluate the simLong function
    eval(parse(text=sim_function_text))

    # Run the simulation for a single iteration
    cat("Running simulation for iteration 1...\n")
    sim_results <- simLong(
      r=1,
      J=J,
      n=n,
      t.end=t.end,
      gbound=gbound,
      ybound=ybound,
      n.folds=n.folds,
      cores=cores,
      estimator=estimator,
      treatment.rule=treatment.rule,
      use.SL=use.SL,
      scale.continuous=scale.continuous,
      debug=debug,
      window_size=window_size
    )

    # Save the results
    saveRDS(sim_results, filename)
    cat("Results saved to:", filename, "\n")
  } else {
    stop("Could not find end of simLong function in original script")
  }
}, error = function(e) {
  cat("Error running simulation:", e$message, "\n")
  # Save error information
  error_info <- list(
    timestamp = Sys.time(),
    error = e$message,
    settings = list(
      estimator = estimator,
      cores = cores,
      use.SL = use.SL,
      doMPI = doMPI,
      filename = filename
    )
  )
  error_filename <- paste0(output_dir, "error_", estimator, "_", format(Sys.time(), "%H%M%S"), ".rds")
  saveRDS(error_info, error_filename)
  cat("Error information saved to:", error_filename, "\n")
})

cat("Script execution completed.\n")
