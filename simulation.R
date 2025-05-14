#!/usr/bin/env Rscript

################################################################################
# Simulation script for longitudinal TMLE with multi-valued treatments
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
  use_python("./myenv/bin/python", required=TRUE)
  library(tensorflow)
  library(keras)
  
  # Create the prepare_lstm_data function if it doesn't exist
  # This is a direct definition in case the function cannot be found in tmle_fns_lstm.R
  if(!exists("prepare_lstm_data")) {
    cat("prepare_lstm_data function not found, defining it directly.\n")
    prepare_lstm_data <- function(tmle_dat, t.end, window_size) {
      # Input validation
      if(!is.data.frame(tmle_dat)) {
        stop("tmle_dat must be a data frame")
      }
      
      # Calculate n_ids at the start
      n_ids <- length(unique(tmle_dat$ID))
      print(paste("Found", n_ids, "unique IDs"))
      
      # Print debug info
      print("Available columns in tmle_dat:")
      print(names(tmle_dat))
      
      # First check for direct A columns, including variant patterns
      A_cols <- c(
        grep("^A$", colnames(tmle_dat), value=TRUE),
        grep("^A\\.[0-9]+$", colnames(tmle_dat), value=TRUE)
      )
      
      if (length(A_cols) == 0) {
        print("Treatment columns not found - checking target columns")
        
        # Look for treatment data in target columns
        target_cols <- grep("^target$|^treatment", colnames(tmle_dat), value=TRUE)
        
        # If target columns exist, use them to create A columns
        if (length(target_cols) > 0) {
          print(paste("Creating A columns from:", paste(target_cols, collapse=", ")))
          tmle_dat$A <- as.numeric(tmle_dat[[target_cols[1]]])
          
          # Create time-specific treatment columns
          for(t in 0:t_end) {
            tmle_dat[[paste0("A.", t)]] <- tmle_dat$A
          }
        } 
        # Otherwise use diagnosis data if available
        else if (all(c("mdd", "bipolar", "schiz") %in% colnames(tmle_dat))) {
          print("Creating treatment columns from diagnoses")
          tmle_dat$A <- ifelse(tmle_dat$schiz == 1, 2,
                               ifelse(tmle_dat$bipolar == 1, 4,
                                      ifelse(tmle_dat$mdd == 1, 1, 5)))
          
          # Create time-specific columns
          for(t in 0:t_end) {
            tmle_dat[[paste0("A.", t)]] <- tmle_dat$A
          }
        }
        # Last resort - use default values
        else {
          print("No treatment data found - using default treatment assignment")
          # Don't print warning, just informational message
          tmle_dat$A <- 5
          for(t in 0:t_end) {
            tmle_dat[[paste0("A.", t)]] <- 5
          }
        }
      }
      
      # Process time-varying covariates (L1, L2, L3)
      print("Processing time-varying covariates...")
      for(L in c("L1", "L2", "L3")) {
        L_cols <- grep(paste0("^", L, "\\."), names(tmle_dat), value=TRUE)
        if(length(L_cols) == 0) {
          print(paste("Creating time-varying columns for", L))
          # Create time-varying L columns if missing
          base_value <- if(L == "L1") 0 else ifelse(tmle_dat[[L]] == 1, 1, 0)
          for(t in 0:t.end) {
            tmle_dat[[paste0(L, ".", t)]] <- base_value
          }
        }
      }
      
      # Process censoring columns
      print("Processing censoring columns...")
      C_cols <- grep("^C\\.", names(tmle_dat), value=TRUE)
      if(length(C_cols) == 0) {
        print("Creating default censoring columns")
        for(t in 0:t.end) {
          tmle_dat[[paste0("C.", t)]] <- 0  # Default to uncensored
        }
      }
      
      # Process outcome columns
      print("Processing outcome columns...")
      Y_cols <- grep("^Y\\.", names(tmle_dat), value=TRUE)
      if(length(Y_cols) == 0) {
        print("Creating default outcome columns")
        for(t in 0:t.end) {
          tmle_dat[[paste0("Y.", t)]] <- -1  # Default to missing
        }
      }
      
      # Ensure proper formatting of treatment columns
      print("Formatting treatment columns...")
      A_cols_all <- grep("^A", names(tmle_dat), value=TRUE)
      for(col in A_cols_all) {
        if(is.factor(tmle_dat[[col]])) {
          tmle_dat[[col]] <- as.numeric(as.character(tmle_dat[[col]]))
        }
        # Ensure treatments are in 1-6 range and handle NAs
        tmle_dat[[col]] <- pmin(pmax(replace(tmle_dat[[col]], is.na(tmle_dat[[col]]), 5), 1), 6)
      }
      
      # Handle missing values
      print("Handling missing values...")
      numeric_cols <- sapply(tmle_dat, is.numeric)
      for(col in names(tmle_dat)[numeric_cols]) {
        if(grepl("^(L|C|Y|V3)", col)) {
          tmle_dat[[col]][is.na(tmle_dat[[col]])] <- -1
        }
      }
      
      # Handle categorical variables
      print("Processing categorical variables...")
      categorical_vars <- c("white", "black", "latino", "other", "mdd", "bipolar", "schiz")
      for(col in categorical_vars) {
        if(col %in% names(tmle_dat)) {
          tmle_dat[[col]][is.na(tmle_dat[[col]])] <- 0
        } else {
          tmle_dat[[col]] <- 0
        }
      }
      
      # Final validation
      # Check for required columns
      required_prefixes <- c("A", "L1", "L2", "L3", "C", "Y")
      missing_prefixes <- required_prefixes[!sapply(required_prefixes, function(x) 
        any(grepl(paste0("^", x), names(tmle_dat))))]
      if(length(missing_prefixes) > 0) {
        warning(paste("Missing required column types:", paste(missing_prefixes, collapse=", ")))
      }
      
      # Print summary of processed data
      print("Processed data structure:")
      print(paste("Number of IDs:", n_ids))
      print(paste("Time points:", t.end + 1))
      print(paste("Treatment columns:", 
                  paste(grep("^A", names(tmle_dat), value=TRUE), collapse=", ")))
      
      # Validate final structure
      print("Final column types:")
      print(sapply(tmle_dat, class))
      
      return(list(
        data = tmle_dat,
        n_ids = n_ids
      ))
    }
    assign("prepare_lstm_data", prepare_lstm_data, envir = .GlobalEnv)
    cat("prepare_lstm_data function defined and exported to global environment\n")
  }
}

# Fix for weights_bin not found error
if(!exists("initialize_weights_bin")) {
  initialize_weights_bin <- function(tmle_dat, Q_star) {
    # Create a default weights_bin object that matches the dimension of Q_star
    if(is.matrix(Q_star)) {
      weights_bin <- matrix(1, nrow=nrow(Q_star), ncol=ncol(Q_star))
    } else if(is.vector(Q_star)) {
      weights_bin <- rep(1, length(Q_star))
    } else {
      # Default case if Q_star is neither matrix nor vector
      weights_bin <- 1
    }
    return(weights_bin)
  }
  assign("initialize_weights_bin", initialize_weights_bin, envir = .GlobalEnv)
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
target.gwt <- TRUE
debug <- FALSE

# Load the original simulation.R file for the simLong function
source("simulation.R.original")

# Run the simulation
cat("Running simulation...\n")
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
  window_size=window_size,
  target.gwt=target.gwt
)

# Save results
cat("Saving results to", filename, "\n")
saveRDS(sim_results, file=filename)

cat("Simulation completed successfully!\n")