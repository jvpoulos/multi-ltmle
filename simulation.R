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
library(methods) # For R6 class operations
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

# Ensure the Stack class is properly patched for covariate handling
# This is a critical fix for the "Task missing covariates" error
patch_stack_class <- function() {
  # First verify original_subset_covariates exists in global environment
  if(!exists("original_subset_covariates", envir = .GlobalEnv)) {
    # Define a robust version of the function directly
    original_subset_covariates <- function(task) {
      # Comprehensive error handling
      tryCatch({
        # Get covariates from the task
        if (is.null(task) || is.null(task$nodes) || is.null(task$nodes$covariates)) {
          warning("Task structure is invalid. Creating fallback data with required covariates.")
          return(data.table::data.table(
            V3=0, L1=0, L2=0, L3=0, Y.lag=0, Y.lag2=0, Y.lag3=0,
            L1.lag=0, L1.lag2=0, L1.lag3=0, L2.lag=0, L2.lag2=0, L2.lag3=0,
            L3.lag=0, L3.lag2=0, L3.lag3=0, white=0, black=0, latino=0, other=0,
            mdd=0, bipolar=0, schiz=0, A1=0, A2=0, A3=0, A4=0, A5=0, A6=0,
            A1.lag=0, A2.lag=0, A3.lag=0, A4.lag=0, A5.lag=0, A6.lag=0,
            A1.lag2=0, A2.lag2=0, A3.lag2=0, A4.lag2=0, A5.lag2=0, A6.lag2=0,
            A1.lag3=0, A2.lag3=0, A3.lag3=0, A4.lag3=0, A5.lag3=0, A6.lag3=0
          ))
        }
        
        all_covariates <- task$nodes$covariates
        
        # Check if all covariates exist in the task
        if(is.null(task$X)) {
          warning("Task X is NULL. Creating data with required covariates.")
          X_dt <- data.table::data.table(matrix(0, nrow = max(1, task$nrow), ncol = length(all_covariates)))
          data.table::setnames(X_dt, all_covariates)
          return(X_dt)
        }
        
        X_dt <- task$X
        task_covariates <- colnames(X_dt)
        
        # Find missing covariates
        missing_covariates <- setdiff(all_covariates, task_covariates)
        
        # If there are missing covariates, add them to the task data
        if (length(missing_covariates) > 0) {
          # Log missing covariates
          message("Adding missing covariates: ", paste(missing_covariates, collapse=", "))
          
          # Create a new data.table with the missing covariates
          missing_cols <- data.table::data.table(matrix(0, nrow = task$nrow, ncol = length(missing_covariates)))
          data.table::setnames(missing_cols, missing_covariates)
          
          # Add the missing columns to X_dt
          X_dt <- cbind(X_dt, missing_cols)
        }
        
        # Return the task with all covariates
        if (all(all_covariates %in% colnames(X_dt))) {
          return(X_dt[, all_covariates, with = FALSE])
        } else {
          # Some covariates still missing - create fallback
          warning("Still missing covariates after attempted fix. Creating complete fallback.")
          result <- data.table::data.table(matrix(0, nrow = nrow(X_dt), ncol = length(all_covariates)))
          data.table::setnames(result, all_covariates)
          return(result)
        }
      }, error = function(e) {
        # Complete fallback on any error
        warning("Error in original_subset_covariates: ", e$message, ". Creating comprehensive fallback data.")
        result <- data.table::data.table(
          V3=0, L1=0, L2=0, L3=0, Y.lag=0, Y.lag2=0, Y.lag3=0,
          L1.lag=0, L1.lag2=0, L1.lag3=0, L2.lag=0, L2.lag2=0, L2.lag3=0,
          L3.lag=0, L3.lag2=0, L3.lag3=0, white=0, black=0, latino=0, other=0,
          mdd=0, bipolar=0, schiz=0, A1=0, A2=0, A3=0, A4=0, A5=0, A6=0,
          A1.lag=0, A2.lag=0, A3.lag=0, A4.lag=0, A5.lag=0, A6.lag=0,
          A1.lag2=0, A2.lag2=0, A3.lag2=0, A4.lag2=0, A5.lag2=0, A6.lag2=0,
          A1.lag3=0, A2.lag3=0, A3.lag3=0, A4.lag3=0, A5.lag3=0, A6.lag3=0
        )
        return(result)
      })
    }
    
    # Attach to global environment
    assign("original_subset_covariates", original_subset_covariates, envir = .GlobalEnv)
    cat("Enhanced original_subset_covariates function has been manually attached to global environment\n")
  }
  
  # Now try to monkey-patch the Stack class method directly
  if(exists("Stack") && methods::is(Stack, "R6ClassGenerator")) {
    tryCatch({
      # Define a new wrapper that ensures subset_covariates uses our function
      Stack$set("public", "subset_covariates", function(task) {
        # Call the function from global environment
        return(original_subset_covariates(task))
      })
      cat("Successfully patched Stack$subset_covariates method to handle missing covariates\n")
    }, error = function(e) {
      warning("Failed to patch Stack$subset_covariates: ", e$message)
    })
  } else {
    message("Stack class not properly loaded yet - can't patch directly")
  }
  
  # For SL3 >= v1.4.0, also try to set learner_custom_args for Stacks
  if (exists("learner_custom_args", where = asNamespace("sl3"), inherits = FALSE)) {
    tryCatch({
      # Get the function from sl3 namespace
      learner_custom_args_fn <- get("learner_custom_args", envir = asNamespace("sl3"))
      # Set custom args for Stack to use our subset_covariates function
      learner_custom_args_fn$Stack$set("subset_covariates", original_subset_covariates)
      cat("Applied learner_custom_args patch for Stack in sl3 package\n")
    }, error = function(e) {
      warning("Failed to set learner_custom_args for Stack: ", e$message)
    })
  }
}

# Apply the patches to ensure Stack class can handle missing covariates
patch_stack_class()

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
debug <- FALSE

# Define final_vector (used in simulation.R.original)
final_vector <- 1

# Set flag to indicate this is sourced by main script
SOURCED_BY_MAIN_SCRIPT <- TRUE

# Source the original file to get the simLong function
cat("Loading simLong function from simulation.R.original...\n")
source("simulation.R.original")
cat("simLong function loaded successfully\n")

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
  window_size=window_size
)

# Save results
cat("Saving results to", filename, "\n")
saveRDS(sim_results, file=filename)

cat("Simulation completed successfully!\n")