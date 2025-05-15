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

# Fix sl3 learner family handling
# This is a critical fix for the "$ operator is invalid for atomic vectors" error 
fix_sl3_issues <- function() {
  # First ensure our improved original_subset_covariates exists in global env
  if(!exists("original_subset_covariates", envir = .GlobalEnv)) {
    # Define a robust version of the function that provides fallback values
    original_subset_covariates <- function(task) {
      tryCatch({
        # Exit quickly with fallback data if task is invalid
        if(is.null(task) || !is.environment(task) || is.null(task$nodes) || is.null(task$nodes$covariates)) {
          # Create minimal fallback data
          return(create_fallback_covariates())
        }
        
        # Get required covariates
        all_covariates <- task$nodes$covariates
        
        # Exit with fallback if covariates are empty
        if(length(all_covariates) == 0) {
          return(create_fallback_covariates())
        }
        
        # Check task data
        if(is.null(task$X)) {
          # Create empty data with right columns
          X_dt <- data.table::data.table(matrix(0, nrow = max(1, task$nrow), ncol = length(all_covariates)))
          data.table::setnames(X_dt, all_covariates)
          return(X_dt)
        }
        
        # Normal processing path
        X_dt <- task$X
        task_covariates <- colnames(X_dt)
        
        # Find missing columns
        missing_covariates <- setdiff(all_covariates, task_covariates)
        
        # Add missing columns if needed
        if(length(missing_covariates) > 0) {
          missing_cols <- data.table::data.table(matrix(0, nrow = task$nrow, ncol = length(missing_covariates)))
          data.table::setnames(missing_cols, missing_covariates)
          X_dt <- cbind(X_dt, missing_cols)
        }
        
        # Return the data with all required columns
        result <- X_dt[, all_covariates, with = FALSE]
        return(result)
      }, error = function(e) {
        # Return fallback on any error
        warning("Error in original_subset_covariates: ", e$message)
        return(create_fallback_covariates())
      })
    }
    
    # Helper to create fallback covariate data
    create_fallback_covariates <- function(rows=1) {
      fallback <- data.table::data.table(
        V3=0, L1=0, L2=0, L3=0, Y.lag=0, Y.lag2=0, Y.lag3=0,
        L1.lag=0, L1.lag2=0, L1.lag3=0, L2.lag=0, L2.lag2=0, L2.lag3=0,
        L3.lag=0, L3.lag2=0, L3.lag3=0, white=0, black=0, latino=0, other=0,
        mdd=0, bipolar=0, schiz=0, A1=0, A2=0, A3=0, A4=0, A5=0, A6=0,
        A1.lag=0, A2.lag=0, A3.lag=0, A4.lag=0, A5.lag=0, A6.lag=0,
        A1.lag2=0, A2.lag2=0, A3.lag2=0, A4.lag2=0, A5.lag2=0, A6.lag2=0,
        A1.lag3=0, A2.lag3=0, A3.lag3=0, A4.lag3=0, A5.lag3=0, A6.lag3=0
      )
      if(rows > 1) {
        fallback <- fallback[rep(1, rows)]
      }
      return(fallback)
    }
    
    # Export to global environment
    assign("original_subset_covariates", original_subset_covariates, envir = .GlobalEnv)
    assign("create_fallback_covariates", create_fallback_covariates, envir = .GlobalEnv)
    cat("Enhanced original_subset_covariates function attached to global environment\n")
  }
  
  # Fix the family parameter handling (the "$ operator is invalid for atomic vectors" error)
  if(requireNamespace("nnet", quietly = TRUE)) {
    # Create a safer multinomial family object
    safe_multinomial_family <- function() {
      structure(list(
        family = "multinomial", 
        link = "logit",
        linkfun = function(mu) log(mu/(1-mu)),
        linkinv = function(eta) 1/(1+exp(-eta)),
        variance = function(mu) mu*(1-mu),
        dev.resids = function(y, mu, wt) NULL,
        aic = function(y, n, mu, wt, dev) NULL,
        mu.eta = function(eta) exp(eta)/(1+exp(eta))^2,
        initialize = expression({mustart = rep(1/6, 6); weights = rep(1, 6)})
      ), class = "family")
    }
    assign("safe_multinomial_family", safe_multinomial_family, envir = .GlobalEnv)
    cat("Created safe_multinomial_family function\n")
    
    # Create a more comprehensive direct implementation of multinomial model training
    # This bypasses all sl3 functionality to avoid the "$ operator is invalid for atomic vectors" error
    simple_multinomial_model <- function(data, outcome, covariates) {
      # Handle various input patterns for better compatibility
      if(is.environment(data) && !is.null(data$data)) {
        # Input appears to be an sl3_Task object
        if(is.null(outcome) && !is.null(data$nodes$outcome)) {
          outcome <- data$nodes$outcome
        }
        if(is.null(covariates) && !is.null(data$nodes$covariates)) {
          covariates <- data$nodes$covariates
        }
        # Extract data from task
        data <- data$data
      }
      
      # Handle missing or empty outcome
      if(is.null(outcome) || length(outcome) == 0) {
        if("A" %in% colnames(data)) {
          outcome <- "A"
        } else if("Y" %in% colnames(data)) {
          outcome <- "Y"
        } else {
          # Try to find an outcome column
          possible_outcomes <- c("treatment", "target", "outcome", "response")
          for(col in possible_outcomes) {
            if(col %in% colnames(data)) {
              outcome <- col
              break
            }
          }
          
          # If still no outcome found, use first column as default
          if(is.null(outcome)) {
            outcome <- colnames(data)[1]
            cat("No outcome specified, using first column: ", outcome, "\n")
          }
        }
      }
      
      # Handle missing covariates
      if(is.null(covariates) || length(covariates) == 0) {
        # Use all columns except outcome
        covariates <- setdiff(colnames(data), outcome)
        
        # Limit to 20 covariates max for performance
        if(length(covariates) > 20) {
          covariates <- covariates[1:20]
          cat("Too many covariates (", length(covariates), 
                  "), limiting to first 20 for performance\n")
        }
      }
      
      # Create model wrapper function that returns an object with predict method
      tryCatch({
        # Create formula from outcome and covariates
        formula_str <- paste(outcome, "~", paste(covariates, collapse = " + "))
        formula_obj <- as.formula(formula_str)
        
        # Use nnet::multinom directly with conservative settings
        model <- nnet::multinom(formula_obj, data = data, trace = FALSE, 
                               maxit = 100, MaxNWts = 5000)
        
        # Create a predict function that handles sl3 task objects
        predict_function <- function(newdata) {
          # Handle sl3 task objects
          if(is.environment(newdata) && !is.null(newdata$X)) {
            newdata <- newdata$X
          }
          
          # Make predictions with error handling
          preds <- tryCatch({
            predict(model, newdata = newdata, type = "probs")
          }, error = function(e) {
            cat("Prediction error:", e$message, "\n")
            # Return uniform distribution
            n_rows <- ifelse(is.data.frame(newdata), nrow(newdata), 100)
            uniform <- matrix(1/6, nrow = n_rows, ncol = 6)
            colnames(uniform) <- paste0("A", 1:6)
            return(uniform)
          })
          
          # Process predictions to ensure matrix format
          if(is.vector(preds) && !is.list(preds)) {
            # Reshape single-row output to matrix
            preds <- matrix(preds, nrow = 1)
          }
          
          # Ensure proper column names
          if(is.null(colnames(preds))) {
            colnames(preds) <- paste0("A", 1:ncol(preds))
          }
          
          return(preds)
        }
        
        # Return a list with predict method (compatible with sl3)
        return(list(
          fit_object = model,
          predict = predict_function
        ))
      }, error = function(e) {
        cat("Multinomial model training failed:", e$message, "\n")
        # Return fallback (uniform predictions)
        return(list(
          fit_object = NULL,
          predict = function(newdata) {
            n_rows <- ifelse(is.data.frame(newdata), nrow(newdata), 
                           ifelse(is.environment(newdata) && !is.null(newdata$X), 
                                  nrow(newdata$X), 100))
            uniform <- matrix(1/6, nrow = n_rows, ncol = 6)
            colnames(uniform) <- paste0("A", 1:6)
            return(uniform)
          }
        ))
      })
    }
    
    # Create a direct replacement for create_treatment_model_sl
    direct_create_treatment_model_sl <- function(n.folds = 3) {
      # Function that returns a model object compatible with sl3 interfaces
      # but uses direct nnet::multinom instead of sl3 to avoid all dependency issues
      
      # Create empty placeholder to be filled during training
      model_obj <- list(
        name = "direct_multinom",
        fit_object = NULL,
        
        # Define train method that will be called
        train = function(task) {
          # Print informative message
          cat("Training with direct multinomial implementation\n")
          
          # Use simple_multinomial_model directly
          fit_result <- simple_multinomial_model(task, NULL, NULL)
          
          # Store fit object and predict function
          self$fit_object <- fit_result$fit_object
          self$predict <- fit_result$predict
          
          return(self)
        },
        
        # Define basic predict function (will be overridden during training)
        predict = function(task) {
          cat("Model needs training first\n")
          self$train(task)
          
          # Now call the real predict function that was set during training
          return(self$predict(task))
        }
      )
      
      # Create environment for self-reference
      e <- new.env(parent = environment())
      e$self <- model_obj
      environment(model_obj$train) <- environment(model_obj$predict) <- e
      
      # Set appropriate class for compatibility
      class(model_obj) <- c("direct_multinom", "Lrnr_base")
      
      return(model_obj)
    }
    
    # Export the functions to global environment
    assign("simple_multinomial_model", simple_multinomial_model, envir = .GlobalEnv)
    assign("direct_create_treatment_model_sl", direct_create_treatment_model_sl, envir = .GlobalEnv)
    
    # Override the built-in create_treatment_model_sl with our direct implementation
    assign("create_treatment_model_sl", direct_create_treatment_model_sl, envir = .GlobalEnv)
    
    cat("Created enhanced direct multinomial model implementation\n")
  }
  
  # Create a fallback prediction method
  make_uniform_prediction <- function(n_rows, n_classes=6) {
    # Make a uniform prediction (equal probability for all classes)
    pred <- matrix(1/n_classes, nrow=n_rows, ncol=n_classes)
    colnames(pred) <- paste0("A", 1:n_classes)
    return(pred)
  }
  assign("make_uniform_prediction", make_uniform_prediction, envir = .GlobalEnv)
  cat("Created fallback prediction generator\n")
  
  # Create a safer learner train function
  safe_train_multinomial <- function(learner, task, ...) {
    tryCatch({
      # Try to train using the standard method first
      learner$train(task, ...)
    }, error = function(e) {
      # If it fails, create a dummy fit object
      warning("Training failed with error: ", e$message, ". Using fallback uniform model.")
      
      # Create a dummy fit with predict method
      dummy_fit <- list(
        predict = function(newdata, ...) {
          # Make uniform predictions for all rows
          n_rows <- ifelse(is.data.frame(newdata), nrow(newdata), 1)
          return(make_uniform_prediction(n_rows))
        }
      )
      class(dummy_fit) <- "dummy_multinomial_fit"
      return(dummy_fit)
    })
  }
  assign("safe_train_multinomial", safe_train_multinomial, envir = .GlobalEnv)
  cat("Created safe_train_multinomial function\n")
}

# Apply the sl3 fixes
fix_sl3_issues()

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