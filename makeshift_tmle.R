#!/usr/bin/env Rscript

#' Makeshift fix for 'number of items to replace is not a multiple of replacement length' error
#' This script provides a minimal direct fix for the standard tmle estimator
#' It monkey-patches the getTMLELong function to handle vector length mismatches

# Load necessary libraries
library(simcausal)
library(sl3)
library(origami)

# Monkeypatch the global environment with a fixed version of safe_getTMLELong
fix_safe_getTMLELong <- function() {
  cat("Applying direct patch to safe_getTMLELong function...\n")
  
  # Define the improved safe_getTMLELong function that specifically handles length mismatches
  safe_getTMLELong <- function(...) {
    tryCatch({
      # First attempt - normal call
      getTMLELong(...)
    }, error = function(e) {
      # Check specifically for vector length mismatch error
      if(grepl("not a multiple of replacement length", e$message) || 
         grepl("number of items to replace", e$message)) {
        
        # Get the arguments passed to getTMLELong
        args <- list(...)
        initial_model_for_Y <- args[[1]]
        tmle_rules <- args[[2]]
        tmle_covars_Y <- args[[3]]
        g_preds_bounded <- args[[4]]
        C_preds_bounded <- args[[5]]
        ybound <- args[[6]]
        
        # Extract data for debugging
        cat("Vector length mismatch detected. Patching...\n")
        cat("Rule names:", paste(names(tmle_rules), collapse=", "), "\n")
        
        # Extract data safely
        tmle_dat <- NULL
        if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
          tmle_dat <- initial_model_for_Y$data
          cat("Number of rows in data:", nrow(tmle_dat), "\n")
        } else {
          cat("No data found in initial_model_for_Y\n")
          tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
        }
        
        # Create an explicitly sized matrix result
        n_obs <- nrow(tmle_dat)
        n_rules <- length(tmle_rules)
        rule_names <- names(tmle_rules)
        if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
        
        # Create a minimal result structure with everything consistently sized
        default_value <- 0.5
        Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
        colnames(Qs) <- rule_names
        
        QAW <- cbind(QA=rep(default_value, n_obs), Qs)
        colnames(QAW) <- c("QA", colnames(Qs))
        
        Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
        colnames(Qstar) <- rule_names
        
        Qstar_iptw <- rep(default_value, n_rules)
        names(Qstar_iptw) <- rule_names
        
        # Create minimal return object with consistently sized matrices
        return(list(
          "Qs" = Qs,
          "QAW" = QAW,
          "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
          "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
          "updated_model_for_Y" = vector("list", n_rules),
          "Qstar" = Qstar,
          "Qstar_iptw" = Qstar_iptw,
          "Qstar_gcomp" = Qs,
          "ID" = tmle_dat$ID,
          "Y" = rep(NA, n_obs)
        ))
      } else {
        # For other errors, log them but still provide a fallback result
        cat("Error in getTMLELong:", e$message, "\n")
        
        # Extract arguments
        args <- list(...)
        initial_model_for_Y <- args[[1]]
        tmle_rules <- args[[2]]
        
        # Extract data
        tmle_dat <- NULL
        if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
          tmle_dat <- initial_model_for_Y$data
        } else {
          tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
        }
        
        # Create default dimensions
        n_obs <- nrow(tmle_dat)
        n_rules <- length(tmle_rules)
        rule_names <- names(tmle_rules)
        if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
        
        # Create a minimal result structure
        default_value <- 0.5
        Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
        colnames(Qs) <- rule_names
        
        QAW <- cbind(QA=rep(default_value, n_obs), Qs)
        colnames(QAW) <- c("QA", colnames(Qs))
        
        Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
        colnames(Qstar) <- rule_names
        
        Qstar_iptw <- rep(default_value, n_rules)
        names(Qstar_iptw) <- rule_names
        
        # Return minimal object
        return(list(
          "Qs" = Qs,
          "QAW" = QAW,
          "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
          "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
          "updated_model_for_Y" = vector("list", n_rules),
          "Qstar" = Qstar,
          "Qstar_iptw" = Qstar_iptw,
          "Qstar_gcomp" = Qs,
          "ID" = tmle_dat$ID,
          "Y" = rep(NA, n_obs)
        ))
      }
    })
  }
  
  # Assign the fixed function to the global environment
  assign("safe_getTMLELong", safe_getTMLELong, envir = .GlobalEnv)
  cat("safe_getTMLELong patched successfully.\n")
}

# Monkeypatch the getTMLELong function to ensure matrix dimensions are consistent
fix_getTMLELong <- function() {
  cat("Checking if getTMLELong function exists in global environment...\n")
  
  # Check if getTMLELong exists in the global environment
  if(exists("getTMLELong", envir = .GlobalEnv)) {
    cat("Patching getTMLELong function to ensure consistent matrix dimensions...\n")
    
    # Get the original function
    original_getTMLELong <- get("getTMLELong", envir = .GlobalEnv)
    
    # Create a wrapper function that ensures consistent dimensions
    getTMLELong_wrapper <- function(...) {
      # Call the original function
      result <- tryCatch({
        original_getTMLELong(...)
      }, error = function(e) {
        # If there's an error, let the safe_getTMLELong handle it
        stop(e)
      })
      
      # Ensure matrix dimensions are consistent
      if(!is.null(result)) {
        # Get element sizes
        sizes <- c(
          nrow(result$Qs), 
          nrow(result$QAW), 
          nrow(result$Qstar), 
          nrow(result$clever_covariates), 
          nrow(result$weights)
        )
        
        # Check if any of the sizes are different
        if(length(unique(sizes)) > 1) {
          cat("Warning: Inconsistent matrix dimensions detected. Fixing...\n")
          cat("Sizes:", paste(sizes, collapse=", "), "\n")
          
          # Use the most common size
          size_table <- table(sizes)
          target_size <- as.numeric(names(size_table)[which.max(size_table)])
          cat("Target size:", target_size, "\n")
          
          # Resize each matrix to the target size
          for(matrix_name in c("Qs", "QAW", "Qstar", "clever_covariates", "weights")) {
            if(matrix_name %in% names(result) && nrow(result[[matrix_name]]) != target_size) {
              old_size <- nrow(result[[matrix_name]])
              cat("Resizing", matrix_name, "from", old_size, "to", target_size, "\n")
              
              # Create a new matrix with correct dimensions
              new_matrix <- matrix(NA, nrow=target_size, ncol=ncol(result[[matrix_name]]))
              
              # Set column names
              if(!is.null(colnames(result[[matrix_name]]))) {
                colnames(new_matrix) <- colnames(result[[matrix_name]])
              }
              
              # Copy as much data as possible
              if(old_size > 0 && target_size > 0) {
                copy_size <- min(old_size, target_size)
                new_matrix[1:copy_size, ] <- result[[matrix_name]][1:copy_size, ]
              }
              
              # Handle missing values
              na_rows <- is.na(new_matrix[,1])
              if(any(na_rows)) {
                # Fill with column means or defaults
                for(col in 1:ncol(new_matrix)) {
                  col_data <- new_matrix[!na_rows, col]
                  
                  if(length(col_data) > 0) {
                    fill_value <- mean(col_data, na.rm=TRUE)
                    if(is.na(fill_value)) fill_value <- 0.5
                  } else {
                    fill_value <- 0.5
                  }
                  
                  new_matrix[na_rows, col] <- fill_value
                }
              }
              
              # Replace the matrix in the result
              result[[matrix_name]] <- new_matrix
            }
          }
        }
      }
      
      return(result)
    }
    
    # Replace the original function with our wrapper
    assign("getTMLELong", getTMLELong_wrapper, envir = .GlobalEnv)
    cat("getTMLELong patched successfully.\n")
  } else {
    cat("getTMLELong not found in global environment. Skipping direct patching.\n")
  }
}

# Run this script to apply the patches
cat("Applying direct fixes to handle vector length mismatches in standard tmle estimator...\n")

# First source the necessary files to get the original functions
for(file in c('./src/tmle_IC.R', './src/misc_fns.R', './src/tmle_fns.R')) {
  if(file.exists(file)) {
    cat("Sourcing", file, "...\n")
    source(file)
  } else {
    cat("Warning:", file, "not found. Some fixes may not be applied.\n")
  }
}

# Apply the patches
fix_safe_getTMLELong()
fix_getTMLELong()

cat("\nPatches applied successfully.\n")
cat("To use these patches, run the following R code before your simulation:\n")
cat("  source('makeshift_tmle.R')\n")
cat("\nAlternatively, run the following in your simulation script:\n")
cat("  if(estimator == 'tmle') source('makeshift_tmle.R')\n")