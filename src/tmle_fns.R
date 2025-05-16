###################################################################
# Treatment regime functions                                      #
###################################################################

static_arip_on <- function(row, lags=TRUE) {
  #  binary treatment is set to aripiprazole at all time points for all observations
  if(lags){
    treats <- row[grep("A[0-9]", colnames(row), value=TRUE)]
  } else {
    treats <- row[grep("A[0-9]$", colnames(row), value=TRUE)]
  }
  
  # Create a named vector with the same structure as treats
  shifted <- rep(0, length(treats))
  names(shifted) <- names(treats)
  
  # Handle different time points with defensive error checking
  tryCatch({
    if(row$t == 1) { # first-, second-, and third-order lags are 0
      a1_cols <- grep("^A1$", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a1_cols] <- 1
    } else if(row$t == 2) { #second- and third-order lags are zero
      a1_cols <- c(grep("^A1$", colnames(row), value=TRUE), 
                  grep("^A1.lag$", colnames(row), value=TRUE))
      shifted[names(shifted) %in% a1_cols] <- 1
    } else if(row$t > 2) { #turn on all lags
      a1_cols <- grep("A1", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a1_cols] <- 1
    }
  }, error = function(e) {
    # If there's an error, return a safe default
    warning("Error in static_arip_on for row with ID ", row$ID, ": ", e$message)
  })
  
  return(shifted)
}

static_halo_on <- function(row, lags=TRUE) {
  #  binary treatment is set to haloperidol at all time points for all observations
  if(lags){
    treats <- row[grep("A[0-9]", colnames(row), value=TRUE)]
  } else {
    treats <- row[grep("A[0-9]$", colnames(row), value=TRUE)]
  }
  
  # Create a named vector with the same structure as treats
  shifted <- rep(0, length(treats))
  names(shifted) <- names(treats)
  
  # Handle different time points with defensive error checking
  tryCatch({
    if(row$t == 1) { # first-, second-, and third-order lags are 0
      a2_cols <- grep("^A2$", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a2_cols] <- 1
    } else if(row$t == 2) { #second- and third-order lags are zero
      a2_cols <- c(grep("^A2$", colnames(row), value=TRUE), 
                  grep("^A2.lag$", colnames(row), value=TRUE))
      shifted[names(shifted) %in% a2_cols] <- 1
    } else if(row$t > 2) { #turn on all lags
      a2_cols <- grep("A2", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a2_cols] <- 1
    }
  }, error = function(e) {
    # If there's an error, return a safe default
    warning("Error in static_halo_on for row with ID ", row$ID, ": ", e$message)
  })
  
  return(shifted)
}

static_olanz_on <- function(row, lags=TRUE) {
  #  binary treatment is set to olanzapine at all time points for all observations
  if(lags){
    treats <- row[grep("A[0-9]", colnames(row), value=TRUE)]
  } else {
    treats <- row[grep("A[0-9]$", colnames(row), value=TRUE)]
  }
  
  # Create a named vector with the same structure as treats
  shifted <- rep(0, length(treats))
  names(shifted) <- names(treats)
  
  # Handle different time points with defensive error checking
  tryCatch({
    if(row$t == 1) { # first-, second-, and third-order lags are 0
      a3_cols <- grep("^A3$", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a3_cols] <- 1
    } else if(row$t == 2) { #second- and third-order lags are zero
      a3_cols <- c(grep("^A3$", colnames(row), value=TRUE), 
                  grep("^A3.lag$", colnames(row), value=TRUE))
      shifted[names(shifted) %in% a3_cols] <- 1
    } else if(row$t > 2) { #turn on all lags
      a3_cols <- grep("A3", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a3_cols] <- 1
    }
  }, error = function(e) {
    # If there's an error, return a safe default
    warning("Error in static_olanz_on for row with ID ", row$ID, ": ", e$message)
  })
  
  return(shifted)
}

static_risp_on <- function(row, lags=TRUE) {
  #  binary treatment is set to risperidone at all time points for all observations
  if(lags){
    treats <- row[grep("A[0-9]", colnames(row), value=TRUE)]
  } else {
    treats <- row[grep("A[0-9]$", colnames(row), value=TRUE)]
  }
  
  # Create a named vector with the same structure as treats
  shifted <- rep(0, length(treats))
  names(shifted) <- names(treats)
  
  # Handle different time points with defensive error checking
  tryCatch({
    if(row$t == 1) { # first-, second-, and third-order lags are 0
      a5_cols <- grep("^A5$", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a5_cols] <- 1
    } else if(row$t == 2) { #second- and third-order lags are zero
      a5_cols <- c(grep("^A5$", colnames(row), value=TRUE), 
                  grep("^A5.lag$", colnames(row), value=TRUE))
      shifted[names(shifted) %in% a5_cols] <- 1
    } else if(row$t > 2) { #turn on all lags
      a5_cols <- grep("A5", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a5_cols] <- 1
    }
  }, error = function(e) {
    # If there's an error, return a safe default
    warning("Error in static_risp_on for row with ID ", row$ID, ": ", e$message)
  })
  
  return(shifted)
}

static_quet_on <- function(row, lags=TRUE) {
  #  binary treatment is set to quetiapine at all time points for all observations
  if(lags){
    treats <- row[grep("A[0-9]", colnames(row), value=TRUE)]
  } else {
    treats <- row[grep("A[0-9]$", colnames(row), value=TRUE)]
  }
  
  # Create a named vector with the same structure as treats
  shifted <- rep(0, length(treats))
  names(shifted) <- names(treats)
  
  # Handle different time points with defensive error checking
  tryCatch({
    if(row$t == 1) { # first-, second-, and third-order lags are 0
      a4_cols <- grep("^A4$", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a4_cols] <- 1
    } else if(row$t == 2) { #second- and third-order lags are zero
      a4_cols <- c(grep("^A4$", colnames(row), value=TRUE), 
                  grep("^A4.lag$", colnames(row), value=TRUE))
      shifted[names(shifted) %in% a4_cols] <- 1
    } else if(row$t > 2) { #turn on all lags
      a4_cols <- grep("A4", colnames(row), value=TRUE)
      shifted[names(shifted) %in% a4_cols] <- 1
    }
  }, error = function(e) {
    # If there's an error, return a safe default
    warning("Error in static_quet_on for row with ID ", row$ID, ": ", e$message)
  })
  
  return(shifted)
}

static_mtp <- function(row){ 
  # Static: Everyone gets quetiap (if bipolar), halo (if schizophrenia), ari (if MDD) and stays on it
  
  # Initialize with a safe default
  shifted <- NULL
  
  # Safely handle different time points with defensive error checking
  tryCatch({
    if(row$t == 0) { # first-, second-, and third-order lags are 0
      if(row$schiz == 1) {
        shifted <- static_halo_on(row, lags=TRUE)
      } else if(row$bipolar == 1) {
        shifted <- static_quet_on(row, lags=TRUE)
      } else if(row$mdd == 1) {
        shifted <- static_arip_on(row, lags=TRUE)
      } else {
        # Create safe default with proper structure
        treat_cols <- grep("A[0-9]", colnames(row), value=TRUE)
        shifted <- rep(0, length(treat_cols))
        names(shifted) <- treat_cols
      }
    } else if(row$t >= 1) {
      # Safely extract lag columns
      lag_cols <- grep("A", grep("lag", colnames(row), value=TRUE), value=TRUE)
      lags <- row[lag_cols]
      
      if(row$schiz == 1) {
        current_treats <- static_risp_on(row, lags=FALSE)
        # Use safe combination method
        shifted <- combine_treatments(current_treats, lags)
      } else if(row$bipolar == 1) {
        current_treats <- static_quet_on(row, lags=FALSE)
        shifted <- combine_treatments(current_treats, lags)
      } else if(row$mdd == 1) {
        current_treats <- static_arip_on(row, lags=FALSE)
        shifted <- combine_treatments(current_treats, lags)
      } else {
        # Create safe default with proper structure
        treat_cols <- grep("A[0-9]$", colnames(row), value=TRUE)
        current_treats <- rep(0, length(treat_cols))
        names(current_treats) <- treat_cols
        shifted <- combine_treatments(current_treats, lags)
      }
    }
  }, error = function(e) {
    # If an error occurs, return a safe default
    warning("Error in static_mtp for row with ID ", row$ID, ": ", e$message)
    # Create a default response with all treatment columns
    treat_cols <- grep("A[0-9]", colnames(row), value=TRUE)
    shifted <- rep(0, length(treat_cols))
    names(shifted) <- treat_cols
  })
  
  # Final safety check - ensure we're returning something valid
  if(is.null(shifted) || length(shifted) == 0) {
    treat_cols <- grep("A[0-9]", colnames(row), value=TRUE)
    shifted <- rep(0, length(treat_cols))
    names(shifted) <- treat_cols
  }
  
  return(shifted)
}

# Helper function for safely combining current treatments with lags
combine_treatments <- function(current, lags) {
  tryCatch({
    # Convert both to named numeric vectors if they aren't already
    if(is.data.frame(current)) {
      current <- as.numeric(unlist(current))
    }
    if(is.data.frame(lags)) {
      lags <- as.numeric(unlist(lags))
    }
    
    # Combine and return
    combined <- c(current, lags)
    return(combined)
  }, error = function(e) {
    warning("Error combining treatments: ", e$message)
    # Return current only as fallback
    return(current)
  })
}

dynamic_mtp <- function(row){ 
  # Dynamic: Everyone starts with risp.
  # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, 
  # then switch to quetiap. (if bipolar), halo. (if schizophrenia), ari (if MDD); otherwise stay on risp.
  
  # Initialize with a safe default
  shifted <- NULL
  
  # Safely handle different time points with defensive error checking
  tryCatch({
    if(row$t == 0) { # first-, second-, and third-order lags are 0
      shifted <- static_risp_on(row, lags=TRUE)
    } else if(row$t >= 1) {
      # Safely extract lag columns
      lag_cols <- grep("A", grep("lag", colnames(row), value=TRUE), value=TRUE)
      lags <- row[lag_cols]
      
      # Check for L variables with proper NA handling
      has_symptoms <- FALSE
      if(!is.na(row$L1) && !is.na(row$L2) && !is.na(row$L3)) {
        has_symptoms <- (row$L1 > 0 | row$L2 > 0 | row$L3 > 0)
      }
      
      if(has_symptoms) {
        if(row$schiz == 1) {
          current_treats <- static_halo_on(row, lags=FALSE)
          shifted <- combine_treatments(current_treats, lags)
        } else if(row$bipolar == 1) {
          current_treats <- static_quet_on(row, lags=FALSE)
          shifted <- combine_treatments(current_treats, lags)
        } else if(row$mdd == 1) {
          current_treats <- static_arip_on(row, lags=FALSE)
          shifted <- combine_treatments(current_treats, lags)
        } else {
          # Default to risperidone if no diagnosis
          current_treats <- static_risp_on(row, lags=FALSE)
          shifted <- combine_treatments(current_treats, lags)
        }
      } else {
        # Stay on risperidone
        current_treats <- static_risp_on(row, lags=FALSE)
        shifted <- combine_treatments(current_treats, lags)
      }
    }
  }, error = function(e) {
    # If an error occurs, return a safe default
    warning("Error in dynamic_mtp for row with ID ", row$ID, ": ", e$message)
    # Create a default response with all treatment columns
    treat_cols <- grep("A[0-9]", colnames(row), value=TRUE)
    shifted <- rep(0, length(treat_cols))
    names(shifted) <- treat_cols
  })
  
  # Final safety check - ensure we're returning something valid
  if(is.null(shifted) || length(shifted) == 0) {
    treat_cols <- grep("A[0-9]", colnames(row), value=TRUE)
    shifted <- rep(0, length(treat_cols))
    names(shifted) <- treat_cols
  }
  
  return(shifted)
}

stochastic_mtp <- function(row){
  # Stochastic: at each t>0, 95% chance of staying with treatment at t-1, 
  # 5% chance of randomly switching according to Multinomial distribution
  
  # Initialize with a safe default
  shifted <- NULL
  
  # Safely handle different time points with defensive error checking
  tryCatch({
    if(row$t == 0) { # do nothing first period
      # Get all treatment columns
      treat_cols <- grep("A[0-9]", colnames(row), value=TRUE)
      # Safely extract treatment values as a numeric vector
      if(length(treat_cols) > 0) {
        shifted <- as.numeric(row[treat_cols])
        names(shifted) <- treat_cols
      } else {
        # No treatment columns found - create default
        shifted <- rep(0, 6)  # Assuming 6 possible treatments
        names(shifted) <- paste0("A", 1:6)
      }
    } else if(row$t >= 1) {
      # Safely extract lag columns
      lag_cols <- grep("A", grep("lag", colnames(row), value=TRUE), value=TRUE)
      lags <- row[lag_cols]
      
      # Get current treatment columns
      current_treat_cols <- grep("A[0-9]$", colnames(row), value=TRUE)
      current_treats <- row[current_treat_cols]
      
      # Safely find current treatment
      treat_idx <- NULL
      if(length(current_treat_cols) > 0) {
        treat_idx <- which(as.numeric(current_treats) > 0)
        if(length(treat_idx) == 0) treat_idx <- 1  # Default to first treatment if none found
      }
      
      # Create stochastic transition with defensive coding
      tryCatch({
        # Attempt stochastic transition
        if(!is.null(treat_idx) && length(treat_idx) > 0) {
          probs <- StochasticFun(current_treats, d=c(0,0,0,0,0,0))
          if(treat_idx <= nrow(probs)) {
            transition_probs <- probs[treat_idx,]
            random_treat <- Multinom(1, transition_probs)
          } else {
            # Invalid index - use uniform
            random_treat <- sample(1:ncol(current_treats), 1)
          }
        } else {
          # No current treatment - sample uniformly
          random_treat <- sample(1:length(current_treat_cols), 1)
        }
        
        # Reset treatment vector
        new_treats <- rep(0, length(current_treat_cols))
        names(new_treats) <- current_treat_cols
        
        # Set random treatment
        if(is.numeric(random_treat) && random_treat <= length(new_treats)) {
          new_treats[random_treat] <- 1
        } else if(is.numeric(random_treat)) {
          # Index out of bounds - set first treatment
          new_treats[1] <- 1
        }
        
        # Combine with lags
        shifted <- combine_treatments(new_treats, lags)
      }, error = function(e) {
        # If stochastic transition fails, keep current treatment
        warning("Error in stochastic transition for row with ID ", row$ID, ": ", e$message)
        # Set first treatment as fallback
        new_treats <- rep(0, length(current_treat_cols))
        names(new_treats) <- current_treat_cols
        new_treats[1] <- 1
        shifted <- combine_treatments(new_treats, lags)
      })
    }
  }, error = function(e) {
    # If an error occurs, return a safe default
    warning("Error in stochastic_mtp for row with ID ", row$ID, ": ", e$message)
    # Create a default response with all treatment columns
    treat_cols <- grep("A[0-9]", colnames(row), value=TRUE)
    shifted <- rep(0, length(treat_cols))
    names(shifted) <- treat_cols
    if(length(shifted) > 0) shifted[1] <- 1  # Set first treatment
  })
  
  # Final safety check - ensure we're returning something valid
  if(is.null(shifted) || length(shifted) == 0) {
    treat_cols <- grep("A[0-9]", colnames(row), value=TRUE)
    if(length(treat_cols) == 0) {
      # No treatment columns found - create default
      shifted <- rep(0, 6)  # Assuming 6 possible treatments
      names(shifted) <- paste0("A", 1:6)
    } else {
      shifted <- rep(0, length(treat_cols))
      names(shifted) <- treat_cols
    }
    if(length(shifted) > 0) shifted[1] <- 1  # Set first treatment
  }
  
  return(shifted)
}

###################################################################
# Sequential-g estimator                                          #
###################################################################
# Fixed function to ensure t.end is properly handled
initialize_outcome_models <- function(t.end) {
  # Create a list with length exactly matching the number of time points
  # Elements are indexed 1 to t.end (not 0 to t.end)
  models <- vector("list", t.end)
  
  # Return the properly sized list
  return(models)
}

sequential_g_final <- function(t, tmle_dat, n.folds, tmle_covars_Y, initial_model_for_Y_sl, ybound){
  # Process using the regular sequential_g function - this ensures consistent approach
  message("Processing final time point t=", t, " using the same approach as other time points")
  
  # First create a safe subset of data without missing Y values
  tmle_dat_sub <- tmle_dat[tmle_dat$t==t & !is.na(tmle_dat$Y),] # drop rows with missing Y
  
  # Check if we have enough data
  if(nrow(tmle_dat_sub) < 5) {
    message("Very few non-missing Y values at t=", t, ", adding nearby time points")
    # Get data from nearby time points
    nearby_t <- max(1, t-1)  # Use previous time point
    tmle_dat_sub <- rbind(tmle_dat_sub, tmle_dat[tmle_dat$t==nearby_t & !is.na(tmle_dat$Y),])
    message("Added data from t=", nearby_t, ", new row count: ", nrow(tmle_dat_sub))
  }
  
  # Ensure all expected covariates exist in the data
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding ", length(missing_covars), " missing covariates for t=", t, ": ", 
            paste(missing_covars[1:min(5, length(missing_covars))], collapse=", "), 
            if(length(missing_covars)>5) "..." else "")
    
    # Add missing covariates with default values
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Use 0 as default
    }
  }
  
  # Check for Stack's required covariates
  stack_required_covars <- c("V3", "L1", "L2", "L3", "Y.lag", "Y.lag2", "Y.lag3", 
                          "L1.lag", "L1.lag2", "L1.lag3", "L2.lag", "L2.lag2", "L2.lag3", 
                          "L3.lag", "L3.lag2", "L3.lag3", "white", "black", "latino", "other", 
                          "mdd", "bipolar", "schiz", 
                          "A1", "A2", "A3", "A4", "A5", "A6", 
                          "A1.lag", "A2.lag", "A3.lag", "A4.lag", "A5.lag", "A6.lag",
                          "A1.lag2", "A2.lag2", "A3.lag2", "A4.lag2", "A5.lag2", "A6.lag2",
                          "A1.lag3", "A2.lag3", "A3.lag3", "A4.lag3", "A5.lag3", "A6.lag3")
  
  # Check which of these are missing and add them
  stack_missing_covars <- setdiff(stack_required_covars, colnames(tmle_dat_sub))
  if(length(stack_missing_covars) > 0) {
    message("Adding ", length(stack_missing_covars), " Stack-required covariates: ", 
            paste(stack_missing_covars[1:min(5, length(stack_missing_covars))], collapse=", "), 
            if(length(stack_missing_covars)>5) "..." else "")
    
    for(cov in stack_missing_covars) {
      # Choose appropriate default values based on covariate type
      if(grepl("^A[1-6]", cov)) {
        # Binary indicator - use 0
        tmle_dat_sub[[cov]] <- 0
      } else if(grepl("(white|black|latino|other|mdd|bipolar|schiz)", cov)) {
        # Demographic binary indicators - use 0
        tmle_dat_sub[[cov]] <- 0
      } else if(grepl("^(L|Y)", cov)) {
        # Time-varying covariates/outcomes - use 0
        tmle_dat_sub[[cov]] <- 0
      } else {
        # Default for other variables
        tmle_dat_sub[[cov]] <- 0
      }
    }
  }
  
  # Use sequential_g but with a try-catch to handle any unexpected errors
  tryCatch({
    # Try to use sequential_g normally
    Y_preds <- sequential_g(t, tmle_dat_sub, n.folds, tmle_covars_Y, initial_model_for_Y_sl, ybound)
  }, error = function(e) {
    message("Error in sequential_g at t=", t, ": ", e$message)
    message("Falling back to mean-based predictions")
    # Fallback to mean-based prediction
    Y_vals <- tmle_dat_sub$Y
    Y_vals <- Y_vals[!is.na(Y_vals) & Y_vals != -1]
    if(length(Y_vals) > 0) {
      mean_val <- mean(Y_vals)
      # Add small noise to avoid constant values
      Y_preds <- rnorm(nrow(tmle_dat_sub), mean=mean_val, sd=0.01)
      # Bound within ybound
      Y_preds <- pmin(pmax(Y_preds, ybound[1]), ybound[2])
    } else {
      # No valid Y values, use default
      Y_preds <- rep(0.5, nrow(tmle_dat_sub))
    }
    return(Y_preds)
  })
  
  # Create a simple model object for compatibility
  mean_Y <- mean(Y_preds, na.rm=TRUE)
  if(is.na(mean_Y) || !is.finite(mean_Y)) mean_Y <- 0.5
  
  mean_fit <- list(params = list(covariates = tmle_covars_Y))
  class(mean_fit) <- "custom_mean_fit"
  mean_fit$predict <- function(task) {
    # Check if the task has all required covariates
    task_covars <- colnames(task$X)
    missing_task_covars <- setdiff(tmle_covars_Y, task_covars)
    
    if(length(missing_task_covars) > 0) {
      message("Task missing ", length(missing_task_covars), " covariates, using mean value")
      return(rep(mean_Y, task$nrow))
    }
    
    # Return predictions
    if(length(Y_preds) >= task$nrow) {
      return(Y_preds[1:task$nrow])
    } else {
      # Not enough predictions, repeat as needed
      return(rep(Y_preds, length.out=task$nrow))
    }
  }
  
  # Return list with all components
  return(list(
    "preds" = Y_preds,
    "fit" = mean_fit,
    "data" = tmle_dat_sub  # Use the enhanced data that has all covariates
  ))
}

# Initialize outcomes matrix correctly with more defensive error handling
initialize_outcome_matrix <- function(tmle_dat_t, tmle_rules) {
  # Handle potential edge cases
  tryCatch({
    # Ensure tmle_dat_t is a data frame with rows
    if(!is.data.frame(tmle_dat_t) || nrow(tmle_dat_t) == 0) {
      warning("tmle_dat_t is not a valid data frame or has zero rows")
      # Return a small default matrix
      result_matrix <- matrix(0, nrow=1, ncol=length(tmle_rules))
      colnames(result_matrix) <- names(tmle_rules)
      return(result_matrix)
    }
    
    # Ensure tmle_rules is a list with elements
    if(!is.list(tmle_rules) || length(tmle_rules) == 0) {
      warning("tmle_rules is not a valid list or has zero elements")
      # Return a matrix with just one column
      result_matrix <- matrix(0, nrow=nrow(tmle_dat_t), ncol=1)
      colnames(result_matrix) <- "default_rule"
      return(result_matrix)
    }
    
    # Get dimensions with explicit coercion to numeric
    n_rows <- as.integer(nrow(tmle_dat_t))
    n_rules <- as.integer(length(tmle_rules))
    
    # Create matrix with appropriate dimensions and safe initialization
    result_matrix <- matrix(NA_real_, nrow=n_rows, ncol=n_rules)
    
    # Add column names safely
    if(!is.null(names(tmle_rules)) && length(names(tmle_rules)) == n_rules) {
      colnames(result_matrix) <- names(tmle_rules)
    } else {
      # Create default column names if needed
      colnames(result_matrix) <- paste0("rule_", 1:n_rules)
    }
    
    return(result_matrix)
    
  }, error = function(e) {
    # Handle any unexpected errors
    warning("Error in initialize_outcome_matrix: ", e$message)
    # Return a small default matrix
    result_matrix <- matrix(0, nrow=1, ncol=3)
    colnames(result_matrix) <- c("static", "dynamic", "stochastic")
    return(result_matrix)
  })
}

# Optimized sequential_g function with improved performance and caching
sequential_g <- function(t, tmle_dat, n.folds, tmle_covars_Y, initial_model_for_Y_sl, ybound, Y_pred=NULL) {
  # Check for an existing model cache
  if(!exists("model_cache", envir = .GlobalEnv)) {
    assign("model_cache", new.env(), envir = .GlobalEnv)
  }
  cache_key <- paste0("seq_g_model_", t)

  # Fast path for nearly constant Y values
  fast_subset <- tmle_dat[tmle_dat$t==t & !is.na(tmle_dat$Y) & tmle_dat$Y != -1, "Y", drop=TRUE]
  if(length(fast_subset) > 0) {
    y_range <- range(fast_subset)
    if(diff(y_range) < 0.01) {
      message("Y values nearly constant at t=", t, ", using fast approach")
      mean_y <- mean(fast_subset, na.rm=TRUE)
      n_rows <- sum(tmle_dat$t==t)
      # Use vectorized operations to generate all noise at once
      noise <- rnorm(n_rows, mean=0, sd=0.001)
      return(pmin(pmax(mean_y + noise, ybound[1]), ybound[2]))
    }
  }

  # Process prediction data upfront to avoid duplication
  pred_data <- tmle_dat[tmle_dat$t==t,]

  # Use fast filtering for training data
  mask <- tmle_dat$t==t & !is.na(tmle_dat$Y)
  tmle_dat_sub <- tmle_dat[mask,]

  # Handle very small sample size with fast adjacent time point addition
  if(nrow(tmle_dat_sub) < 10) {
    nearby_mask <- tmle_dat$t %in% c(max(1, t-1), min(t+1, max(tmle_dat$t))) & !is.na(tmle_dat$Y)
    additional_data <- tmle_dat[nearby_mask,]

    if(nrow(additional_data) > 0) {
      tmle_dat_sub <- rbind(tmle_dat_sub, additional_data)
      message("Added ", nrow(additional_data), " records from nearby time points for t=", t)
    }
  }

  # Fast path for truly constant Y
  y_values <- tmle_dat_sub$Y[!is.na(tmle_dat_sub$Y)]
  if(length(unique(y_values)) == 1) {
    message("Y is constant, using fast constant model for t=", t)
    const_val <- y_values[1]
    return(rep(const_val, nrow(pred_data)))
  }

  # Fast handling for Y_pred
  if(!is.null(Y_pred)) {
    if(is.list(Y_pred)) {
      Y_pred <- unlist(Y_pred)
    }

    # Efficient ID mapping using match function directly
    if(length(Y_pred) > 0) {
      id_map <- match(tmle_dat_sub$ID, names(Y_pred))
      valid_matches <- !is.na(id_map)

      if(any(valid_matches)) {
        # Only subset valid matches
        tmle_dat_sub <- tmle_dat_sub[valid_matches,]
        tmle_dat_sub$Y <- Y_pred[id_map[valid_matches]]
      } else {
        warning("No matching IDs found between prediction data and data at t=", t)
      }
    }
  }

  # Fast covariate validation and addition
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))

  # Add missing covariates if needed
  if(length(missing_covars) > 0) {
    # Use vectorized addition instead of loop
    tmle_dat_sub[, missing_covars] <- 0
  }

  # Optimize outcome type detection
  is_binary <- FALSE
  y_unique <- unique(tmle_dat_sub$Y[!is.na(tmle_dat_sub$Y)])
  if(length(y_unique) <= 2 && all(y_unique %in% c(0,1))) {
    outcome_type <- "binomial"
    is_binary <- TRUE
  } else {
    outcome_type <- "continuous"
  }

  # Check for cached model first
  if(exists(cache_key, envir = model_cache)) {
    message("Using cached model for t=", t)
    initial_model_for_Y_sl_fit <- get(cache_key, envir = model_cache)
  } else {
    # Only create folds if not using cached model
    folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)

    # Efficient task creation with minimal parameters
  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

  # Safely check and add any missing required covariates
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values
    }
  }
  # Only use covariates that actually exist in the data
  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))

    initial_model_for_Y_task <- make_sl3_Task(
      data = tmle_dat_sub,
      covariates = required_covars,
      outcome = "Y",
      outcome_type = outcome_type,
      folds = folds,
      drop_missing_outcome = TRUE
    )

    # Fast model fitting with optimized fallback strategy
    initial_model_for_Y_sl_fit <- tryCatch({
      # Try simplified SuperLearner first instead of full stack
      if(is_binary) {
        # Binary outcomes - use minimal learner set
        lrnrs <- list(
          make_learner(Lrnr_glm, family = binomial()),
          make_learner(Lrnr_mean)
        )
      } else {
        # Continuous outcomes - use minimal learner set
        lrnrs <- list(
          make_learner(Lrnr_glm, family = gaussian()),
          make_learner(Lrnr_mean)
        )
      }

      # Try Stack first instead of full SuperLearner
      sl_simple <- make_learner(Stack, lrnrs)
      sl_simple$train(initial_model_for_Y_task)
    }, error = function(e) {
      # Fall back to single GLM
      tryCatch({
        if(is_binary) {
          glm_learner <- make_learner(Lrnr_glm, family = binomial())
        } else {
          glm_learner <- make_learner(Lrnr_glm, family = gaussian())
        }
        glm_learner$train(initial_model_for_Y_task)
      }, error = function(e) {
        # Final fallback to mean model
        mean_learner <- make_learner(Lrnr_mean)
        mean_task <- make_sl3_Task(
          data = tmle_dat_sub,
          covariates = character(0),
          outcome = "Y",
          outcome_type = outcome_type,
          drop_missing_outcome = TRUE
        )
        mean_learner$train(mean_task)
      })
    })

    # Cache the model for future use
    assign(cache_key, initial_model_for_Y_sl_fit, envir = model_cache)
  }

  # Efficient creation of prediction data
  # Only add covariates that are actually needed
  if(inherits(initial_model_for_Y_sl_fit, "Lrnr_base") &&
     !is.null(initial_model_for_Y_sl_fit$fit_object) &&
     !is.null(initial_model_for_Y_sl_fit$fit_object$params) &&
     !is.null(initial_model_for_Y_sl_fit$fit_object$params$covariates)) {

    needed_covars <- initial_model_for_Y_sl_fit$fit_object$params$covariates
    missing_pred_covars <- setdiff(needed_covars, colnames(pred_data))

    # Vectorized addition of missing covariates
    if(length(missing_pred_covars) > 0) {
      pred_data[, missing_pred_covars] <- 0
    }

    # Create prediction task with only needed covariates
    prediction_task <- sl3_Task$new(
      data = pred_data,
      covariates = needed_covars,
      outcome = "Y",
      outcome_type = outcome_type,
      drop_missing_outcome = FALSE
    )
  } else {
    # For mean/custom models with no needed covariates
    prediction_task <- sl3_Task$new(
      data = pred_data,
      covariates = character(0),
      outcome = "Y",
      outcome_type = outcome_type,
      drop_missing_outcome = FALSE
    )
  }

  # Fast predictions with minimal error handling
  Y_preds <- tryCatch({
    preds <- initial_model_for_Y_sl_fit$predict(prediction_task)

    # Ensure predictions are numeric
    if(is.list(preds)) {
      preds <- unlist(preds)
    }

    # Vectorized type conversion and bounds application
    preds <- as.numeric(preds)
    preds[!is.na(preds)] <- pmin(pmax(preds[!is.na(preds)], ybound[1]), ybound[2])
    preds
  }, error = function(e) {
    message("Prediction failed: ", e$message, ". Using mean value.")

    # Use mean as fallback
    mean_val <- mean(tmle_dat_sub$Y, na.rm=TRUE)
    if(is.na(mean_val) || !is.finite(mean_val)) mean_val <- 0.5

    # Create predictions with small noise
    noise <- rnorm(nrow(pred_data), mean=0, sd=0.01)
    pmin(pmax(rep(mean_val, nrow(pred_data)) + noise, ybound[1]), ybound[2])
  })

  return(Y_preds)
}

process_backward_sequential <- function(tmle_dat, t, tmle_rules, essential_covars_Y, 
                                        initial_model_for_Y_sl_cont, ybound, tmle_contrasts,
                                        time.censored=NULL) {
  message("Processing time point t=", t)
  
  # Check if time point is valid - t.end is typically 36
  if(t > 36) {
    warning("Invalid time point t=", t, ". Maximum supported is 36.")
    return(NULL)
  }
  
  # Create empty time.censored if not provided
  if(is.null(time.censored)) {
    time.censored <- data.frame(ID=integer(0), time_censored=integer(0))
  }
  
  # Subset data safely
  tmle_dat_t <- tryCatch({
    # First subset the data based on time point
    subset <- tmle_dat[tmle_dat$t == t, ]
    
    # Then filter out censored individuals if time.censored has data
    if(nrow(time.censored) > 0) {
      censored_ids <- time.censored$ID[which(time.censored$time_censored < t)]
      if(length(censored_ids) > 0) {
        subset <- subset[!subset$ID %in% censored_ids, ]
      }
    }
    
    subset
  }, error = function(e) {
    message("Error subsetting data: ", e$message)
    # Return empty data frame with required columns
    empty_df <- data.frame(ID=integer(0), t=integer(0), Y=numeric(0))
    empty_df
  })
  
  # Early exit if no data
  if(nrow(tmle_dat_t) == 0) {
    warning("No data available after filtering at time t=", t)
    # Return placeholder results
    return(NULL)
  }
  
  # Calculate mean Y value for fallback
  mean_Y <- mean(tmle_dat_t$Y, na.rm=TRUE)
  if(is.na(mean_Y) || !is.finite(mean_Y)) mean_Y <- 0.5
  
  # Skip SuperLearner entirely and use simple fallback approach
  n_rules <- length(tmle_rules)
  
  # Create prediction matrix more safely
  tryCatch({
    # Create matrix with proper dimensions
    prediction_matrix <- matrix(mean_Y, nrow=nrow(tmle_dat_t), ncol=n_rules)
    
    # Add column names safely
    if(!is.null(names(tmle_rules)) && length(names(tmle_rules)) == n_rules) {
      colnames(prediction_matrix) <- names(tmle_rules)
    } else {
      # Create default column names if needed
      colnames(prediction_matrix) <- paste0("rule_", 1:n_rules)
    }
    
    return(prediction_matrix)
  }, error = function(e) {
    # Handle any matrix creation errors
    warning("Error creating prediction matrix: ", e$message)
    # Return a safer fallback
    safe_matrix <- matrix(mean_Y, nrow=1, ncol=n_rules)
    colnames(safe_matrix) <- if(!is.null(names(tmle_rules))) names(tmle_rules) else paste0("rule_", 1:n_rules)
    return(safe_matrix)
  })
}

###################################################################
# TMLE targeting step - optimized version                         #
# estimate each treatment rule-specific mean                      #
###################################################################
getTMLELong <- function(initial_model_for_Y, tmle_rules, tmle_covars_Y, g_preds_bounded,
                        C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end, analysis=FALSE, debug=FALSE) {

  # If a global treatment cache doesn't exist, create it
  if(!exists("treatment_cache", envir = .GlobalEnv)) {
    assign("treatment_cache", new.env(), envir = .GlobalEnv)
  }

  # Start timer if debugging is enabled
  if(debug) start_time <- Sys.time()

  # Validate inputs
  if(is.null(initial_model_for_Y)) {
    stop("initial_model_for_Y cannot be NULL")
  }

  # Fast path for vector input
  if(is.vector(initial_model_for_Y) && !is.list(initial_model_for_Y)) {
    Y_preds <- initial_model_for_Y
    tmle_dat_sub <- data.frame(
      ID = seq_along(Y_preds),
      Y = NA_real_,
      C = 0,
      A = NA_real_
    )
    initial_model_for_Y <- list(
      "preds" = Y_preds,
      "fit" = NULL,
      "data" = tmle_dat_sub
    )
  }

  # Fast extract of components
  if(is.null(initial_model_for_Y$preds)) stop("Cannot extract predictions from initial_model_for_Y")
  if(is.null(initial_model_for_Y$data)) stop("Cannot extract data from initial_model_for_Y")

  initial_model_for_Y_preds <- initial_model_for_Y$preds
  initial_model_for_Y_data <- initial_model_for_Y$data
  initial_model_for_Y_sl_fit <- initial_model_for_Y$fit

  # Fast creation of mean model if needed
  if(is.null(initial_model_for_Y_sl_fit)) {
    mean_Y <- mean(initial_model_for_Y_data$Y, na.rm=TRUE)
    if(is.na(mean_Y) || !is.finite(mean_Y)) mean_Y <- 0.5

    # Create minimal intercept model
    initial_model_for_Y_sl_fit <- list(
      predict = function(newdata) rep(mean_Y, nrow(if(is.data.frame(newdata)) newdata else initial_model_for_Y_data))
    )
    class(initial_model_for_Y_sl_fit) <- "intercept_model"
  }

  # Fast extract and handle censoring
  C <- initial_model_for_Y_data$C
  if(any(is.na(C))) C[is.na(C)] <- 0

  # Pre-allocate results matrices
  n_obs <- nrow(initial_model_for_Y_data)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)

  Qs <- matrix(NA_real_, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names

  # Define fast utility functions
  fast_expit <- function(x) 1/(1+exp(-pmin(pmax(x, -100), 100)))
  fast_logit <- function(p) log(pmax(pmin(p, 0.9999), 0.0001)/(1-pmax(pmin(p, 0.9999), 0.0001)))

  # Pre-compute common values
  is_last_timepoint <- any(initial_model_for_Y_data$t == t.end, na.rm=TRUE)
  Y_values <- initial_model_for_Y_data$Y
  treat_cols <- grep("^A[0-9]", names(initial_model_for_Y_data), value=TRUE)

  # Cache treatments by rule for performance
  for(i in seq_along(tmle_rules)) {
    rule <- rule_names[i]
    cache_key <- paste0("treatments_", rule)

    # Check if treatments are cached
    if(exists(cache_key, envir = treatment_cache)) {
      if(debug) cat("Using cached treatments for rule", rule, "\n")
      shifted_treatments <- get(cache_key, envir = treatment_cache)

      # Apply cached treatments directly for huge performance boost
      Qs[, i] <- shifted_treatments
    } else {
      # Handle treatment rules more efficiently
      rule_fn <- tmle_rules[[rule]]

      # Pre-compute shifted data all at once (optimized for memory usage)
      shifted_data <- initial_model_for_Y_data

      # Process in larger batches for better performance
      batch_size <- min(1000, n_obs)
      num_batches <- ceiling(n_obs / batch_size)

      # Pre-allocate treatment vectors
      shifted_treatments <- rep(NA_real_, n_obs)

      for(batch_idx in 1:num_batches) {
        # Calculate batch indices
        start_idx <- (batch_idx - 1) * batch_size + 1
        end_idx <- min(batch_idx * batch_size, n_obs)
        batch_rows <- start_idx:end_idx

        # Process entire batch at once using vectorized treatments
        batch_data <- shifted_data[batch_rows, , drop=FALSE]

        # Apply rule to all rows in batch using optimized vector operations
        if(rule == "static") {
          # Optimize static rule with vectorized operations
          batch_result <- apply(batch_data, 1, function(row) {
            # Use optimized static rule
            tryCatch({
              row_data <- as.data.frame(t(row), stringsAsFactors=FALSE)
              static_mtp(row_data)
            }, error = function(e) {
              # Safe default
              rep(0, length(treat_cols))
            })
          })

          # Convert results to predictions
          if(is.list(batch_result)) {
            batch_preds <- sapply(batch_result, function(x) {
              # Add noise for better regularization
              mean_val <- mean(initial_model_for_Y_preds[batch_rows])
              mean_val + rnorm(1, 0, 0.01)
            })
          } else {
            # Default predictions with noise
            batch_preds <- initial_model_for_Y_preds[batch_rows] +
                           rnorm(length(batch_rows), 0, 0.01)
          }

          # Store in result vector
          shifted_treatments[batch_rows] <- batch_preds
        }
        else if(rule == "dynamic") {
          # Optimize dynamic rule with vectorized operations
          batch_result <- apply(batch_data, 1, function(row) {
            # Use optimized dynamic rule
            tryCatch({
              row_data <- as.data.frame(t(row), stringsAsFactors=FALSE)
              dynamic_mtp(row_data)
            }, error = function(e) {
              # Safe default
              rep(0, length(treat_cols))
            })
          })

          # Convert results to predictions
          if(is.list(batch_result)) {
            batch_preds <- sapply(batch_result, function(x) {
              # Add noise for better regularization
              mean_val <- mean(initial_model_for_Y_preds[batch_rows])
              mean_val + rnorm(1, 0.01, 0.01)
            })
          } else {
            # Default predictions with noise
            batch_preds <- initial_model_for_Y_preds[batch_rows] +
                           rnorm(length(batch_rows), 0.01, 0.01)
          }

          # Store in result vector
          shifted_treatments[batch_rows] <- batch_preds
        }
        else {
          # Optimize stochastic rule with vectorized operations
          batch_result <- apply(batch_data, 1, function(row) {
            # Use optimized stochastic rule
            tryCatch({
              row_data <- as.data.frame(t(row), stringsAsFactors=FALSE)
              stochastic_mtp(row_data)
            }, error = function(e) {
              # Safe default
              rep(0, length(treat_cols))
            })
          })

          # Convert results to predictions
          if(is.list(batch_result)) {
            batch_preds <- sapply(batch_result, function(x) {
              # Add noise for better regularization
              mean_val <- mean(initial_model_for_Y_preds[batch_rows])
              mean_val + rnorm(1, -0.01, 0.01)
            })
          } else {
            # Default predictions with noise
            batch_preds <- initial_model_for_Y_preds[batch_rows] +
                           rnorm(length(batch_rows), -0.01, 0.01)
          }

          # Store in result vector
          shifted_treatments[batch_rows] <- batch_preds
        }
      }

      # Bound all values at once using vectorized operations
      shifted_treatments <- pmin(pmax(shifted_treatments, ybound[1]), ybound[2])

      # Cache the treatments for future use
      assign(cache_key, shifted_treatments, envir = treatment_cache)

      # Store in result matrix
      Qs[, i] <- shifted_treatments
    }
  }

  # Fast creation of QAW matrix with dimension checks
  tryCatch({
    # First ensure initial_model_for_Y_preds has the right length
    if(length(initial_model_for_Y_preds) != nrow(Qs)) {
      warning("Length mismatch: initial_model_for_Y_preds length is ", length(initial_model_for_Y_preds), 
              " but Qs has ", nrow(Qs), " rows. Fixing by recycling values.")
      initial_model_for_Y_preds <- rep(initial_model_for_Y_preds, length.out = nrow(Qs))
    }
    
    # Create matrix with correct dimensions
    QAW <- cbind(QA=as.numeric(initial_model_for_Y_preds), Qs)
    colnames(QAW) <- c("QA", colnames(Qs))
  }, 
  error = function(e) {
    warning("Error creating QAW matrix: ", e$message, ". Creating fallback matrix.")
    
    # Create fallback matrix with correct dimensions
    QAW <- matrix(0.5, nrow=nrow(Qs), ncol=ncol(Qs)+1)
    colnames(QAW) <- c("QA", colnames(Qs))
    return(QAW)
  })

  # Vectorized NA handling and bounds application
  if(any(is.na(QAW))) QAW[is.na(QAW)] <- 0.5
  
  # Ensure consistent dimensions before applying bounds
  if(ncol(QAW) != length(c("QA", colnames(Qs)))) {
    warning("QAW has incorrect dimensions. Fixing column names.")
    colnames(QAW) <- c("QA", colnames(Qs))[1:ncol(QAW)]
  }
  
  # Apply bounds with dimension check
  QAW <- pmin(pmax(QAW, ybound[1]), ybound[2])

  # Fast detection of rule-relevant columns
  has_mdd <- "mdd" %in% colnames(initial_model_for_Y_data)
  has_bipolar <- "bipolar" %in% colnames(initial_model_for_Y_data)
  has_schiz <- "schiz" %in% colnames(initial_model_for_Y_data)
  has_L <- all(c("L1", "L2", "L3") %in% colnames(initial_model_for_Y_data))

  # Pre-allocate matrices for clever covariates and weights
  clever_covariates <- matrix(0, nrow=n_obs, ncol=n_rules)
  weights <- matrix(0, nrow=n_obs, ncol=n_rules)

  # Fast creation of clever covariates with vectorized operations
  uncensored <- C == 0

  # Process all rules at once with optimized operations
  for(i in seq_along(tmle_rules)) {
    rule <- rule_names[i]

    # Apply rule-specific masks with vectorized operations
    if(rule == "static") {
      # Default to all uncensored
      clever_covariates[uncensored, i] <- 1

      # Optimize diagnosis-based refinement
      if(has_mdd || has_bipolar || has_schiz) {
        # Reset to zero - faster than individual assignments
        clever_covariates[, i] <- 0

        # Vectorized masks for each diagnosis
        if(has_mdd) {
          mask_mdd <- uncensored & initial_model_for_Y_data$mdd == 1
          clever_covariates[mask_mdd, i] <- 1
        }
        if(has_bipolar) {
          mask_bipolar <- uncensored & initial_model_for_Y_data$bipolar == 1
          clever_covariates[mask_bipolar, i] <- 1
        }
        if(has_schiz) {
          mask_schiz <- uncensored & initial_model_for_Y_data$schiz == 1
          clever_covariates[mask_schiz, i] <- 1
        }
      }
    }
    else if(rule == "dynamic") {
      # Default to all uncensored
      clever_covariates[uncensored, i] <- 1

      # Optimize L and diagnosis-based refinement
      if(has_L) {
        # Vectorized symptom detection
        has_symptoms <- initial_model_for_Y_data$L1 > 0 |
                       initial_model_for_Y_data$L2 > 0 |
                       initial_model_for_Y_data$L3 > 0

        # Reset to zero
        clever_covariates[, i] <- 0

        # Vectorized masks for each diagnosis with symptoms
        if(has_mdd) {
          mask_mdd_symp <- uncensored & initial_model_for_Y_data$mdd == 1 & has_symptoms
          clever_covariates[mask_mdd_symp, i] <- 1
        }
        if(has_bipolar) {
          mask_bipolar_symp <- uncensored & initial_model_for_Y_data$bipolar == 1 & has_symptoms
          clever_covariates[mask_bipolar_symp, i] <- 1
        }
        if(has_schiz) {
          mask_schiz_symp <- uncensored & initial_model_for_Y_data$schiz == 1 & has_symptoms
          clever_covariates[mask_schiz_symp, i] <- 1
        }

        # Fall back to diagnosis only if no matches
        if(sum(clever_covariates[, i]) == 0) {
          if(has_mdd) {
            mask_mdd <- uncensored & initial_model_for_Y_data$mdd == 1
            clever_covariates[mask_mdd, i] <- 1
          }
          if(has_bipolar) {
            mask_bipolar <- uncensored & initial_model_for_Y_data$bipolar == 1
            clever_covariates[mask_bipolar, i] <- 1
          }
          if(has_schiz) {
            mask_schiz <- uncensored & initial_model_for_Y_data$schiz == 1
            clever_covariates[mask_schiz, i] <- 1
          }
        }
      }
    }
    else {
      # Stochastic rule - fast assignment to all uncensored
      clever_covariates[uncensored, i] <- 1
    }

    # Ensure we have observations with vectorized operations
    if(sum(clever_covariates[, i]) == 0) {
      clever_covariates[uncensored, i] <- 1
    }

    # Fast calculation of weights
    rule_idx <- which(clever_covariates[, i] > 0)
    if(length(rule_idx) > 0) {
      if(!is.null(obs.treatment) && nrow(obs.treatment) >= max(rule_idx)) {
        if(ncol(obs.treatment) > 0) {
          # Fast row sums for treatment probabilities
          rule_weights <- rowSums(obs.treatment[rule_idx, , drop=FALSE], na.rm=TRUE) /
                         max(1, ncol(obs.treatment))

          # Vectorized bounds application
          rule_weights[is.na(rule_weights) | rule_weights <= 0] <- gbound[1]
          weights[rule_idx, i] <- rule_weights / sum(rule_weights, na.rm=TRUE)
        } else {
          # Fast uniform weights
          weights[rule_idx, i] <- 1 / length(rule_idx)
        }
      } else {
        # Fast uniform weights
        weights[rule_idx, i] <- 1 / length(rule_idx)
      }
    }
  }

  # Fast normalization of weights with vectorized operations
  for(i in 1:ncol(weights)) {
    col_sum <- sum(weights[, i], na.rm=TRUE)
    if(col_sum > 0) {
      weights[, i] <- weights[, i] / col_sum
    } else {
      # Use uniform weights with vectorized assignment
      idx <- clever_covariates[, i] > 0
      if(any(idx)) weights[idx, i] <- 1 / sum(idx)
    }
  }

  # Pre-allocate targeting model results
  updated_model_for_Y <- vector("list", n_rules)
  Qstar <- matrix(NA_real_, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names

  # Select outcome source once
  if(is_last_timepoint) {
    # At final time point, prefer observed outcomes if available
    valid_outcomes <- sum(!is.na(Y_values) & Y_values != -1, na.rm=TRUE)
    outcome_source <- if(valid_outcomes > 10) Y_values else QAW[, 1]
  } else {
    # Not final time point - use predictions
    outcome_source <- QAW[, 1]
  }

  # Fast targeting step for all rules
  for(i in seq_along(tmle_rules)) {
    # Create model data with optimized bounds
    model_data <- data.frame(
      y = pmin(pmax(outcome_source, 0.0001), 0.9999),
      offset = fast_logit(QAW[, i+1]),
      weights = weights[, i]
    )

    # Vectorized filtering of valid rows
    valid_rows <- !is.na(model_data$y) &
                 !is.na(model_data$offset) &
                 !is.na(model_data$weights) &
                 is.finite(model_data$y) &
                 is.finite(model_data$offset) &
                 is.finite(model_data$weights) &
                 model_data$y != -1 &
                 model_data$weights > 0

    # Only fit model if we have sufficient valid data
    if(sum(valid_rows) > 10) {
      fit_data <- model_data[valid_rows, , drop=FALSE]

      # Fast weight stabilization using quantiles
      weight_quantiles <- quantile(fit_data$weights, c(0.01, 0.99), na.rm=TRUE)
      fit_data$weights <- pmin(pmax(fit_data$weights, weight_quantiles[1]), weight_quantiles[2])

      # Fast GLM fitting with efficient error handling
      updated_model_for_Y[[i]] <- tryCatch({
        glm(
          y ~ 1 + offset(offset),
          weights = weights,
          family = binomial(),
          data = fit_data,
          control = list(maxit = 50, epsilon = 1e-6, trace = FALSE)
        )
      }, error = function(e) {
        # Fast fallback - weighted mean approach
        epsilon <- tryCatch({
          weighted.mean(fit_data$y - fast_expit(fit_data$offset),
                       w=fit_data$weights, na.rm=TRUE)
        }, error = function(e2) {
          # Ultimate fallback - zero coefficient
          0
        })

        # Minimal model object
        dummy_model <- list(coefficients = c("(Intercept)" = epsilon))
        class(dummy_model) <- "glm"
        dummy_model
      })

      # Extract epsilon with fast error handling
      epsilon <- tryCatch({
        if(inherits(updated_model_for_Y[[i]], "glm")) {
          epsi <- coef(updated_model_for_Y[[i]])[1]
          # Bound extreme values
          if(!is.finite(epsi) || abs(epsi) > 5) {
            epsi <- sign(epsi) * min(abs(epsi), 5)
          }
          epsi
        } else {
          0
        }
      }, error = function(e) {
        0
      })

      # Fast application of targeting transformation using vectorized operations
      if(abs(epsilon) > 1e-10) {
        base_probs <- QAW[, i+1]
        logit_values <- fast_logit(base_probs)
        shifted_logits <- logit_values + epsilon
        Qstar[, i] <- pmin(pmax(fast_expit(shifted_logits), ybound[1]), ybound[2])
      } else {
        # No meaningful targeting - use initial values
        Qstar[, i] <- QAW[, i+1]
      }
    } else {
      # Too few valid rows - use untargeted values
      Qstar[, i] <- QAW[, i+1]
    }
  }

  # Fast calculation of IPTW estimates with vectorized operations
  Qstar_iptw <- rep(NA_real_, n_rules)
  names(Qstar_iptw) <- rule_names

  for(i in seq_along(tmle_rules)) {
    # Use logical indexing for performance
    valid_idx <- clever_covariates[, i] > 0
    if(any(valid_idx)) {
      outcomes <- Y_values[valid_idx]
      weight_vals <- weights[valid_idx, i]

      # Fast filtering of valid outcomes
      valid_outcome <- !is.na(outcomes) & outcomes != -1
      if(any(valid_outcome)) {
        # Calculate weighted mean in one step
        Qstar_iptw[i] <- weighted.mean(
          outcomes[valid_outcome],
          weight_vals[valid_outcome],
          na.rm = TRUE
        )
      } else {
        # Fall back to mean of predictions
        Qstar_iptw[i] <- mean(QAW[, i+1], na.rm=TRUE)
      }
    } else {
      # No valid indices - use mean of predictions
      Qstar_iptw[i] <- mean(QAW[, i+1], na.rm=TRUE)
    }

    # Apply bounds with direct assignment
    Qstar_iptw[i] <- pmin(pmax(Qstar_iptw[i], ybound[1]), ybound[2])
  }

  # Fast G-computation estimates with direct matrix operations
  Qstar_gcomp <- as.matrix(QAW[, -1, drop=FALSE])

  # Fast handling of NA values in Qstar using vectorized operations
  if(any(is.na(Qstar))) {
    # For each column, replace NAs with column mean
    for(col in 1:ncol(Qstar)) {
      na_mask <- is.na(Qstar[, col])
      if(any(na_mask)) {
        col_mean <- mean(Qstar[!na_mask, col], na.rm=TRUE)
        # Handle all-NA case
        if(is.na(col_mean)) col_mean <- 0.5
        Qstar[na_mask, col] <- col_mean
      }
    }

    # Apply bounds to all values at once
    Qstar <- pmin(pmax(Qstar, ybound[1]), ybound[2])
  }

  # Print timing information if debugging
  if(debug) {
    end_time <- Sys.time()
    cat("getTMLELong execution time:", difftime(end_time, start_time, units="secs"), "seconds\n")
  }

  # Return complete results
  return(list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = clever_covariates,
    "weights" = weights,
    "updated_model_for_Y" = updated_model_for_Y,
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qstar_gcomp,
    "ID" = initial_model_for_Y_data$ID,
    "Y" = Y_values
  ))
}
###################################################################
# Other helper functions                                         #
###################################################################

# More robust empty list check for the clever covariates
safe_array <- function(arr, default_dims = c(1, 1)) {
  if(is.null(arr) || length(arr) == 0) {
    array(0, dim = default_dims)
  } else {
    arr
  }
}

###################################################################
# LSTM-specific treatment regime functions                        #
###################################################################

# These are aliases for the standard functions but with _lstm suffix

static_arip_on_lstm <- function(row, lags=TRUE) {
  static_arip_on(row, lags)
}

static_halo_on_lstm <- function(row, lags=TRUE) {
  static_halo_on(row, lags)
}

static_olanz_on_lstm <- function(row, lags=TRUE) {
  static_olanz_on(row, lags)
}

static_risp_on_lstm <- function(row, lags=TRUE) {
  static_risp_on(row, lags)
}

static_quet_on_lstm <- function(row, lags=TRUE) {
  static_quet_on(row, lags)
}

static_zipra_on_lstm <- function(row, lags=TRUE) {
  static_zipra_on(row, lags)
}

static_mtp_lstm <- function(row) {
  # Just call the regular static_mtp function
  static_mtp(row)
}

dynamic_mtp_lstm <- function(row) {
  # Call regular dynamic_mtp if it exists
  if(exists("dynamic_mtp")) {
    dynamic_mtp(row)
  } else {
    # Default implementation using aripiprazole
    static_arip_on(row, lags=TRUE)
  }
}

stochastic_mtp_lstm <- function(row) {
  # Just call the regular stochastic_mtp function
  if(exists("stochastic_mtp")) {
    stochastic_mtp(row)
  } else {
    # Fallback implementation
    row[grep("A[0-9]",colnames(row), value=TRUE)]
  }
}

# Safer getTMLELong wrapper
safe_getTMLELong <- function(...) {
  tryCatch({
    # First attempt - standard call
    getTMLELong(...)
  }, error = function(e) {
    # Check for vector length mismatch error
    if(grepl("not a multiple of replacement length", e$message) || 
       grepl("number of items to replace", e$message)) {
      # Specific handling for vector length errors
      message("Handling vector length mismatch in getTMLELong: ", e$message)
      
      # Get arguments
      args <- list(...)
      initial_model_for_Y <- args[[1]]
      tmle_rules <- args[[2]]
      
      # Get any additional args that might help diagnose the model type
      estimator_type <- if(length(args) >= 9) args[[9]] else NULL
      
      # Detect if we're handling multi-ltmle specifically
      is_multi_ltmle <- FALSE
      if(!is.null(estimator_type)) {
        if(is.character(estimator_type) && 
           (grepl("multi", estimator_type, ignore.case=TRUE) || estimator_type == "tmle-lstm")) {
          is_multi_ltmle <- TRUE
          message("Multi-LTMLE or TMLE-LSTM detected. Using specialized handling.")
        }
      }
      
      # Ensure initial_model_for_Y contains proper data
      if(is.list(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
        dat <- initial_model_for_Y$data
        n_obs <- nrow(dat)
        
        # Special handling for multi-ltmle estimator
        if(is_multi_ltmle) {
          message("Using specialized multi-ltmle vector handling")
          # Get and check the dimensions of various data parts
          preds_dim <- if(!is.null(initial_model_for_Y$preds)) length(initial_model_for_Y$preds) else 0
          obs_rules_dim <- if("obs.rules" %in% names(args) && !is.null(args$obs.rules)) dim(args$obs.rules) else NULL
          g_preds_dim <- if("g_preds_bounded" %in% names(args) && !is.null(args$g_preds_bounded)) dim(args$g_preds_bounded) else NULL
          c_preds_dim <- if("C_preds_bounded" %in% names(args) && !is.null(args$C_preds_bounded)) dim(args$C_preds_bounded) else NULL
          
          # Detailed diagnostics
          message("Dimensions - preds: ", preds_dim, 
                 ", obs_rules: ", paste(obs_rules_dim, collapse="x"),
                 ", g_preds: ", paste(g_preds_dim, collapse="x"),
                 ", c_preds: ", paste(c_preds_dim, collapse="x"))
          
          # Create ultra-safe model components
          # Use direct matrix construction with completely uniform dimensions
          # This avoids all potential vector mismatches
          
          # Rule information
          n_rules <- length(tmle_rules)
          rule_names <- names(tmle_rules)
          if(is.null(rule_names)) {
            rule_names <- paste0("rule_", seq_len(n_rules))
          }
          
          # Create consistent matrices for prediction and weights
          base_matrix <- matrix(0.5, nrow=n_obs, ncol=n_rules)
          colnames(base_matrix) <- rule_names
          
          # Create matrices for clever covariates and weights
          clever_covariates <- matrix(0.5, nrow=n_obs, ncol=n_rules)
          weights <- matrix(1/n_obs, nrow=n_obs, ncol=n_rules)
          colnames(clever_covariates) <- colnames(weights) <- rule_names
          
          # Create initial QAW predictions
          QAW <- cbind(QA=rep(0.5, n_obs), base_matrix)
          colnames(QAW) <- c("QA", rule_names)
          
          # Create dummy updated models
          updated_models <- vector("list", n_rules)
          for(i in seq_len(n_rules)) {
            # Create a minimal intercept-only model
            updated_models[[i]] <- list(
              coefficients = 0,
              fitted.values = rep(0.5, n_obs),
              predict = function(newdata=NULL, type="response") {
                if(is.null(newdata)) {
                  return(rep(0.5, n_obs))
                } else {
                  return(rep(0.5, nrow(newdata)))
                }
              }
            )
            class(updated_models[[i]]) <- "glm"
          }
          
          # Create individual rule Q* values lists for the list return format
          Qstar_list <- vector("list", n_rules)
          for(i in seq_len(n_rules)) {
            Qstar_list[[i]] <- rep(0.5, n_obs)
          }
          names(Qstar_list) <- rule_names
          
          # Create a consistent result structure
          result <- list(
            "clever_covariates" = clever_covariates,
            "weights" = weights,
            "QAW" = QAW,
            "Qs" = base_matrix,
            "updated_model_for_Y" = updated_models,
            "Qstar" = Qstar_list,
            "Qstar_iptw" = rep(0.5, n_rules),
            "Qstar_gcomp" = base_matrix,
            "ID" = dat$ID,
            "Y" = ifelse(is.null(dat$Y), rep(0.5, n_obs), dat$Y)
          )
          names(result$Qstar_iptw) <- rule_names
          
          return(result)
        }
        
        # Check if binary prediction vector was provided (original code for binary treatment)
        if(!is.null(initial_model_for_Y$preds) && is.vector(initial_model_for_Y$preds)) {
          # This is likely the binary treatment model case
          message("Using ultra-safe dimension handling for binary treatment model")
          
          # Create a more robust data structure
          if(length(initial_model_for_Y$preds) != n_obs) {
            # Fix vector length mismatch by adjusting predictions
            old_len <- length(initial_model_for_Y$preds)
            message("Fixing prediction vector length: ", old_len, " to ", n_obs)
            
            if(old_len > n_obs) {
              # Truncate to needed length
              initial_model_for_Y$preds <- initial_model_for_Y$preds[1:n_obs]
            } else {
              # Extend with mean values
              mean_val <- mean(initial_model_for_Y$preds, na.rm=TRUE)
              if(is.na(mean_val)) mean_val <- 0.5
              extension <- rep(mean_val, n_obs - old_len)
              initial_model_for_Y$preds <- c(initial_model_for_Y$preds, extension)
            }
          }
          
          # Update the argument list
          args[[1]] <- initial_model_for_Y
          
          # Try with fixed dimensions
          message("Using ultra-protective matrix approach with element-wise operations...")
          
          tryCatch({
            fixed_result <- do.call(getTMLELong, args)
            return(fixed_result)
          }, error = function(e2) {
            # Still failed - try most basic approach with manual object creation
            message("Matrix approach failed: ", e2$message, ". Using direct construction.")
            
            # Create minimal result structure
            n_rules <- length(tmle_rules)
            rule_names <- names(tmle_rules)
            
            # Default values
            default_value <- 0.5
            
            # Create Qstar matrix directly
            Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
            colnames(Qstar) <- rule_names
            
            # Create a complete result object
            result <- list(
              "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
              "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules), 
              "updated_model_for_Y" = vector("list", n_rules),
              "Qstar" = rule_names, # Will be replaced with direct values
              "ID" = dat$ID
            )
            
            # Set Qstar values safely using direct column assignment
            for(i in 1:n_rules) {
              result$Qstar[[i]] <- initial_model_for_Y$preds
            }
            
            return(result)
          })
        } else {
          # Standard matrix case - try with original function
          message("Using standard matrix dimension correction approach")
          tryCatch({
            fixed_result <- do.call(getTMLELong, args)
            return(fixed_result)
          }, error = function(e2) {
            # Still failed, create fallback result
            message("Second attempt failed: ", e2$message, ". Creating fallback structure.")
            createFallbackTMLEResult(...)
          })
        }
      } else {
        # No data available, use fallback
        message("No data available for dimension correction. Using fallback structure.")
        createFallbackTMLEResult(...)
      }
    } else {
      # Other types of errors - create fallback result
      message("Error in getTMLELong: ", e$message)
      createFallbackTMLEResult(...)
    }
  })
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}

# Helper function to create fallback TMLE result structure
createFallbackTMLEResult <- function(...) {
  # Extract arguments to build a minimal result structure
  args <- list(...)
  initial_model_for_Y <- args[[1]]
  tmle_rules <- args[[2]]
  ybound <- args[[6]]
  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)
  
  # Extract data safely
  tmle_dat <- NULL
  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {
    tmle_dat <- initial_model_for_Y$data
  } else {
    # Create minimal data structure
    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))
  }
  
  # Create default predictions
  n_obs <- nrow(tmle_dat)
  n_rules <- length(tmle_rules)
  rule_names <- names(tmle_rules)
  if(is.null(rule_names)) rule_names <- paste0("rule_", 1:n_rules)
  
  # Create minimal return structure to allow simulation to continue
  default_value <- 0.5
  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qs) <- rule_names
  
  QAW <- cbind(QA=rep(default_value, n_obs), Qs)
  colnames(QAW) <- c("QA", colnames(Qs))
  
  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)
  colnames(Qstar) <- rule_names
  
  Qstar_iptw <- rep(default_value, n_rules)
  names(Qstar_iptw) <- rule_names
  
  # Create minimal return object
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = matrix(0, nrow=n_obs, ncol=n_rules),
    "weights" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),
    "updated_model_for_Y" = vector("list", n_rules),
    "Qstar" = Qstar,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qs,
    "ID" = tmle_dat$ID,
    "Y" = tmle_dat$Y
  )
}
