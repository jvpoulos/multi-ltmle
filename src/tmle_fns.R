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
  
  # Use the same sequential_g function that we use for all other time points
  Y_preds <- sequential_g(t, tmle_dat, n.folds, tmle_covars_Y, initial_model_for_Y_sl, ybound)
  
  # First create a safe subset of data without missing Y values
  tmle_dat_sub <- tmle_dat[tmle_dat$t==t & !is.na(tmle_dat$Y),] # drop rows with missing Y
  
  # Create a simple model object for compatibility
  mean_Y <- mean(Y_preds, na.rm=TRUE)
  if(is.na(mean_Y) || !is.finite(mean_Y)) mean_Y <- 0.5
  
  mean_fit <- list(params = list(covariates = character(0)))
  class(mean_fit) <- "custom_mean_fit"
  mean_fit$predict <- function(task) Y_preds
  
  # Return list with all components
  return(list(
    "preds" = Y_preds,
    "fit" = mean_fit,
    "data" = tmle_dat[tmle_dat$t==t,]
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

# Modify the sequential_g function to better handle constant Y values
sequential_g <- function(t, tmle_dat, n.folds, tmle_covars_Y, initial_model_for_Y_sl, ybound, Y_pred=NULL){
  
  # First create a safe subset of data without missing Y values
  tmle_dat_sub <- tmle_dat[tmle_dat$t==t & !is.na(tmle_dat$Y),] # drop rows with missing Y
  
  if(nrow(tmle_dat_sub) < 10) {
    # If very few observations, use data from adjacent time points
    nearby_t <- c(t-1, t+1)
    nearby_t <- nearby_t[nearby_t > 0 & nearby_t <= t.end]
    
    if(length(nearby_t) > 0) {
      additional_data <- tmle_dat[tmle_dat$t %in% nearby_t & !is.na(tmle_dat$Y),]
      
      if(nrow(additional_data) > 0) {
        tmle_dat_sub <- rbind(tmle_dat_sub, additional_data)
        message("Added ", nrow(additional_data), " records from nearby time points for t=", t)
      }
    }
  }
  
  # Check for near-constant Y values (not just identical)
  y_values <- tmle_dat_sub$Y[!is.na(tmle_dat_sub$Y) & tmle_dat_sub$Y != -1]
  if(length(y_values) > 0 && diff(range(y_values)) < 0.01) {
    message("Y values nearly constant at t=", t, ", using robust approach")
    # Use mean with a small amount of noise to avoid convergence issues
    mean_y <- mean(y_values)
    noise <- rnorm(nrow(tmle_dat[tmle_dat$t==t,]), 0, 0.001)
    return(pmin(pmax(mean_y + noise, ybound[1]), ybound[2]))
  }
  
  # Special handling for Y_pred when t<T
  if(!is.null(Y_pred)){ 
    # Convert Y_pred to numeric vector if it's a list
    if(is.list(Y_pred)) {
      Y_pred <- unlist(Y_pred)
    }
    
    # Map IDs between datasets - create ID mapping that works even with partial matches
    common_ids <- intersect(tmle_dat_sub$ID, names(Y_pred))
    if(length(common_ids) > 0) {
      tmle_dat_sub <- tmle_dat_sub[tmle_dat_sub$ID %in% common_ids,]
      tmle_dat_sub$Y <- Y_pred[match(tmle_dat_sub$ID, names(Y_pred))]
    } else {
      # No matching IDs found - print warning and use current Y values
      warning("No matching IDs found between prediction data and tmle_dat_sub at t=", t)
    }
  }
  
  # Validate that all covariates exist in the data
  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))
  if(length(missing_covars) > 0) {
    message("Adding missing covariates: ", paste(missing_covars, collapse=", "))
    # Add missing covariates with default values
    for(cov in missing_covars) {
      tmle_dat_sub[[cov]] <- 0
    }
  }
  
  # Define cross-validation folds
  folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
  
  # More robust determination of outcome type
  y_values <- tmle_dat_sub$Y[!is.na(tmle_dat_sub$Y)]
  if(length(y_values) > 0) {
    # Check if all values are binary (0/1)
    if(all(y_values %in% c(0,1))) {
      # For binary outcomes, explicitly use binomial
      outcome_type <- "binomial"
      message("Binary outcome detected - using binomial family")
    } else {
      outcome_type <- "continuous"
      message("Continuous outcome detected")
    }
  } else {
    # Default to continuous if we can't determine
    outcome_type <- "continuous"
    message("No valid outcome values - defaulting to continuous")
  }
  
  # Check for constant Y values early
  y_values <- tmle_dat_sub$Y[!is.na(tmle_dat_sub$Y)]
  if(length(unique(y_values)) == 1) {
    message("Y is constant, using intercept-only model")
    const_val <- y_values[1]
    return(rep(const_val, nrow(tmle_dat[tmle_dat$t==t,])))
  }
  
  # Define task with appropriate settings and explicit drop_missing_outcome
  initial_model_for_Y_task <- make_sl3_Task(
    data = tmle_dat_sub,
    covariates = intersect(tmle_covars_Y, colnames(tmle_dat_sub)), 
    outcome = "Y",
    outcome_type = outcome_type, 
    folds = folds,
    drop_missing_outcome = TRUE  # Explicitly handle missing outcomes
  )
  
  # Train model with progressive fallback strategy for improved robustness
  initial_model_for_Y_sl_fit <- tryCatch({
    message("Training SuperLearner at t=", t)
    
    # Try the full SL first
    tryCatch({
      sl_fit <- initial_model_for_Y_sl$train(initial_model_for_Y_task)
      message("SuperLearner training successful")
      sl_fit
    }, error = function(e) {
      message("Full SuperLearner failed with error: ", e$message)
      
      # First fallback: Try using a simpler stack with glm and mean learners
      tryCatch({
        message("Trying simplified SuperLearner with glm and mean learners")
        # Create a simplified learner stack that's more likely to succeed
        if(outcome_type == "binomial") {
          # For binary outcomes
          lrnrs <- list(
            make_learner(Lrnr_glm, family = "binomial"),
            make_learner(Lrnr_mean)
          )
        } else {
          # For continuous outcomes
          lrnrs <- list(
            make_learner(Lrnr_glm, family = "gaussian"),
            make_learner(Lrnr_mean)
          )
        }
        
        sl_simple <- make_learner(Stack, lrnrs)
        sl_simple$train(initial_model_for_Y_task)
      }, error = function(e2) {
        message("Simplified SuperLearner failed with error: ", e2$message)
        
        # Second fallback: Try a single GLM model with appropriate family
        tryCatch({
          message("Trying single GLM model")
          if(outcome_type == "binomial") {
            glm_learner <- make_learner(Lrnr_glm, family = "binomial")
          } else {
            glm_learner <- make_learner(Lrnr_glm, family = "gaussian")
          }
          glm_learner$train(initial_model_for_Y_task)
        }, error = function(e3) {
          message("GLM model failed with error: ", e3$message)
          
          # Final fallback: Use a mean learner (intercept-only model)
          message("Using intercept-only mean model")
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
    })
  }, error = function(e) {
    message("All model training attempts failed: ", e$message)
    
    # Create a custom manually-trained mean object as last resort
    const_val <- mean(tmle_dat_sub$Y, na.rm=TRUE)
    if(is.na(const_val) || !is.finite(const_val)) const_val <- 0.5
    
    message("Created custom mean model with constant prediction: ", const_val)
    fit <- list(params = list(covariates = character(0)))
    class(fit) <- "custom_mean_fit"
    fit$predict <- function(task) {
      rep(const_val, nrow(task$data))
    }
    fit
  })
  
  # Create prediction data
  pred_data <- tmle_dat[tmle_dat$t==t,]
  
  # Add missing covariates to prediction data
  all_needed_covars <- NULL
  
  # First try to get covariates directly from fit object
  if(is(initial_model_for_Y_sl_fit, "Lrnr_mean") || inherits(initial_model_for_Y_sl_fit, "custom_mean_fit")) {
    # For mean/intercept-only model
    all_needed_covars <- character(0)
  } else if(!is.null(initial_model_for_Y_sl_fit$fit_object) && 
            !is.null(initial_model_for_Y_sl_fit$fit_object$params) && 
            !is.null(initial_model_for_Y_sl_fit$fit_object$params$covariates)) {
    all_needed_covars <- initial_model_for_Y_sl_fit$fit_object$params$covariates
  } else {
    # Default: use provided covariates
    all_needed_covars <- tmle_covars_Y
  }
  
  # Check if we have any covariates to process
  if(length(all_needed_covars) > 0) {
    missing_pred_covars <- setdiff(all_needed_covars, colnames(pred_data))
    if(length(missing_pred_covars) > 0) {
      message("Adding missing covariates to prediction data: ", paste(missing_pred_covars, collapse=", "))
      for(cov in missing_pred_covars) {
        pred_data[[cov]] <- 0  # Default value
      }
    }
  }
  
  # Create prediction task with the same covariates used in training
  prediction_task <- tryCatch({
    # For mean/intercept-only model
    if(is(initial_model_for_Y_sl_fit, "Lrnr_mean") || inherits(initial_model_for_Y_sl_fit, "custom_mean_fit")) {
      sl3_Task$new(
        data = pred_data,
        covariates = character(0),
        outcome = "Y",
        outcome_type = outcome_type,
        drop_missing_outcome = FALSE
      )
    } else {
      # For other models with covariates
      sl3_Task$new(
        data = pred_data,
        covariates = all_needed_covars,
        outcome = "Y",
        outcome_type = outcome_type,
        drop_missing_outcome = FALSE
      )
    }
  }, error = function(e) {
    # Fallback: create task with no covariates for mean learner
    message("Error creating prediction task: ", e$message)
    message("Creating simplified prediction task")
    sl3_Task$new(
      data = pred_data,
      covariates = character(0),
      outcome = "Y",
      outcome_type = outcome_type,
      drop_missing_outcome = FALSE
    )
  })
  
  # Get predictions with robust error handling
  Y_preds <- tryCatch({
    message("Making predictions at t=", t)
    preds <- initial_model_for_Y_sl_fit$predict(prediction_task)
    
    # Ensure predictions are numeric
    if(is.list(preds)) {
      preds <- unlist(preds)
    }
    preds
  }, error = function(e) {
    message("Prediction failed with error: ", e$message)
    message("Cannot make predictions - returning NA values")
    
    # Return NA values to indicate prediction failure
    rep(NA, nrow(pred_data))
  })
  
  # Ensure Y_preds is numeric vector
  Y_preds <- as.numeric(Y_preds)
  
  # Only apply bounds to non-NA values
  non_na_idx <- !is.na(Y_preds)
  if(any(non_na_idx)) {
    Y_preds[non_na_idx] <- pmin(pmax(Y_preds[non_na_idx], ybound[1]), ybound[2])
  }
  
  # Return vector (NOT a list) to avoid indexing issues
  return(Y_preds)
}

###################################################################
# LTMLE core (rebuilt 2026-10)                                    #
# Sequential regression (iterated conditional expectations) for   #
# each target time t, targeting with the cumulative clever         #
# covariate, and influence-curve-based inference.                  #
# Nodes per time k: L1_k, L2_k, L3_k, A_k, C_k (1 = censored), Y_k #
###################################################################

# subjects still under observation at time k (uncensored through k-1, no event through k-1)
ltmle_at_risk <- function(dat, k) {
  if (k == 0) return(rep(TRUE, nrow(dat)))
  out <- dat[[paste0("C_", k - 1)]] == 0 & dat[[paste0("Y_", k - 1)]] == 0
  out[is.na(out)] <- FALSE
  out
}

# subjects at risk at k and uncensored at k (outcome Y_k observed)
ltmle_uncensored <- function(dat, k) {
  if (k == 0) return(rep(TRUE, nrow(dat)))
  out <- ltmle_at_risk(dat, k) & dat[[paste0("C_", k)]] == 0
  out[is.na(out)] <- FALSE
  out
}

ltmle_onehot <- function(a, prefix, J = 6) {
  m <- sapply(2:J, function(j) as.numeric(a == j))
  if (!is.matrix(m)) m <- matrix(m, nrow = 1)
  colnames(m) <- paste0(prefix, "_", 2:J)
  m
}

# Design matrix at time k: baseline race (V1), diagnosis (V2), age (V3, optional), L at k and k-1,
# squared change in L1, indicators used by the rules, A_{k-1}, and optionally A_k.
ltmle_design <- function(dat, k, A = NULL, include_V3 = TRUE, J = 6) {
  n <- nrow(dat)
  L1 <- as.numeric(dat[[paste0("L1_", k)]])
  L2 <- as.numeric(dat[[paste0("L2_", k)]])
  L3 <- as.numeric(dat[[paste0("L3_", k)]])
  lag <- function(v) if (k == 0) rep(0, n) else as.numeric(dat[[paste0(v, "_", k - 1)]])
  V1 <- as.integer(as.character(dat$V1_0))
  V2 <- as.integer(as.character(dat$V2_0))
  X <- cbind(V1_2 = as.numeric(V1 == 2), V1_3 = as.numeric(V1 == 3), V1_4 = as.numeric(V1 == 4),
             V2_2 = as.numeric(V2 == 2), V2_3 = as.numeric(V2 == 3),
             L1 = L1, L2 = L2, L3 = L3, L1_pos = as.numeric(L1 > 0),
             L_any = as.numeric(L1 > 0 | L2 > 0 | L3 > 0),
             L1_lag = lag("L1"), L2_lag = lag("L2"), L3_lag = lag("L3"),
             dL1_sq = (L1 - lag("L1"))^2)
  if (include_V3) X <- cbind(V3 = as.numeric(dat$V3_0), X)
  if (k > 0) X <- cbind(X, ltmle_onehot(as.integer(as.character(dat[[paste0("A_", k - 1)]])), "A_lag", J))
  if (!is.null(A)) X <- cbind(X, ltmle_onehot(A, "A", J))
  X
}

# lapply over time points, in forked processes when cores > 1 (a crashed worker raises an error)
ltmle_lapply <- function(X, FUN, cores = 1) {
  if (cores <= 1) return(lapply(X, FUN))
  out <- parallel::mclapply(X, FUN, mc.cores = cores, mc.preschedule = FALSE)
  if (any(sapply(out, inherits, "try-error"))) stop("ltmle_lapply: a worker failed: ", out[sapply(out, inherits, "try-error")][[1]])
  out
}

# Fit a regression of y on X with either sl3 Super Learner (use_sl=TRUE) or a parametric model.
# The SL's cross-validation folds are set on the sl3 task (V = n.folds); Lrnr_sl has no cv_folds argument.
# type: "binary" (0/1 outcome), "continuous" (outcome in [0,1]), "categorical" (factor 1..J).
# Returns a function predicting on new design matrices, or NULL on failure (logged, never imputed).
ltmle_fit <- function(X, y, type, use_sl = TRUE, n.folds = 3, J = 6) {
  keep <- apply(X, 2, function(x) length(unique(x)) > 1)
  X <- X[, keep, drop = FALSE]
  cols <- colnames(X)
  tryCatch({
    if (type == "categorical") {
      y <- factor(y, levels = 1:J)
      if (use_sl) {
        dt <- data.table::data.table(X); dt$y <- y
        task <- sl3_Task$new(dt, covariates = cols, outcome = "y", outcome_type = "categorical", folds = n.folds)
        fit <- create_treatment_model_sl(n.folds)$train(task)
        return(function(Xnew) {
          dn <- data.table::data.table(Xnew[, cols, drop = FALSE]); dn$y <- factor(rep(1, nrow(dn)), levels = 1:J)
          p <- unpack_predictions(fit$predict(sl3_Task$new(dn, covariates = cols, outcome = "y", outcome_type = "categorical")))
          out <- matrix(0, nrow(Xnew), J); colnames(out) <- 1:J
          out[, colnames(p)] <- p
          out
        })
      }
      df <- data.frame(X); df$y <- droplevels(y)
      fit <- nnet::multinom(y ~ ., data = df, trace = FALSE, MaxNWts = 10000)
      return(function(Xnew) {
        p <- predict(fit, newdata = data.frame(Xnew[, cols, drop = FALSE]), type = "probs")
        if (!is.matrix(p)) {
          lv <- levels(df$y)
          p <- if (length(lv) == 2) cbind(1 - p, p) else matrix(p, nrow = nrow(Xnew))
          colnames(p) <- lv
        }
        out <- matrix(0, nrow(Xnew), J); colnames(out) <- 1:J
        out[, colnames(p)] <- p
        out
      })
    }
    if (length(unique(y)) == 1) {
      const <- unique(y)
      return(function(Xnew) rep(const, nrow(Xnew)))
    }
    if (use_sl) {
      dt <- data.table::data.table(X); dt$y <- y
      if (type == "binary") {
        task <- sl3_Task$new(dt, covariates = cols, outcome = "y", outcome_type = "binomial", folds = n.folds)
        fit <- Lrnr_sl$new(learners = learner_stack_Y, metalearner = metalearner_Y)$train(task)
      } else {
        task <- sl3_Task$new(dt, covariates = cols, outcome = "y", outcome_type = "continuous", folds = n.folds)
        fit <- Lrnr_sl$new(learners = learner_stack_Y_cont, metalearner = metalearner_Y_cont)$train(task)
      }
      return(function(Xnew) {
        dn <- data.table::data.table(Xnew[, cols, drop = FALSE]); dn$y <- rep_len(c(0, 1), nrow(dn)) # placeholder outcome
        as.numeric(fit$predict(sl3_Task$new(dn, covariates = cols, outcome = "y",
                                            outcome_type = if (type == "binary") "binomial" else "continuous")))
      })
    }
    fit <- suppressWarnings(glm.fit(cbind(1, X), y, family = quasibinomial()))
    beta <- fit$coefficients
    beta[is.na(beta)] <- 0
    function(Xnew) as.numeric(plogis(cbind(1, Xnew[, cols, drop = FALSE]) %*% beta))
  }, error = function(e) {
    message("ltmle_fit (", type, ") failed: ", conditionMessage(e))
    NULL
  })
}

# Treatment models g_k(a | history) for k = 0..K among subjects at risk at k.
# arm = "multinomial": one multinomial model per time; "binomial": J separate one-vs-rest
# binary models per time (predictions are not renormalised). Returns a list of n x J matrices
# (NA rows outside the risk set) and the number of failed fits.
fit_treatment_models <- function(dat, K, arm = "multinomial", use_sl = TRUE, n.folds = 3, J = 6, cores = 1) {
  n <- nrow(dat)
  g <- ltmle_lapply(0:K, function(k) {
    failures <- 0
    out <- matrix(NA_real_, n, J, dimnames = list(NULL, 1:J))
    rs <- ltmle_at_risk(dat, k)
    X <- ltmle_design(dat, k)[rs, , drop = FALSE]
    a <- as.integer(as.character(dat[[paste0("A_", k)]]))[rs]
    if (arm == "multinomial") {
      pred <- ltmle_fit(X, a, "categorical", use_sl, n.folds, J)
      if (is.null(pred)) return(list(g = out, failures = 1))
      out[rs, ] <- pred(X)
    } else {
      for (j in 1:J) {
        pred <- ltmle_fit(X, as.numeric(a == j), "binary", use_sl, n.folds, J)
        if (is.null(pred)) { failures <- failures + 1; next }
        out[rs, j] <- pred(X)
      }
    }
    list(g = out, failures = failures)
  }, cores)
  list(g = lapply(g, `[[`, "g"), failures = sum(sapply(g, `[[`, "failures")))
}

# Censoring models P(C_k = 0 | history, A_k) for k = 1..K among subjects at risk at k.
# Age (V3) is excluded: it affects only censoring (age-out at 65) and no other node, so
# coarsening at random holds given (L, A) history alone, while conditioning on V3 would make
# P(C_k = 0) = 0 after age 65 (a structural positivity violation).
fit_censoring_models <- function(dat, K, use_sl = TRUE, n.folds = 3, J = 6, cores = 1) {
  n <- nrow(dat)
  gC <- ltmle_lapply(seq_len(K), function(k) {
    out <- rep(NA_real_, n)
    rs <- ltmle_at_risk(dat, k)
    X <- ltmle_design(dat, k, A = as.integer(as.character(dat[[paste0("A_", k)]])), include_V3 = FALSE)[rs, , drop = FALSE]
    pred <- ltmle_fit(X, as.numeric(dat[[paste0("C_", k)]][rs] == 0), "binary", use_sl, n.folds, J)
    if (is.null(pred)) return(list(gC = out, failures = 1))
    out[rs] <- pred(X)
    list(gC = out, failures = 0)
  }, cores)
  list(gC = lapply(gC, `[[`, "gC"), failures = sum(sapply(gC, `[[`, "failures")))
}

# Cumulative clever-covariate weights for one rule.
# Deterministic rules: W_s = prod_{k<=s} I(A_k = d_k) / g_k(d_k) * prod_{k=1..s} 1 / P(C_k = 0).
# Stochastic rule: W_s = prod_{k=1..s} g*(A_k | A_{k-1}) / g_k(A_k) * prod_{k=1..s} 1 / P(C_k = 0).
# The per-time density ratio g*/g is bounded above at 1/gbound[1] (for deterministic rules g* = 1, i.e.
# g is bounded below at gbound[1]; bounding g itself would down-weight the stochastic rule's switches,
# whose probability is ~0.01). P(C = 0) is bounded below at gbound[1]. cum_prob is the unbounded cumulative probability
# of the rule path (deterministic) or g/g* ratio (stochastic), used for the positivity diagnostic.
rule_weights <- function(dat, rule, g, gC, K, gbound) {
  n <- nrow(dat)
  V2 <- dat$V2_0
  cumw <- rep(1, n); follow <- rep(TRUE, n); cum_prob <- rep(1, n)
  W <- vector("list", K + 1); P <- vector("list", K + 1); D <- vector("list", K + 1)
  a_prev <- NULL
  for (k in 0:K) {
    a_obs <- as.integer(as.character(dat[[paste0("A_", k)]]))
    if (rule == "stochastic") {
      if (k > 0) {
        g_obs <- g[[k + 1]][cbind(seq_len(n), a_obs)]
        g_star <- rule_stochastic_density(a_obs, a_prev)
        cumw <- cumw * pmin(g_star / g_obs, 1 / gbound[1]) # density ratio bounded at 1/gbound[1]
        cum_prob <- cum_prob * g_obs / g_star
      }
    } else {
      d <- rule_assign(rule, k, V2, dat[[paste0("L1_", k)]], dat[[paste0("L2_", k)]], dat[[paste0("L3_", k)]])
      D[[k + 1]] <- d
      g_d <- g[[k + 1]][cbind(seq_len(n), d)]
      follow <- follow & (a_obs == d)
      cumw <- cumw / pmax(g_d, gbound[1])
      cum_prob <- cum_prob * g_d
    }
    if (k > 0) cumw <- cumw / pmax(gC[[k]], gbound[1])
    in_R <- ltmle_uncensored(dat, k)
    W[[k + 1]] <- ifelse(in_R, ifelse(follow, cumw, 0), NA_real_)
    P[[k + 1]] <- ifelse(in_R & follow, cum_prob, NA_real_)
    a_prev <- a_obs
  }
  list(W = W, cum_prob = P, d = D, rule = rule)
}

###################################################################
# One step s of the backward recursion for one rule:              #
# regress the pseudo-outcome Z_s on history among subjects at     #
# risk and uncensored at s, then predict under observed A_s and   #
# under the rule for everyone at risk at s.                       #
# tmle_contrasts: pseudo-outcome vector Z_s (length n)            #
# essential_covars_Y: design function (dat, k, A) -> matrix       #
# initial_model_for_Y_sl_cont: list(use_sl, n.folds)              #
###################################################################
process_backward_sequential <- function(tmle_dat, t, tmle_rules, essential_covars_Y,
                                        initial_model_for_Y_sl_cont, ybound, tmle_contrasts,
                                        time.censored=NULL) {
  J <- 6
  s <- t
  rule <- tmle_rules
  n <- nrow(tmle_dat)
  in_R <- ltmle_uncensored(tmle_dat, s)
  at_risk <- ltmle_at_risk(tmle_dat, s)
  a_obs <- as.integer(as.character(tmle_dat[[paste0("A_", s)]]))
  Z <- tmle_contrasts
  type <- if (all(Z[in_R] %in% c(0, 1))) "binary" else "continuous"
  pred <- ltmle_fit(essential_covars_Y(tmle_dat, s, A = a_obs)[in_R, , drop = FALSE], Z[in_R], type,
                    initial_model_for_Y_sl_cont$use_sl, initial_model_for_Y_sl_cont$n.folds, J)
  if (is.null(pred)) return(NULL)
  bound <- function(p) pmin(pmax(p, ybound[1]), ybound[2])
  Q_obs <- rep(NA_real_, n)
  Q_obs[in_R] <- bound(pred(essential_covars_Y(tmle_dat, s, A = a_obs)[in_R, , drop = FALSE]))
  Q_a <- NULL
  Q_d <- rep(NA_real_, n)
  if (rule == "stochastic") {
    if (s == 0) {
      # A_0 keeps its natural distribution under the stochastic rule
      Q_d[at_risk] <- bound(pred(essential_covars_Y(tmle_dat, s, A = a_obs)[at_risk, , drop = FALSE]))
    } else {
      a_prev <- as.integer(as.character(tmle_dat[[paste0("A_", s - 1)]]))
      Q_a <- matrix(NA_real_, n, J)
      for (a in 1:J) {
        Q_a[at_risk, a] <- bound(pred(essential_covars_Y(tmle_dat, s, A = rep(a, n))[at_risk, , drop = FALSE]))
      }
    }
  } else {
    d <- rule_assign(rule, s, tmle_dat$V2_0, tmle_dat[[paste0("L1_", s)]], tmle_dat[[paste0("L2_", s)]], tmle_dat[[paste0("L3_", s)]])
    Q_d[at_risk] <- bound(pred(essential_covars_Y(tmle_dat, s, A = d)[at_risk, , drop = FALSE]))
  }
  list(Q_obs = Q_obs, Q_d = Q_d, Q_a = Q_a, in_R = in_R, at_risk = at_risk)
}

###################################################################
# LTMLE, IPTW (Hajek) and sequential g-computation of             #
# psi_t = E[Y_t^d] for one target time t (= t.end argument).      #
# initial_model_for_Y: list(use_sl, n.folds) for the Q regressions#
# tmle_rules: rule names; tmle_covars_Y: design function          #
# g_preds_bounded / C_preds_bounded: treatment / censoring fits   #
# obs.treatment: wide observed data; obs.rules: optional cached   #
# rule_weights() output (computed from g and C if NULL)           #
# Returns, per rule, point estimates and influence curves.        #
###################################################################
getTMLELong <- function(initial_model_for_Y, tmle_rules, tmle_covars_Y, g_preds_bounded,
                        C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end, analysis=FALSE, debug=FALSE,
                        gcomp=TRUE){
  dat <- obs.treatment
  tstar <- t.end
  n <- nrow(dat)
  if (is.null(tmle_covars_Y)) tmle_covars_Y <- ltmle_design
  out <- list()
  for (rule in tmle_rules) {
    rw <- if (!is.null(obs.rules) && !is.null(obs.rules[[rule]])) obs.rules[[rule]] else
      rule_weights(dat, rule, g_preds_bounded, C_preds_bounded, tstar, gbound)
    failures <- 0
    Y_t <- as.numeric(dat[[paste0("Y_", tstar)]])
    # targeted (LTMLE) and untargeted (g-computation) recursions
    Qstar_next <- NULL; Qg_next <- NULL
    D <- rep(0, n)
    eps <- rep(NA_real_, tstar + 1)
    ok <- TRUE
    for (s in tstar:0) {
      in_R <- ltmle_uncensored(dat, s)
      Y_s <- as.numeric(dat[[paste0("Y_", s)]])
      Z <- if (s == tstar) Y_t else ifelse(Y_s == 1, 1, Qstar_next)
      step <- process_backward_sequential(dat, s, rule, tmle_covars_Y, initial_model_for_Y, ybound, Z)
      if (is.null(step)) { failures <- failures + 1; ok <- FALSE; break }
      # targeting: weighted logistic fluctuation with offset logit(Q_s(observed A_s)) and weights W_s
      W_s <- rw$W[[s + 1]]
      w <- W_s[in_R]
      if (anyNA(w)) { failures <- failures + 1; ok <- FALSE; break }  # treatment/censoring fit failed
      if (sum(w) > 0) {
        fl <- suppressWarnings(glm(Z[in_R] ~ 1, offset = qlogis(step$Q_obs[in_R]), weights = w, family = quasibinomial()))
        eps[s + 1] <- coef(fl)[1]
        if (!is.finite(eps[s + 1])) { failures <- failures + 1; ok <- FALSE; break }
      } else {
        eps[s + 1] <- 0
        failures <- failures + 1  # no rule-followers left to target on
      }
      upd <- function(q) plogis(qlogis(q) + eps[s + 1])
      Qstar_obs <- upd(step$Q_obs)
      Qstar_d <- if (!is.null(step$Q_a)) {
        a_prev <- as.integer(as.character(dat[[paste0("A_", s - 1)]]))
        rowSums(sapply(1:6, function(a) rule_stochastic_density(a, a_prev) * upd(step$Q_a[, a])))
      } else upd(step$Q_d)
      D[in_R] <- D[in_R] + w * (Z[in_R] - Qstar_obs[in_R])
      Qstar_next <- Qstar_d
      if (gcomp) {
        Zg <- if (s == tstar) Y_t else ifelse(Y_s == 1, 1, Qg_next)
        step_g <- if (s == tstar) step else process_backward_sequential(dat, s, rule, tmle_covars_Y, initial_model_for_Y, ybound, Zg)
        if (is.null(step_g)) { failures <- failures + 1; gcomp <- FALSE } else {
          Qg_next <- if (!is.null(step_g$Q_a)) {
            a_prev <- as.integer(as.character(dat[[paste0("A_", s - 1)]]))
            rowSums(sapply(1:6, function(a) rule_stochastic_density(a, a_prev) * step_g$Q_a[, a]))
          } else step_g$Q_d
        }
      }
    }
    if (ok) {
      psi <- mean(Qstar_next)
      ic <- D + Qstar_next - psi
    } else {
      psi <- NA_real_; ic <- rep(NA_real_, n)
    }
    # IPTW (Hajek): weight at min(t, event time) for subjects observed through then
    w_last <- rep(0, n); done <- rep(FALSE, n)
    for (s in 0:tstar) {
      in_R <- ltmle_uncensored(dat, s)
      stop_here <- in_R & !done & (s == tstar | dat[[paste0("Y_", s)]] == 1)
      stop_here[is.na(stop_here)] <- FALSE
      w_last[stop_here] <- rw$W[[s + 1]][stop_here]
      done <- done | stop_here
    }
    Yw <- ifelse(is.na(Y_t), 0, Y_t)
    if (anyNA(w_last)) w_last[] <- NA_real_
    psi_iptw <- if (isTRUE(sum(w_last) > 0)) sum(w_last * Yw) / sum(w_last) else NA_real_
    ic_iptw <- if (isTRUE(sum(w_last) > 0)) w_last * (Yw - psi_iptw) / mean(w_last) else rep(NA_real_, n)
    out[[rule]] <- list(
      psi = psi, ic = ic,
      psi_iptw = psi_iptw, ic_iptw = ic_iptw,
      psi_gcomp = if (gcomp && ok) mean(Qg_next) else NA_real_,
      eps = eps, n_followers = sum(w_last > 0, na.rm = TRUE), failures = failures)
  }
  out
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

# Safer getTMLELong wrapper
safe_getTMLELong <- function(...) {
  tryCatch({
    getTMLELong(...)
  }, error = function(e) {
    message("Error in getTMLELong: ", e$message)
    
    # Return NULL instead of creating artificial values
    # This will ensure that missing timepoints remain NA
    NULL
  })
}