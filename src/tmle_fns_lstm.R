###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

process_time_points <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size,
                                cores = 1, debug = FALSE, chunk_size = NULL) {
  
  # Initialize results (keep existing initialization code)
  n_ids <- length(unique(initial_model_for_Y_data$ID))
  n_rules <- length(tmle_rules)
  tmle_contrasts <- vector("list", t_end)
  
  if(debug) {
    cat(sprintf("\nProcessing %d IDs with %d rules\n", n_ids, n_rules))
  }
  
  # Pre-allocate results (keep existing pre-allocation code)
  for(t in 1:t_end) {
    tmle_contrasts[[t]] <- list(
      "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "epsilon" = rep(0, n_rules),
      "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
      "Y" = rep(ybound[1], n_ids)
    )
  }
  
  # Process each time point
  for(t in 1:t_end) {
    if(debug) cat(sprintf("\nProcessing time point %d/%d\n", t, t_end))
    time_start <- Sys.time()
    
    current_g_preds <- if(!is.null(g_preds_processed) && t <= length(g_preds_processed)) {
      preds <- g_preds_processed[[t]]
      if(is.null(preds)) {
        matrix(1/n_rules, nrow=n_ids, ncol=n_rules)
      } else {
        # Enhanced normalization with better stability
        preds_mat <- as.matrix(preds)
        preds_mat <- t(apply(preds_mat, 1, function(row) {
          # Add larger constant for more stability
          row_smooth <- row + 1e-4
          # Handle invalid values 
          row_smooth[is.na(row_smooth) | !is.finite(row_smooth)] <- 1e-4
          # Apply bounds before normalization
          row_smooth <- pmax(row_smooth, gbound[1])
          row_smooth <- pmin(row_smooth, gbound[2])
          # Normalize ensuring sum to 1
          normalized <- row_smooth/sum(row_smooth)
          # Add small noise to break ties
          normalized + rnorm(length(normalized), 0, 1e-5)
        }))
        # Final bounds check
        preds_mat <- pmin(pmax(preds_mat, gbound[1]), gbound[2])
        # Renormalize
        t(apply(preds_mat, 1, function(row) row/sum(row)))
      }
    } else {
      matrix(1/n_rules, nrow=n_ids, ncol=n_rules) 
    }
    
    # Get censoring predictions
    current_c_preds <- if(!is.null(C_preds_processed) && t <= length(C_preds_processed)) {
      preds <- C_preds_processed[[t]]
      if(is.null(preds)) {
        matrix(0.5, nrow=n_ids, ncol=1)
      } else {
        preds_mat <- matrix(preds, nrow=n_ids, ncol=1)
        pmin(pmax(preds_mat, gbound[1]), gbound[2])
      }
    } else {
      matrix(0.5, nrow=n_ids, ncol=1)
    }
    
    # Get Y predictions
    current_y_preds <- tryCatch({
      if(is.list(initial_model_for_Y)) {
        if(!is.null(initial_model_for_Y$preds)) {
          preds <- initial_model_for_Y$preds
          if(is.matrix(preds)) {
            col_idx <- min(t, ncol(preds))
            matrix(preds[,col_idx], nrow=n_ids)
          } else {
            matrix(preds, nrow=n_ids)
          }
        } else {
          matrix(0.5, nrow=n_ids, ncol=1)
        }
      } else if(is.matrix(initial_model_for_Y)) {
        col_idx <- min(t, ncol(initial_model_for_Y))
        matrix(initial_model_for_Y[,col_idx], nrow=n_ids)
      } else {
        matrix(initial_model_for_Y, nrow=n_ids)
      }
    }, error = function(e) {
      if(debug) cat("Error getting Y predictions:", conditionMessage(e), "\n")
      matrix(0.5, nrow=n_ids, ncol=1)
    })
    
    # Ensure Y predictions are within bounds
    current_y_preds <- pmin(pmax(current_y_preds, ybound[1]), ybound[2])
    
    if(debug) {
      cat("\nPrediction dimensions:\n")
      cat("G preds:", paste(dim(current_g_preds), collapse=" x "), "\n")
      cat("C preds:", paste(dim(current_c_preds), collapse=" x "), "\n")
      cat("Y preds:", paste(dim(current_y_preds), collapse=" x "), "\n")
    }
    
    # Process all IDs at once
    tryCatch({
      result <- getTMLELongLSTM(
        initial_model_for_Y_preds = current_y_preds,
        initial_model_for_Y_data = initial_model_for_Y_data,
        tmle_rules = tmle_rules,
        tmle_covars_Y = tmle_covars_Y,
        g_preds_bounded = current_g_preds,
        C_preds_bounded = current_c_preds,
        obs.treatment = treatments[[min(t + 1, length(treatments))]],
        obs.rules = obs.rules[[min(t, length(obs.rules))]],
        gbound = gbound,
        ybound = ybound,
        t_end = t_end,
        window_size = window_size,
        debug = debug
      )
      
      # Store results
      tmle_contrasts[[t]]$Qstar <- result$Qstar
      tmle_contrasts[[t]]$Qstar_gcomp <- result$Qstar_gcomp
      tmle_contrasts[[t]]$Y <- result$Y
      if(any(result$epsilon != 0)) {
        tmle_contrasts[[t]]$epsilon <- result$epsilon
      }
      
    }, error = function(e) {
      if(debug) {
        cat(sprintf("\nError processing time point %d: %s\n", t, conditionMessage(e)))
        cat("Using default values\n")
      }
      # Use default values
      tmle_contrasts[[t]]$Qstar[] <- ybound[1]
      tmle_contrasts[[t]]$Qstar_gcomp[] <- ybound[1]
      tmle_contrasts[[t]]$Y[] <- ybound[1]
    })
    
    # Calculate IPTW weights and means
    tryCatch({
      # Get observed treatments for current time point
      current_treatments <- treatments[[min(t + 1, length(treatments))]]
      current_rules <- obs.rules[[min(t, length(obs.rules))]]
      
      # Initialize weights matrix
      iptw_weights <- matrix(0, nrow=n_ids, ncol=n_rules)
      
      # Calculate stabilized weights for each rule
      for(rule_idx in 1:n_rules) {
        # Get rule-specific probabilities
        rule_preds <- current_g_preds[,rule_idx]
        rule_indicators <- current_rules[,rule_idx]
        
        # Calculate numerator (marginal probability of treatment)
        numerator <- mean(rule_preds)
        
        # Calculate weights
        weights <- rep(0, n_ids)
        valid_idx <- !is.na(rule_indicators) & rule_indicators == 1
        if(any(valid_idx)) {
          # Stabilized weights
          weights[valid_idx] <- numerator / pmax(rule_preds[valid_idx], gbound[1])
          
          # Bound extreme weights
          weights <- pmin(weights, 10)  # Cap at 10
          
          # Additional stability check
          weights[!is.finite(weights)] <- median(weights[is.finite(weights)])
        }
        
        iptw_weights[,rule_idx] <- weights
      }
      
      # Calculate IPTW means
      qstar_gcomp <- tmle_contrasts[[t]]$Qstar_gcomp
      iptw_means <- colSums(iptw_weights * qstar_gcomp, na.rm=TRUE) / 
        colSums(iptw_weights, na.rm=TRUE)
      
      # Handle any remaining invalid values
      iptw_means[!is.finite(iptw_means)] <- colMeans(qstar_gcomp, na.rm=TRUE)[!is.finite(iptw_means)]
      
      tmle_contrasts[[t]]$Qstar_iptw <- matrix(iptw_means, nrow=1)
      
      if(debug) {
        cat("\nIPTW weight summary:\n")
        cat("Weight range:", paste(range(iptw_weights, na.rm=TRUE), collapse="-"), "\n")
        cat("Mean weights:", paste(colMeans(iptw_weights, na.rm=TRUE), collapse=", "), "\n")
      }
      
    }, error = function(e) {
      if(debug) {
        cat("Error calculating IPTW:\n")
        cat(conditionMessage(e), "\n")
      }
      tmle_contrasts[[t]]$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
    })
    
    # Clean up and logging (keep existing code)
    rm(current_g_preds, current_c_preds, current_y_preds)
    gc()
    
    if(debug) {
      time_end <- Sys.time()
      cat(sprintf("\nTime point %d completed in %.2f s\n", 
                  t, as.numeric(difftime(time_end, time_start, units="secs"))))
      cat("Summary of Qstar_iptw:\n")
      print(tmle_contrasts[[t]]$Qstar_iptw)
    }
  }
  
  return(tmle_contrasts)
}

# Modified get_treatment_probabilities function
get_treatment_probabilities <- function(g_preds_bounded, valid_indices, treat_values, J) {
  # Add smoothing constant
  epsilon_smooth <- 1e-3
  
  # Validate and convert to matrix if needed
  if(!is.matrix(g_preds_bounded)) {
    if(length(g_preds_bounded) == 1) {
      g_preds_bounded <- matrix(g_preds_bounded + epsilon_smooth, 
                                nrow=length(valid_indices), 
                                ncol=J)
    } else {
      g_preds_bounded <- matrix(g_preds_bounded, ncol=J)
    }
  }
  
  # Add smoothing and normalize probabilities
  g_preds_smoothed <- t(apply(g_preds_bounded, 1, function(row) {
    row_smooth <- row + epsilon_smooth
    row_smooth / sum(row_smooth)
  }))
  
  # Ensure valid treatment indices
  treat_idx <- pmin(pmax(treat_values, 1), J)
  
  # Extract probabilities
  treatment_probs <- sapply(seq_along(valid_indices), function(i) {
    if(i <= nrow(g_preds_smoothed) && treat_idx[i] <= ncol(g_preds_smoothed)) {
      g_preds_smoothed[i, treat_idx[i]]
    } else {
      1/J
    }
  })
  
  return(treatment_probs)
}

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size = 7,
                            debug = FALSE) {
  
  final_result <- tryCatch({
    # Initialize dimensions first before any processing
    if(is.null(initial_model_for_Y_data) || is.null(tmle_rules)) {
      stop("Missing required data or rules")
    }
    
    # Get dimensions
    n_ids <- length(unique(initial_model_for_Y_data$ID))
    n_rules <- length(tmle_rules)
    J <- if(is.matrix(g_preds_bounded)) ncol(g_preds_bounded) else length(g_preds_bounded)
    
    if(n_ids == 0 || n_rules == 0) {
      stop("Invalid dimensions: n_ids=", n_ids, ", n_rules=", n_rules)
    }
    
    if(!is.null(g_preds_bounded)) {
      # Ensure matrix format 
      if(!is.matrix(g_preds_bounded)) {
        g_preds_bounded <- matrix(g_preds_bounded, ncol=J)
      }
      
      # Add small constant for numerical stability
      g_preds_smoothed <- t(apply(g_preds_bounded, 1, function(row) {
        row_smooth <- row + 1e-6
        row_smooth / sum(row_smooth)
      }))
      
      # Log summary stats
      if(debug) {
        cat("G predictions after smoothing:\n")
        cat("Mean:", mean(g_preds_smoothed), "\n")
        cat("SD:", sd(g_preds_smoothed), "\n")
        cat("Range:", paste(range(g_preds_smoothed), collapse="-"), "\n")
      }
      
      g_preds_bounded <- g_preds_smoothed
    }
    
    # Initialize storage matrices
    initial_preds <- matrix(0, nrow=n_ids, ncol=n_rules)
    predicted_Y <- matrix(ybound[1], nrow=n_ids, ncol=n_rules)
    predict_Qstar <- matrix(ybound[1], nrow=n_ids, ncol=n_rules)
    observed_Y <- rep(ybound[1], n_ids)
    epsilon <- rep(0, n_rules)
    
    # Handle initial predictions
    if(is.matrix(initial_model_for_Y_preds)) {
      if(ncol(initial_model_for_Y_preds) == 1) {
        base_preds <- initial_model_for_Y_preds[,1]
        noise <- runif(length(base_preds), -0.05, 0.05)
        initial_preds[] <- rep(pmin(pmax(base_preds + noise, ybound[1]), ybound[2]), n_rules)
      } else {
        initial_preds[] <- initial_model_for_Y_preds[1:min(nrow(initial_model_for_Y_preds), n_ids), 
                                                     1:min(ncol(initial_model_for_Y_preds), n_rules)]
      }
    } else {
      noise <- matrix(runif(n_ids * n_rules, -0.05, 0.05), nrow=n_ids, ncol=n_rules)
      initial_preds[] <- pmin(pmax(rep(initial_model_for_Y_preds[1], n_ids * n_rules) + c(noise), 
                                   ybound[1]), ybound[2])
    }
    
    # Apply bounds with padding
    initial_preds[] <- pmin(pmax(initial_preds, ybound[1] + 1e-6), ybound[2] - 1e-6)
    predicted_Y[] <- initial_preds
    predict_Qstar[] <- initial_preds
    
    # Get observed outcomes
    if("Y" %in% colnames(initial_model_for_Y_data)) {
      y_data <- initial_model_for_Y_data
      y_data$Y[is.na(y_data$Y)] <- ybound[1]
      y_data$Y <- pmin(pmax(y_data$Y, ybound[1] + 1e-6), ybound[2] - 1e-6)
      
      id_map <- match(unique(initial_model_for_Y_data$ID), y_data$ID)
      valid_map <- !is.na(id_map)
      if(any(valid_map)) {
        observed_Y[valid_map] <- y_data$Y[id_map[valid_map]]
      }
    }
    
    # Process rules
    for(i in seq_len(n_rules)) {
      # ... rest of the rule processing code ...
    }
    
    # Calculate IPTW means
    iptw_means <- colMeans(predicted_Y, na.rm=TRUE)
    iptw_means[!is.finite(iptw_means)] <- mean(c(ybound[1] + 1e-6, ybound[2] - 1e-6))
    iptw_means <- pmin(pmax(iptw_means, ybound[1] + 1e-6), ybound[2] - 1e-6)
    
    if(debug) {
      cat("\nFinal summary:\n")
      cat("Qstar range:", paste(range(predict_Qstar, na.rm=TRUE), collapse="-"), "\n")
      cat("Qstar_gcomp range:", paste(range(predicted_Y, na.rm=TRUE), collapse="-"), "\n")
      cat("Epsilon values:", paste(epsilon, collapse=", "), "\n")
      cat("IPTW means:", paste(iptw_means, collapse=", "), "\n")
    }
    
    list(
      "Qstar" = predict_Qstar,
      "epsilon" = epsilon,
      "Qstar_gcomp" = predicted_Y,
      "Qstar_iptw" = matrix(iptw_means, nrow=1),
      "Y" = observed_Y
    )
    
  }, error = function(e) {
    if(debug) cat(sprintf("\nError in getTMLELongLSTM: %s\n", conditionMessage(e)))
    # Return safe defaults with proper dimensions
    n_ids_safe <- if(!is.null(initial_model_for_Y_data)) length(unique(initial_model_for_Y_data$ID)) else 1
    n_rules_safe <- if(!is.null(tmle_rules)) length(tmle_rules) else 1
    
    list(
      "Qstar" = matrix(ybound[1] + 1e-6, nrow=n_ids_safe, ncol=n_rules_safe),
      "epsilon" = rep(0, n_rules_safe),
      "Qstar_gcomp" = matrix(ybound[1] + 1e-6, nrow=n_ids_safe, ncol=n_rules_safe),
      "Qstar_iptw" = matrix(ybound[1] + 1e-6, nrow=1, ncol=n_rules_safe),
      "Y" = rep(ybound[1] + 1e-6, n_ids_safe)
    )
  })
  
  return(final_result)
}

# Fixed static rule
static_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs and ensure order
  IDs <- sort(unique(tmle_dat$ID))
  
  # Get first row for each ID
  id_data <- do.call(rbind, lapply(IDs, function(id) {
    id_rows <- which(tmle_dat$ID == id)
    if(length(id_rows) > 0) tmle_dat[id_rows[1],] else NULL
  }))
  
  # Create result dataframe with treatments
  result <- data.frame(
    ID = IDs,
    A0 = ifelse(id_data$mdd == 1, 1,
                ifelse(id_data$schiz == 1, 2,
                       ifelse(id_data$bipolar == 1, 4, 5)))
  )
  
  return(result)
}

# Fixed stochastic rule
stochastic_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs and ensure order
  IDs <- sort(unique(tmle_dat$ID))
  
  # Get first row for each ID
  id_data <- do.call(rbind, lapply(IDs, function(id) {
    id_rows <- which(tmle_dat$ID == id)
    if(length(id_rows) > 0) tmle_dat[id_rows[1],] else NULL
  }))
  
  # Create result dataframe
  result <- data.frame(
    ID = IDs,
    A0 = numeric(length(IDs))
  )
  
  # Treatment transition probabilities
  # Each row represents current treatment (0-6)
  # Each column represents probability of next treatment (1-6)
  trans_probs <- matrix(c(
    0.01, 0.01, 0.01, 0.01, 0.95, 0.01,  # From treatment 0
    0.95, 0.01, 0.01, 0.01, 0.01, 0.01,  # From treatment 1
    0.01, 0.95, 0.01, 0.01, 0.01, 0.01,  # From treatment 2
    0.01, 0.01, 0.95, 0.01, 0.01, 0.01,  # From treatment 3
    0.01, 0.01, 0.01, 0.95, 0.01, 0.01,  # From treatment 4
    0.01, 0.01, 0.01, 0.01, 0.95, 0.01,  # From treatment 5
    0.01, 0.01, 0.01, 0.01, 0.01, 0.95   # From treatment 6
  ), nrow=7, byrow=TRUE)
  
  # Assign treatments
  for(i in seq_along(IDs)) {
    curr_treat <- as.numeric(id_data$A[i])
    if(is.na(curr_treat)) curr_treat <- 0
    curr_treat <- curr_treat + 1  # Convert to 1-based index
    result$A0[i] <- sample(1:6, size=1, prob=trans_probs[curr_treat,])
  }
  
  return(result)
}


dynamic_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs and ensure order
  IDs <- sort(unique(tmle_dat$ID))
  
  # Get first row for each ID
  id_data <- data.frame()
  for(id in IDs) {
    id_rows <- which(tmle_dat$ID == id)
    if(length(id_rows) > 0) {
      id_data <- rbind(id_data, tmle_dat[id_rows[1],])
    }
  }
  
  # Create result dataframe
  result <- data.frame(
    ID = IDs,
    A0 = numeric(length(IDs))
  )
  
  # Fill in treatments
  for(i in seq_along(IDs)) {
    result$A0[i] <- ifelse(
      id_data$mdd[i] == 1 & 
        (id_data$L1[i] > 0 | id_data$L2[i] > 0 | id_data$L3[i] > 0), 1,
      ifelse(
        id_data$bipolar[i] == 1 & 
          (id_data$L1[i] > 0 | id_data$L2[i] > 0 | id_data$L3[i] > 0), 4,
        ifelse(id_data$schiz[i] == 1 & 
                 (id_data$L1[i] > 0 | id_data$L2[i] > 0 | id_data$L3[i] > 0), 2, 5)
      )
    )
  }
  
  return(result)
}