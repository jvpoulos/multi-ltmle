###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

process_time_points <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, g_preds_bin_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size,
                                cores = 1, debug = FALSE, chunk_size = NULL) {
  
  # Initialize results
  n_ids <- length(unique(initial_model_for_Y_data$ID))
  n_rules <- length(tmle_rules)
  tmle_contrasts <- vector("list", t_end)
  tmle_contrasts_bin <- vector("list", t_end)
  
  if(debug) {
    cat(sprintf("\nProcessing %d IDs with %d rules\n", n_ids, n_rules))
  }
  
  # Pre-allocate results for both multinomial and binary cases
  for(t in 1:t_end) {
    tmle_contrasts[[t]] <- list(
      "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "epsilon" = rep(0, n_rules),
      "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
      "Y" = rep(ybound[1], n_ids)
    )
    tmle_contrasts_bin[[t]] <- list(
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
    
    # Process multinomial predictions
    current_g_preds <- process_g_preds(g_preds_processed, t, n_ids, J, gbound, debug)
    current_g_preds_list <- lapply(1:J, function(j) matrix(current_g_preds[,j], ncol=1))
    
    # Process binary predictions
    current_g_preds_bin <- process_g_preds(g_preds_bin_processed, t, n_ids, J, gbound, debug)
    current_g_preds_bin_list <- lapply(1:J, function(j) matrix(current_g_preds_bin[,j], ncol=1))
    
    # Get shared components (only need to compute once)
    current_c_preds <- get_c_preds(C_preds_processed, t, n_ids, gbound)
    current_y_preds <- get_y_preds(initial_model_for_Y, t, n_ids, ybound, debug)
    
    # Process both cases simultaneously
    tryCatch({
      # Multinomial case
      result_multi <- getTMLELongLSTM(
        initial_model_for_Y_preds = current_y_preds,
        initial_model_for_Y_data = initial_model_for_Y_data,
        tmle_rules = tmle_rules,
        tmle_covars_Y = tmle_covars_Y,
        g_preds_bounded = current_g_preds_list,
        C_preds_bounded = current_c_preds,
        obs.treatment = treatments[[min(t + 1, length(treatments))]],
        obs.rules = obs.rules[[min(t, length(obs.rules))]],
        gbound = gbound,
        ybound = ybound,
        t_end = t_end,
        window_size = window_size,
        debug = debug
      )
      
      # Binary case 
      result_bin <- getTMLELongLSTM(
        initial_model_for_Y_preds = current_y_preds,
        initial_model_for_Y_data = initial_model_for_Y_data,
        tmle_rules = tmle_rules,
        tmle_covars_Y = tmle_covars_Y,
        g_preds_bounded = current_g_preds_bin_list,
        C_preds_bounded = current_c_preds,
        obs.treatment = treatments[[min(t + 1, length(treatments))]],
        obs.rules = obs.rules[[min(t, length(obs.rules))]],
        gbound = gbound,
        ybound = ybound,
        t_end = t_end,
        window_size = window_size,
        debug = debug
      )
      
      # Store results for both cases
      store_results(tmle_contrasts[[t]], result_multi)
      store_results(tmle_contrasts_bin[[t]], result_bin)
      
    }, error = function(e) {
      if(debug) {
        cat(sprintf("\nError processing time point %d: %s\n", t, conditionMessage(e)))
        cat("Using default values\n")
      }
      # Use default values for both cases
      use_default_values(tmle_contrasts[[t]], ybound)
      use_default_values(tmle_contrasts_bin[[t]], ybound)
    })
    
    # Calculate IPTW weights and means for both cases
    tryCatch({
      current_rules <- obs.rules[[min(t, length(obs.rules))]]
      
      # Multinomial IPTW
      iptw_result_multi <- calculate_iptw(current_g_preds, current_rules, 
                                          tmle_contrasts[[t]]$Qstar_gcomp, 
                                          n_rules, gbound, debug)
      tmle_contrasts[[t]]$Qstar_iptw <- iptw_result_multi
      
      # Binary IPTW
      iptw_result_bin <- calculate_iptw(current_g_preds_bin, current_rules,
                                        tmle_contrasts_bin[[t]]$Qstar_gcomp,
                                        n_rules, gbound, debug)
      tmle_contrasts_bin[[t]]$Qstar_iptw <- iptw_result_bin
      
    }, error = function(e) {
      if(debug) log_iptw_error(e, current_g_preds, current_rules)
      tmle_contrasts[[t]]$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
      tmle_contrasts_bin[[t]]$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
    })
    
    # Clean up
    rm(current_g_preds, current_g_preds_bin, current_c_preds, current_y_preds)
    gc()
    
    if(debug) log_completion(t, time_start, tmle_contrasts[[t]], tmle_contrasts_bin[[t]])
  }
  
  # Return both sets of results
  return(list(
    "multinomial" = tmle_contrasts,
    "binary" = tmle_contrasts_bin
  ))
}

# Helper functions
process_g_preds <- function(preds_processed, t, n_ids, J, gbound, debug) {
  current_preds <- if(!is.null(preds_processed) && t <= length(preds_processed)) {
    preds <- preds_processed[[t]]
    if(is.null(preds)) {
      matrix(1/J, nrow=n_ids, ncol=J)
    } else {
      if(is.matrix(preds)) {
        if(ncol(preds) != J) {
          matrix(rep(preds, J), ncol=J)
        } else {
          preds
        }
      } else {
        matrix(rep(preds, J), ncol=J)
      }
    }
  } else {
    matrix(1/J, nrow=n_ids, ncol=J)
  }
  
  # Normalize
  t(apply(current_preds, 1, function(row) {
    if(any(!is.finite(row))) return(rep(1/J, J))
    row <- pmin(pmax(row, gbound[1]), gbound[2])
    row / sum(row)
  }))
}

get_c_preds <- function(C_preds_processed, t, n_ids, gbound) {
  if(!is.null(C_preds_processed) && t <= length(C_preds_processed)) {
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
}

get_y_preds <- function(initial_model_for_Y, t, n_ids, ybound, debug) {
  result <- tryCatch({
    if(is.list(initial_model_for_Y)) {
      if(!is.null(initial_model_for_Y$preds)) {
        preds <- initial_model_for_Y$preds
        if(is.matrix(preds)) {
          col_idx <- min(t, ncol(preds))
          preds <- preds[,col_idx]
        }
        extreme_idx <- preds < 0.001 | preds > 0.999
        preds[extreme_idx] <- pmin(pmax(preds[extreme_idx], 0.001), 0.999)
        matrix(preds, nrow=n_ids)
      } else {
        matrix(0.5, nrow=n_ids, ncol=1)
      }
    } else {
      matrix(initial_model_for_Y, nrow=n_ids)
    }
  }, error = function(e) {
    if(debug) cat("Error getting Y predictions:", conditionMessage(e), "\n")
    matrix(0.5, nrow=n_ids, ncol=1)
  })
  pmin(pmax(result, ybound[1]), ybound[2])
}

store_results <- function(tmle_contrast, result) {
  tmle_contrast$Qstar <- result$Qstar
  tmle_contrast$Qstar_gcomp <- result$Qstar_gcomp 
  tmle_contrast$Y <- result$Y
  if(any(result$epsilon != 0)) {
    tmle_contrast$epsilon <- result$epsilon
  }
}

use_default_values <- function(tmle_contrast, ybound) {
  tmle_contrast$Qstar[] <- ybound[1]
  tmle_contrast$Qstar_gcomp[] <- ybound[1]
  tmle_contrast$Y[] <- ybound[1]
}

calculate_iptw <- function(g_preds, rules, qstar_gcomp, n_rules, gbound, debug) {
  iptw_weights <- matrix(0, nrow=nrow(g_preds), ncol=n_rules)
  iptw_means <- numeric(n_rules)
  
  for(rule_idx in 1:n_rules) {
    valid_idx <- !is.na(rules[,rule_idx]) & rules[,rule_idx] == 1
    if(any(valid_idx)) {
      # Calculate stabilized weights with better bounds
      marginal_prob <- mean(rules[,rule_idx] == 1, na.rm=TRUE)
      weights <- rep(0, nrow(g_preds))
      rule_probs <- pmin(pmax(g_preds[,rule_idx], gbound[1]), gbound[2])
      weights[valid_idx] <- marginal_prob / rule_probs[valid_idx]
      weights <- pmin(weights, 5)  # Less aggressive truncation
      
      iptw_weights[,rule_idx] <- weights
      
      # Calculate weighted mean with better handling
      outcomes <- qstar_gcomp[,rule_idx]
      valid_outcomes <- valid_idx & !is.na(outcomes)
      if(any(valid_outcomes)) {
        total_weight <- sum(weights[valid_outcomes])
        if(total_weight > 0) {
          weighted_sum <- sum(weights[valid_outcomes] * outcomes[valid_outcomes])
          iptw_means[rule_idx] <- weighted_sum / total_weight
        }
      }
    }
  }
  
  # Ensure means are within bounds but not uniform
  iptw_means[iptw_means < 0.1] <- 0.1
  iptw_means[is.na(iptw_means)] <- 0.5
  
  return(matrix(iptw_means, nrow=1))
}

log_iptw_error <- function(e, g_preds, rules) {
  cat("Error calculating IPTW:\n")
  cat(conditionMessage(e), "\n")
  cat("Dimensions:\n")
  cat("g_preds:", paste(dim(g_preds), collapse=" x "), "\n") 
  cat("rules:", paste(dim(rules), collapse=" x "), "\n")
}

log_completion <- function(t, time_start, multi_result, bin_result) {
  time_end <- Sys.time()
  cat(sprintf("\nTime point %d completed in %.2f s\n", 
              t, as.numeric(difftime(time_end, time_start, units="secs"))))
  cat("Summary of multinomial Qstar_iptw:\n")
  print(multi_result$Qstar_iptw)
  cat("Summary of binary Qstar_iptw:\n")
  print(bin_result$Qstar_iptw)
}

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size = 7,
                            debug = FALSE) {
  
  final_result <- tryCatch({
    # Initialize dimensions
    if(is.null(initial_model_for_Y_data) || is.null(tmle_rules)) {
      stop("Missing required data or rules")
    }
 
    # Ensure valid_indices is not NULL
    valid_indices <- !is.na(rule_result$ID)
    if(length(valid_indices) == 0) {
      return(list(
        "Qstar" = matrix(0.5, nrow=n_ids, ncol=n_rules),
        "epsilon" = rep(0, n_rules),
        "Qstar_gcomp" = matrix(0.5, nrow=n_ids, ncol=n_rules),
        "Qstar_iptw" = matrix(0.5, nrow=1, ncol=n_rules),
        "Y" = rep(0.5, n_ids)
      ))
    }
    
    n_ids <- length(unique(initial_model_for_Y_data$ID))
    n_rules <- length(tmle_rules)
    J <- length(g_preds_bounded)  # Length of the list is J
    
    if(n_ids == 0 || n_rules == 0 || J == 0) {
      stop("Invalid dimensions: n_ids=", n_ids, ", n_rules=", n_rules, ", J=", J)
    }
    
    # Initialize predicted_Y with actual predictions
    predicted_Y <- matrix(NA, nrow=n_ids, ncol=n_rules)
    for(i in seq_len(n_rules)) {
      if(is.matrix(initial_model_for_Y_preds)) {
        predicted_Y[,i] <- initial_model_for_Y_preds[,1]
      } else {
        predicted_Y[,i] <- initial_model_for_Y_preds
      }
    }
    # Apply bounds to predictions
    predicted_Y <- pmin(pmax(predicted_Y, 0.1), 0.9)
    
    # Initialize predict_Qstar with same predictions
    predict_Qstar <- predicted_Y
    
    # Initialize epsilon
    epsilon <- rep(0, n_rules)
    
    # Calculate observed outcomes
    observed_Y <- if("Y" %in% colnames(initial_model_for_Y_data)) {
      y_data <- initial_model_for_Y_data 
      y_values <- y_data$Y
      y_values[is.na(y_values)] <- 0.5
      pmin(pmax(y_values[match(unique(initial_model_for_Y_data$ID), y_data$ID)], 0.1), 0.9)
    } else {
      rep(0.5, n_ids)
    }
    
    # Process each rule
    for(i in seq_len(n_rules)) {
      rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
      valid_indices <- match(rule_result$ID, unique(initial_model_for_Y_data$ID))
      valid_indices <- valid_indices[!is.na(valid_indices)]
      
      if(length(valid_indices) > 0) {
        # Convert list of treatment probabilities to matrix
        g_matrix <- do.call(cbind, lapply(g_preds_bounded, function(x) x[valid_indices,1]))
        
        # Get rule indicators
        rule_indicators <- obs.rules[, i]
        rule_indicators <- rule_indicators[valid_indices]
        
        # Calculate H (clever covariate) without C_preds_bounded
        H <- numeric(n_ids)
        H[valid_indices] <- rule_indicators / rowSums(g_matrix)  # Removed C_preds_bounded
        H <- pmin(pmax(H, -3), 3)  # Less aggressive bounds
        
        # Less restrictive valid data filtering
        valid_data_indices <- rule_indicators == 1  # Only require rule indicators
        
        if(any(valid_data_indices)) {
          glm_data <- data.frame(
            y = observed_Y[valid_indices][valid_data_indices],
            h = H[valid_indices][valid_data_indices],
            offset = qlogis(predict_Qstar[valid_indices, i][valid_data_indices])
          )
          
          # Less restrictive row filtering
          valid_rows <- !is.na(glm_data$y) & !is.na(glm_data$h) & !is.na(glm_data$offset)
          
          glm_data <- glm_data[valid_rows,]
          
          # Less restrictive GLM conditions
          if(nrow(glm_data) >= 5) {
            fit <- tryCatch({
              glm(y ~ h + offset(offset), 
                  family = binomial(),
                  data = glm_data,
                  control = glm.control(maxit = 100))
            }, error = function(e) NULL)
            
            if(!is.null(fit)) {
              eps <- coef(fit)["h"]
              if(!is.na(eps)) {
                epsilon[i] <- eps
              } else {
                epsilon[i] <- 0  # Default to 0 instead of NA
              }
              # Update predictions only if we got valid epsilon
              if(!is.na(epsilon[i])) {
                logit_pred <- qlogis(predict_Qstar[,i])
                logit_update <- epsilon[i] * H
                valid_update <- !is.na(H) & is.finite(H)
                logit_pred[valid_update] <- logit_pred[valid_update] + 
                  pmin(pmax(logit_update[valid_update], -2), 2)
                predict_Qstar[,i] <- pmin(pmax(plogis(logit_pred), 0.1), 0.9)
              }
            }
          }
        }
      }
    }
    
    # Calculate IPTW means
    iptw_means <- sapply(1:n_rules, function(i) {
      weights <- rep(0, n_ids)
      if(i <= ncol(obs.rules)) {
        valid_idx <- !is.na(obs.rules[,i]) & obs.rules[,i] == 1
        if(any(valid_idx)) {
          marginal_prob <- mean(obs.rules[,i] == 1, na.rm=TRUE)
          g_matrix <- do.call(cbind, g_preds_bounded)
          rule_probs <- rowSums(g_matrix[valid_idx,])
          weights[valid_idx] <- marginal_prob / pmax(rule_probs, gbound[1])
          weights <- pmin(weights, 5)
        }
      }
      
      if(any(weights > 0)) {
        outcomes <- if(!all(predict_Qstar[,i] == 0.5)) {
          predict_Qstar[,i]
        } else {
          predicted_Y[,i]
        }
        weighted.mean(outcomes[weights > 0], w=weights[weights > 0], na.rm=TRUE)
      } else {
        mean(predicted_Y[,i], na.rm=TRUE)
      }
    })
    
    if(debug) {
      cat("\nFinal summary:\n")
      cat("Qstar range:", paste(range(predict_Qstar, na.rm=TRUE), collapse="-"), "\n")
      cat("Qstar_gcomp range:", paste(range(predicted_Y, na.rm=TRUE), collapse="-"), "\n")
      cat("Epsilon values:", paste(epsilon, collapse=", "), "\n")
      cat("IPTW means:", paste(iptw_means, collapse=", "), "\n")
    }
    
    # Return both matrix and list formats
    return(list(
      "Qstar" = predict_Qstar,
      "epsilon" = epsilon,
      "Qstar_gcomp" = predicted_Y,
      "Qstar_iptw" = matrix(iptw_means, nrow=1),
      "Y" = observed_Y
    ))
    
  }, error = function(e) {
    if(debug) cat(sprintf("\nError in getTMLELongLSTM: %s\n", conditionMessage(e)))
    
    n_ids_safe <- if(!is.null(initial_model_for_Y_data)) {
      length(unique(initial_model_for_Y_data$ID))
    } else {
      1 
    }
    n_rules_safe <- if(!is.null(tmle_rules)) length(tmle_rules) else 1
    
    list(
      "Qstar" = matrix(0.5, nrow=n_ids_safe, ncol=n_rules_safe),
      "epsilon" = rep(0, n_rules_safe), 
      "Qstar_gcomp" = matrix(0.5, nrow=n_ids_safe, ncol=n_rules_safe),
      "Qstar_iptw" = matrix(0.5, nrow=1, ncol=n_rules_safe),
      "Y" = rep(0.5, n_ids_safe)
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