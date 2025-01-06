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
    
    # Ensure treatment probabilities maintain individual variation
    current_g_preds <- if(!is.null(g_preds_processed) && t <= length(g_preds_processed)) {
      preds <- g_preds_processed[[t]]
      if(is.null(preds)) {
        matrix(1/n_rules, nrow=n_ids, ncol=n_rules)
      } else {
        # Convert to matrix while preserving individual-level variation
        if(!is.matrix(preds)) {
          preds <- matrix(preds, ncol=1)
        }
        if(ncol(preds) == 1) {
          preds_mat <- matrix(0, nrow=nrow(preds), ncol=n_rules)
          for(i in 1:n_rules) {
            # Use individual-specific probabilities
            preds_mat[,i] <- preds[,1]
          }
          preds_mat
        } else {
          preds
        }
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
    # Calculate IPTW weights and means
    tryCatch({
      # Get observed treatments for current time point
      current_rules <- obs.rules[[min(t, length(obs.rules))]]
      qstar_gcomp <- tmle_contrasts[[t]]$Qstar_gcomp # Get this from result
      
      # Improve IPTW calculation
      iptw_weights <- matrix(0, nrow=nrow(current_g_preds), ncol=n_rules)
      iptw_means <- numeric(n_rules)
      
      for(rule_idx in 1:n_rules) {
        # Calculate stabilized weights
        valid_idx <- !is.na(current_rules[,rule_idx]) & current_rules[,rule_idx] == 1
        if(any(valid_idx)) {
          # Get marginal probability of rule
          marginal_prob <- mean(current_rules[,rule_idx] == 1, na.rm=TRUE)
          
          # Calculate stabilized weights with better bounds
          weights <- rep(0, nrow(current_g_preds))
          rule_probs <- pmin(pmax(current_g_preds[,rule_idx], gbound[1]), gbound[2])
          weights[valid_idx] <- marginal_prob / rule_probs[valid_idx]
          
          # Bound extreme weights
          weights <- pmin(weights, 10)
          
          iptw_weights[,rule_idx] <- weights
          
          # Calculate mean for this rule using the weights
          outcomes <- qstar_gcomp[,rule_idx]
          valid_outcomes <- valid_idx & !is.na(outcomes)
          if(any(valid_outcomes)) {
            total_weight <- sum(weights[valid_outcomes])
            if(total_weight > 0) {
              iptw_means[rule_idx] <- sum(weights[valid_outcomes] * 
                                            outcomes[valid_outcomes]) / total_weight
            }
          }
        }
      }
      
      tmle_contrasts[[t]]$Qstar_iptw <- matrix(iptw_means, nrow=1)
      
      if(debug) {
        cat("\nIPTW weight summary:\n")
        cat("Weight range:", paste(range(iptw_weights, na.rm=TRUE), collapse="-"), "\n")
        cat("Mean weights per rule:", paste(colMeans(iptw_weights, na.rm=TRUE), collapse=", "), "\n")
        cat("Number of non-zero weights per rule:", paste(colSums(iptw_weights > 0, na.rm=TRUE), collapse=", "), "\n")
      }
      
    }, error = function(e) {
      if(debug) {
        cat("Error calculating IPTW:\n")
        cat(conditionMessage(e), "\n")
        cat("Dimensions:\n")
        cat("current_g_preds:", paste(dim(current_g_preds), collapse=" x "), "\n") 
        cat("current_rules:", paste(dim(current_rules), collapse=" x "), "\n")
        cat("qstar_gcomp:", paste(dim(qstar_gcomp), collapse=" x "), "\n")
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

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size = 7,
                            debug = FALSE) {
  
  final_result <- tryCatch({
    # Initialize dimensions
    if(is.null(initial_model_for_Y_data) || is.null(tmle_rules)) {
      stop("Missing required data or rules")
    }
    
    n_ids <- length(unique(initial_model_for_Y_data$ID))
    n_rules <- length(tmle_rules)
    J <- if(is.matrix(g_preds_bounded)) ncol(g_preds_bounded) else length(g_preds_bounded)
    
    if(n_ids == 0 || n_rules == 0) {
      stop("Invalid dimensions: n_ids=", n_ids, ", n_rules=", n_rules)
    }
    
    # Initialize storage matrices
    predict_Qstar <- matrix(ybound[1], nrow=n_ids, ncol=n_rules)
    predicted_Y <- matrix(ybound[1], nrow=n_ids, ncol=n_rules)
    epsilon <- rep(0, n_rules)
    
    # Calculate observed outcomes
    observed_Y <- rep(ybound[1], n_ids)
    if("Y" %in% colnames(initial_model_for_Y_data)) {
      y_data <- initial_model_for_Y_data 
      y_data$Y[is.na(y_data$Y)] <- ybound[1]
      y_data$Y <- pmin(pmax(y_data$Y, ybound[1]), ybound[2])
      
      id_map <- match(unique(initial_model_for_Y_data$ID), y_data$ID)
      valid_map <- !is.na(id_map)
      if(any(valid_map)) {
        observed_Y[valid_map] <- y_data$Y[id_map[valid_map]]
      }
    }
    
    # Process each rule
    for(i in seq_len(n_rules)) {
      rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
      valid_indices <- match(rule_result$ID, unique(initial_model_for_Y_data$ID))
      valid_indices <- valid_indices[!is.na(valid_indices)]
      
      if(length(valid_indices) > 0) {
        # Get rule-specific probabilities
        rule_preds <- if(ncol(g_preds_bounded) == 1) {
          g_preds_bounded[valid_indices, 1]
        } else {
          g_preds_bounded[valid_indices, i]
        }
        
        # Get rule indicators from obs.rules
        rule_indicators <- obs.rules[, i]
        rule_indicators <- rule_indicators[valid_indices]
        
        # Calculate H (clever covariate)
        H <- numeric(n_ids)
        H[valid_indices] <- rule_indicators / 
          pmax(rule_preds * C_preds_bounded[valid_indices], gbound[1])
        
        # Get initial predictions
        if(is.matrix(initial_model_for_Y_preds)) {
          predicted_Y[valid_indices, i] <- initial_model_for_Y_preds[valid_indices,
                                                                     min(i, ncol(initial_model_for_Y_preds))]
        } else {
          predicted_Y[valid_indices, i] <- initial_model_for_Y_preds[valid_indices]
        }
        
        # Calculate valid data indicators BEFORE using them
        valid_data <- H != 0 & is.finite(H) & !is.na(observed_Y[valid_indices])
        
        # Fix GLM fitting to avoid NAs and extreme values
        if(sum(valid_data) > 0) {
          # Calculate bounded Y values 
          y_bounded <- pmin(pmax(observed_Y[valid_indices][valid_data], 0.001), 0.999)
          
          # Initial predictions bounded
          initial_bounded <- pmin(pmax(predicted_Y[valid_indices, i][valid_data], 0.001), 0.999)
          
          # Clever covariate
          h_valid <- H[valid_indices][valid_data]
          
          # Only proceed if we have variation in Y
          if(var(y_bounded) > 0) {
            glm_data <- data.frame(
              y = y_bounded,
              h = h_valid,
              offset = qlogis(initial_bounded)
            )
            
            # Remove any infinite/NA values 
            glm_data <- glm_data[is.finite(glm_data$y) & 
                                   is.finite(glm_data$h) & 
                                   is.finite(glm_data$offset),]
            
            if(nrow(glm_data) > 0) {
              fit <- try({
                glm(y ~ h + offset(offset), 
                    family = binomial(),
                    data = glm_data,
                    control = list(maxit = 100))
              }, silent = TRUE)
              
              if(!inherits(fit, "try-error")) {
                epsilon[i] <- coef(fit)["h"]
                
                # Bound epsilon
                epsilon[i] <- sign(epsilon[i]) * min(abs(epsilon[i]), 2)
                
                if(!is.na(epsilon[i])) {
                  # Calculate updated predictions on logit scale
                  logit_pred <- qlogis(pmin(pmax(predicted_Y[,i], 0.001), 0.999))
                  logit_update <- epsilon[i] * H[valid_indices]
                  
                  # Bound updates
                  logit_update <- pmin(pmax(logit_update, -4), 4)
                  
                  # Update predictions 
                  updated_logit <- logit_pred
                  updated_logit[valid_indices] <- updated_logit[valid_indices] + logit_update
                  
                  # Transform back maintaining bounds
                  predict_Qstar[,i] <- pmin(pmax(plogis(updated_logit), 0.001), 0.999)
                }
              }
            }
          }
        }
      }
    }
    
    # Calculate IPTW means with proper handling
    iptw_means <- colMeans(predict_Qstar, na.rm=TRUE)
    iptw_means[!is.finite(iptw_means)] <- mean(c(ybound[1], ybound[2]))
    iptw_means <- pmin(pmax(iptw_means, ybound[1]), ybound[2])
    
    if(debug) {
      cat("\nFinal summary:\n")
      cat("Qstar range:", paste(range(predict_Qstar, na.rm=TRUE), collapse="-"), "\n")
      cat("Qstar_gcomp range:", paste(range(predicted_Y, na.rm=TRUE), collapse="-"), "\n")
      cat("Epsilon values:", paste(epsilon, collapse=", "), "\n")  
      cat("IPTW means:", paste(iptw_means, collapse=", "), "\n")
    }
    
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
      "Qstar" = matrix(ybound[1], nrow=n_ids_safe, ncol=n_rules_safe),
      "epsilon" = rep(0, n_rules_safe),
      "Qstar_gcomp" = matrix(ybound[1], nrow=n_ids_safe, ncol=n_rules_safe),
      "Qstar_iptw" = matrix(ybound[1], nrow=1, ncol=n_rules_safe),
      "Y" = rep(ybound[1], n_ids_safe)
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