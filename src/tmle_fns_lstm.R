###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

process_time_points <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, g_preds_bin_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size,
                                cores = 1, debug = FALSE) {
  
  # Initialize results
  n_ids <- length(unique(initial_model_for_Y_data$ID))
  n_rules <- length(tmle_rules)
  
  if(debug) {
    cat(sprintf("\nProcessing %d IDs with %d rules\n", n_ids, n_rules))
    cat(sprintf("Using %d cores\n", cores))
  }
  
  # Pre-allocate list of time points to process
  time_points <- 1:t_end
  
  # Setup parallel processing
  if(cores > 1) {
    cl <- parallel::makeCluster(cores)
    
    # Set up debug output at cluster level first
    parallel::clusterEvalQ(cl, {
      debug_output <- function(...) {
        if(debug) {
          msg <- paste0(...)
          cat(msg, "\n", file=stderr())
          flush(stderr())
        }
      }
      
      # Define debug_print using debug_output
      debug_print <- function(...) {
        debug_output(...)
      }
    })
    
    # Export debug flag first
    parallel::clusterExport(cl, "debug")
    
    # Export main functions
    parallel::clusterExport(cl, c("getTMLELongLSTM", "process_g_preds", 
                                  "get_c_preds", "get_y_preds", "store_results", 
                                  "use_default_values", "calculate_iptw", 
                                  "log_iptw_error", "log_completion",
                                  "static_mtp_lstm", "dynamic_mtp_lstm", 
                                  "stochastic_mtp_lstm"))
    
    # Create and export environment variables
    cluster_env <- list(
      debug = debug,
      J = length(g_preds_processed[[1]][[1]]),
      n_ids = n_ids,
      n_rules = n_rules,
      t_end = t_end,
      window_size = window_size,
      initial_model_for_Y = initial_model_for_Y,
      initial_model_for_Y_data = initial_model_for_Y_data,
      tmle_rules = tmle_rules,
      tmle_covars_Y = tmle_covars_Y,
      g_preds_processed = g_preds_processed,
      g_preds_bin_processed = g_preds_bin_processed,
      C_preds_processed = C_preds_processed,
      treatments = treatments,
      obs.rules = obs.rules,
      gbound = gbound,
      ybound = ybound
    )
    
    # Export variables
    parallel::clusterExport(cl, names(cluster_env), envir=list2env(cluster_env))
    
    # Load required packages
    parallel::clusterEvalQ(cl, {
      library(stats)
    })
    
    # Process time points in parallel
    results <- parallel::parLapply(cl, time_points, function(t) {
      # Process single time point (same code as before)
      if(debug) debug_print(sprintf("\nProcessing time point %d/%d\n", t, t_end))
      time_start <- Sys.time()
      
      # Initialize results for this time point
      tmle_contrast <- list(
        "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
        "epsilon" = rep(0, n_rules),
        "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
        "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
        "Y" = rep(ybound[1], n_ids)
      )
      tmle_contrast_bin <- list(
        "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
        "epsilon" = rep(0, n_rules),
        "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
        "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
        "Y" = rep(ybound[1], n_ids)
      )
      
      # Process multinomial predictions
      current_g_preds <- process_g_preds(g_preds_processed, t, n_ids, J, gbound, debug)
      current_g_preds_list <- lapply(1:J, function(j) matrix(current_g_preds[,j], ncol=1))
      
      # Process binary predictions
      current_g_preds_bin <- process_g_preds(g_preds_bin_processed, t, n_ids, J, gbound, debug)
      current_g_preds_bin_list <- lapply(1:J, function(j) matrix(current_g_preds_bin[,j], ncol=1))
      
      # Get shared components
      current_c_preds <- get_c_preds(C_preds_processed, t, n_ids, gbound)
      current_y_preds <- get_y_preds(initial_model_for_Y, t, n_ids, ybound, debug)
      
      # Process both cases
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
        
        # Store results
        store_results(tmle_contrast, result_multi)
        store_results(tmle_contrast_bin, result_bin)
        
      }, error = function(e) {
        if(debug) {
          debug_print(sprintf("\nError processing time point %d: %s\n", t, conditionMessage(e)))
          debug_print("Using default values\n")
        }
        use_default_values(tmle_contrast, ybound)
        use_default_values(tmle_contrast_bin, ybound)
      })
      
      # Calculate IPTW weights and means
      tryCatch({
        current_rules <- obs.rules[[min(t, length(obs.rules))]]
        
        # Multinomial IPTW
        iptw_result_multi <- calculate_iptw(current_g_preds, current_rules, 
                                            tmle_contrast$Qstar_gcomp, 
                                            n_rules, gbound, debug)
        tmle_contrast$Qstar_iptw <- iptw_result_multi
        
        # Binary IPTW
        iptw_result_bin <- calculate_iptw(current_g_preds_bin, current_rules,
                                          tmle_contrast_bin$Qstar_gcomp,
                                          n_rules, gbound, debug)
        tmle_contrast_bin$Qstar_iptw <- iptw_result_bin
        
      }, error = function(e) {
        if(debug) log_iptw_error(e, current_g_preds, current_rules)
        tmle_contrast$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
        tmle_contrast_bin$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
      })
      
      if(debug) {
        time_end <- Sys.time()
        debug_print(sprintf("\nTime point %d completed in %.2f s\n", 
                            t, as.numeric(difftime(time_end, time_start, units="secs"))))
      }
      
      # Return results for this time point
      list(multinomial = tmle_contrast, 
           binary = tmle_contrast_bin)
    })
    
    parallel::stopCluster(cl)
    
  } else {
    # Sequential processing 
    results <- lapply(time_points, function(t) {
      # Process single time point
      if(debug) cat(sprintf("\nProcessing time point %d/%d\n", t, t_end))
      time_start <- Sys.time()
      
      # Initialize results for this time point
      tmle_contrast <- list(
        "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
        "epsilon" = rep(0, n_rules),
        "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
        "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
        "Y" = rep(ybound[1], n_ids)
      )
      tmle_contrast_bin <- list(
        "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
        "epsilon" = rep(0, n_rules),
        "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
        "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
        "Y" = rep(ybound[1], n_ids)
      )
      
      # Process multinomial predictions
      current_g_preds <- process_g_preds(g_preds_processed, t, n_ids, J, gbound, debug)
      current_g_preds_list <- lapply(1:J, function(j) matrix(current_g_preds[,j], ncol=1))
      
      # Process binary predictions  
      current_g_preds_bin <- process_g_preds(g_preds_bin_processed, t, n_ids, J, gbound, debug)
      current_g_preds_bin_list <- lapply(1:J, function(j) matrix(current_g_preds_bin[,j], ncol=1))
      
      # Get shared components
      current_c_preds <- get_c_preds(C_preds_processed, t, n_ids, gbound)
      current_y_preds <- get_y_preds(initial_model_for_Y, t, n_ids, ybound, debug)
      
      # Process both cases
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
        
        # Store results
        store_results(tmle_contrast, result_multi)
        store_results(tmle_contrast_bin, result_bin)
        
      }, error = function(e) {
        if(debug) {
          cat(sprintf("\nError processing time point %d: %s\n", t, conditionMessage(e)))
          cat("Using default values\n")
        }
        use_default_values(tmle_contrast, ybound)
        use_default_values(tmle_contrast_bin, ybound)
      })
      
      # Calculate IPTW weights and means
      tryCatch({
        current_rules <- obs.rules[[min(t, length(obs.rules))]]
        
        # Multinomial IPTW
        iptw_result_multi <- calculate_iptw(current_g_preds, current_rules, 
                                            tmle_contrast$Qstar_gcomp, 
                                            n_rules, gbound, debug)
        tmle_contrast$Qstar_iptw <- iptw_result_multi
        
        # Binary IPTW
        iptw_result_bin <- calculate_iptw(current_g_preds_bin, current_rules,
                                          tmle_contrast_bin$Qstar_gcomp,
                                          n_rules, gbound, debug)
        tmle_contrast_bin$Qstar_iptw <- iptw_result_bin
        
      }, error = function(e) {
        if(debug) log_iptw_error(e, current_g_preds, current_rules)
        tmle_contrast$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
        tmle_contrast_bin$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
      })
      
      if(debug) {
        time_end <- Sys.time()
        cat(sprintf("\nTime point %d completed in %.2f s\n", 
                    t, as.numeric(difftime(time_end, time_start, units="secs"))))
      }
      
      # Return results for this time point
      list(multinomial = tmle_contrast,
           binary = tmle_contrast_bin)
    })
  }
  
  # Restructure results into final format
  tmle_contrasts <- vector("list", t_end)
  tmle_contrasts_bin <- vector("list", t_end) 
  
  for(t in 1:t_end) {
    tmle_contrasts[[t]] <- results[[t]]$multinomial
    tmle_contrasts_bin[[t]] <- results[[t]]$binary
  }
  
  return(list(
    "multinomial" = tmle_contrasts,
    "binary" = tmle_contrasts_bin
  ))
}

# Helper functions
process_g_preds <- function(preds_processed, t, n_ids, J, gbound, debug) {
  if(!is.null(preds_processed) && t <= length(preds_processed)) {
    preds <- preds_processed[[t]]
    
    if(is.null(preds)) {
      if(debug) cat("No predictions for time", t, "using uniform\n")
      return(matrix(1/J, nrow=n_ids, ncol=J))
    }
    
    # Convert to matrix with proper dimensions
    if(!is.matrix(preds)) {
      if(debug) cat("Converting predictions to matrix\n")
      # Ensure nrow=n_ids is first dimension
      preds <- matrix(preds, nrow=n_ids, byrow=FALSE)
    }
    
    # Ensure dimensions are correct
    if(ncol(preds) != J) {
      if(debug) cat("Adjusting matrix dimensions\n")
      if(ncol(preds) == 1 && J > 1) {
        # Expand to J columns for multinomial case
        expanded <- matrix(0, nrow=n_ids, ncol=J)
        treatments <- round(preds[,1])
        for(i in 1:n_ids) {
          if(treatments[i] >= 0 && treatments[i] < J) {
            expanded[i,] <- 0.1/(J-1)
            expanded[i, treatments[i] + 1] <- 0.9
          } else {
            expanded[i,] <- 1/J
          }
        }
        preds <- expanded
      }
    }
    
    # Normalize probabilities
    if(debug) cat("Normalizing probabilities\n")
    preds <- t(apply(preds, 1, function(row) {
      if(any(!is.finite(row))) return(rep(1/J, J))
      row <- pmin(pmax(row, gbound[1]), gbound[2])
      row / sum(row)
    }))
    
    if(debug) {
      cat("Final prediction matrix dimensions:", paste(dim(preds), collapse=" x "), "\n")
      cat("Row sums range:", paste(range(rowSums(preds)), collapse="-"), "\n")
    }
    
    return(preds)
  } else {
    if(debug) cat("No predictions available, using uniform\n") 
    return(matrix(1/J, nrow=n_ids, ncol=J))
  }
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
        # Use much less aggressive bounds
        extreme_idx <- preds < 0.001 | preds > 0.999
        preds[extreme_idx] <- pmin(pmax(preds[extreme_idx], 0.1), 0.9)
        matrix(preds, nrow=n_ids)
      } else {
        # Add more variation in defaults
        matrix(runif(n_ids, 0.3, 0.7), nrow=n_ids, ncol=1)
      }
    } else {
      # Keep more variation
      vals <- as.numeric(initial_model_for_Y)
      matrix(pmin(pmax(vals, ybound[1]), ybound[2]), nrow=n_ids)
    }
  }, error = function(e) {
    if(debug) cat("Error getting Y predictions:", conditionMessage(e), "\n")
    # More varied defaults
    matrix(runif(n_ids, 0.3, 0.7), nrow=n_ids, ncol=1)
  })
  # Less aggressive final bounds
  pmin(pmax(result, 0.1), 0.9)
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
  if(!is.matrix(g_preds)) {
    g_preds <- matrix(g_preds, ncol=ncol(rules))
  }
  # Ensure dimensions match
  if(nrow(g_preds) != nrow(rules)) {
    g_preds <- t(g_preds)
  }
  
  iptw_weights <- matrix(0, nrow=nrow(g_preds), ncol=n_rules)
  iptw_means <- numeric(n_rules)
  
  for(rule_idx in 1:n_rules) {
    valid_idx <- !is.na(rules[,rule_idx]) & rules[,rule_idx] == 1
    if(any(valid_idx)) {
      marginal_prob <- mean(rules[,rule_idx] == 1, na.rm=TRUE)
      weights <- rep(0, nrow(g_preds))
      rule_probs <- pmin(pmax(g_preds[valid_idx,min(rule_idx,ncol(g_preds))], gbound[1]), gbound[2])
      weights[valid_idx] <- marginal_prob / rule_probs
      weights <- pmin(weights, 10)  # More conservative bound
      
      iptw_weights[,rule_idx] <- weights
      
      # Compute weighted mean
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
    if(is.null(initial_model_for_Y_data) || is.null(tmle_rules)) {
      stop("Missing required data or rules")
    }
    
    n_ids <- length(unique(initial_model_for_Y_data$ID))
    n_rules <- length(tmle_rules)
    J <- length(g_preds_bounded)
    is_binary_case <- J == 1
    
    if(debug) {
      cat("Processing", n_ids, "IDs with", n_rules, "rules\n")
      cat("J =", J, "\n")
    }
    
    # Initialize matrices
    predicted_Y <- matrix(NA, nrow=n_ids, ncol=n_rules)
    predict_Qstar <- matrix(NA, nrow=n_ids, ncol=n_rules)
    epsilon <- rep(0, n_rules)
    
    # Get observed outcomes
    observed_Y <- rep(NA, n_ids)
    if("Y" %in% colnames(initial_model_for_Y_data)) {
      y_values <- initial_model_for_Y_data$Y
      y_matched <- match(unique(initial_model_for_Y_data$ID),
                         initial_model_for_Y_data$ID)
      observed_Y <- y_values[y_matched]
    }
    observed_Y[is.na(observed_Y)] <- 0.5
    
    # Construct g_matrix based on case
    g_matrix <- if(is_binary_case) {
      # Binary case: construct P(A=1)
      probs <- g_preds_bounded[[1]]
      if(is.matrix(probs)) probs <- probs[,1]
      # Bound probabilities, but allow more extreme values
      probs <- pmin(pmax(probs, 0.01), 0.99)
      matrix(probs, ncol=1)
    } else {
      # Multinomial case
      g_mat <- matrix(0, nrow=n_ids, ncol=J)
      for(j in 1:J) {
        probs <- g_preds_bounded[[j]]
        if(is.matrix(probs)) probs <- probs[,1]
        g_mat[,j] <- pmin(pmax(probs, gbound[1]), gbound[2])
      }
      # Normalize probabilities
      g_mat <- t(apply(g_mat, 1, function(row) row/sum(row)))
      g_mat
    }
    
    # Process each rule
    for(i in seq_len(n_rules)) {
      rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
      valid_indices <- match(rule_result$ID, unique(initial_model_for_Y_data$ID))
      valid_indices <- valid_indices[!is.na(valid_indices)]
      
      if(length(valid_indices) > 0) {
        # Get initial predictions for this rule
        initial_preds <- if(is.matrix(initial_model_for_Y_preds)) {
          if(ncol(initial_model_for_Y_preds) >= i) {
            initial_model_for_Y_preds[valid_indices, i]
          } else {
            initial_model_for_Y_preds[valid_indices, 1]
          }
        } else {
          initial_model_for_Y_preds[valid_indices]
        }
        
        # Get rule-specific treatments
        rule_treatments <- as.numeric(rule_result$A0)
        
        # Set initial predictions with bounds
        predicted_Y[valid_indices, i] <- pmin(pmax(initial_preds, ybound[1]), ybound[2])
        predict_Qstar[valid_indices, i] <- predicted_Y[valid_indices, i]
        
        # Process rule indicators
        rule_indicators <- obs.rules[, i]
        valid_rules <- !is.na(rule_indicators) & rule_indicators == 1
        
        # Calculate clever covariate
        H <- numeric(n_ids)
        for(idx in which(valid_rules)) {
          if(rule_treatments[idx] > 0 && rule_treatments[idx] <= J) {
            if(is_binary_case) {
              # Binary case: use logit-based clever covariate
              p1 <- g_matrix[idx, 1]
              p0 <- 1 - p1
              H[idx] <- if(rule_treatments[idx] == 1) 1/p1 else -1/p0
            } else {
              # Multinomial case: use treatment-specific probability
              H[idx] <- 1/g_matrix[idx, rule_treatments[idx]]
            }
          }
        }
        
        # Apply case-specific bounds to clever covariate
        H <- if(is_binary_case) {
          pmin(pmax(H, -4), 4)  # Wider bounds for binary
        } else {
          pmin(pmax(H, -2), 2)  # Standard bounds for multinomial
        }
        
        # Targeting step
        valid_idx <- which(valid_rules)
        if(length(valid_idx) >= 5) {
          # Current predictions with more variation
          current_preds <- predict_Qstar[valid_idx, i]
          bounded_preds <- if(is_binary_case) {
            pmin(pmax(current_preds, 0.001), 0.999)  # Binary bounds
          } else {
            pmin(pmax(current_preds, 0.01), 0.99)    # Multinomial bounds
          }
          
          # Prepare GLM data
          glm_data <- data.frame(
            y = observed_Y[valid_idx],
            h = H[valid_idx],
            offset = qlogis(bounded_preds)
          )
          
          # Looser validity checks to allow more updates
          valid_rows <- complete.cases(glm_data) & 
            is.finite(glm_data$offset) & 
            is.finite(glm_data$h) &
            !is.na(glm_data$y) &
            abs(glm_data$h) < (if(is_binary_case) 10 else 5) # Looser bounds
          
          if(sum(valid_rows) >= 5) {
            glm_data <- glm_data[valid_rows,]
            
            # GLM with more stable settings
            fit <- try(glm(y ~ h + offset(offset),
                           family = binomial(),
                           data = glm_data,
                           control = list(maxit = 100, epsilon = 1e-10)))
            
            if(!inherits(fit, "try-error")) {
              eps <- coef(fit)["h"]
              if(!is.na(eps)) {  # Remove abs(eps) < 2 check
                epsilon[i] <- eps
                logit_pred <- qlogis(predict_Qstar[,i])
                logit_update <- epsilon[i] * H
                valid_update <- !is.na(H) & is.finite(H)
                
                # Less restrictive bounds
                bounds <- if(is_binary_case) c(-5, 5) else c(-3, 3)
                logit_pred[valid_update] <- logit_pred[valid_update] + 
                  pmin(pmax(logit_update[valid_update], bounds[1]), bounds[2])
                
                predict_Qstar[,i] <- pmin(pmax(plogis(logit_pred),
                                               if(is_binary_case) 0.001 else 0.01,
                                               if(is_binary_case) 0.999 else 0.99))
              }
            }
          }
        }
      }
    }
    
    # Calculate IPTW estimates with improved stability
    iptw_means <- sapply(1:n_rules, function(i) {
      rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
      rule_treatments <- as.numeric(rule_result$A0)
      valid_idx <- !is.na(obs.rules[,i]) & obs.rules[,i] == 1
      
      if(any(valid_idx)) {
        marginal_prob <- mean(obs.rules[,i] == 1, na.rm=TRUE)
        weights <- rep(0, n_ids)
        
        for(idx in which(valid_idx)) {
          if(rule_treatments[idx] > 0 && rule_treatments[idx] <= J) {
            if(is_binary_case) {
              p <- if(rule_treatments[idx] == 1) g_matrix[idx, 1] else 1 - g_matrix[idx, 1]
            } else {
              p <- g_matrix[idx, rule_treatments[idx]]
            }
            weights[idx] <- marginal_prob / p
          }
        }
        
        # Bound weights for stability
        weights <- pmin(weights, 10)
        
        # Calculate weighted mean
        valid_outcomes <- valid_idx & !is.na(predict_Qstar[,i])
        if(any(valid_outcomes)) {
          weighted.mean(predict_Qstar[valid_outcomes,i], 
                        weights[valid_outcomes], na.rm=TRUE)
        } else {
          mean(predict_Qstar[,i], na.rm=TRUE)
        }
      } else {
        mean(predict_Qstar[,i], na.rm=TRUE)
      }
    })
    
    if(debug) {
      cat("\nFinal summary:\n")
      cat("Qstar range:", paste(range(predict_Qstar, na.rm=TRUE), collapse="-"), "\n")
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
    if(debug) cat("Error in getTMLELongLSTM:", e$message, "\n")
    
    # Safe defaults preserving dimensions
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