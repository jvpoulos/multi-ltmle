###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

process_time_points <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, g_preds_bin_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size, n_ids,
                                cores = 1, debug = FALSE) {
  
  # Initialize results
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
    
    # Set debug output function
    debug_output <- function(msg, debug) {
      if(debug) {
        cat(msg, "\n")
        flush.console()
      }
    }
    
    # Export debug function first
    parallel::clusterExport(cl, "debug_output", envir=environment())
    
    # Setup debug on each node
    parallel::clusterEvalQ(cl, {
      # Create persistent debug_print function in cluster
      debug_print <- function(msg) {
        if(exists('debug') && debug) {
          debug_output(msg, debug)
        }
      }
    })
    
    # Export all required objects and functions
    objects_to_export <- c(
      "debug",
      "getTMLELongLSTM",
      "process_g_preds",
      "get_c_preds", 
      "get_y_preds",
      "store_results",
      "use_default_values",
      "calculate_iptw",
      "log_iptw_error",
      "log_completion",
      "static_mtp_lstm",
      "dynamic_mtp_lstm",
      "stochastic_mtp_lstm"
    )
    
    parallel::clusterExport(cl, objects_to_export, envir=environment())
    
    # Set required packages
    parallel::clusterEvalQ(cl, {
      library(stats)
      library(Matrix)  # Add any other required packages
    })
    
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
    
    # Export variables with explicit environment
    parallel::clusterExport(cl, names(cluster_env), envir=list2env(cluster_env))
    
    # Process time points in parallel
    results <- parallel::parLapply(cl, time_points, function(t) {
      # Process single time point (same code as before)
      if(debug) debug_print(sprintf("\nProcessing time point %d/%d\n", t, t_end))
      time_start <- Sys.time()
      
      # Initialize with proper matrices
      tmle_contrast <- list(
        "Qstar" = matrix(as.numeric(NA), nrow=n_ids, ncol=n_rules),
        "epsilon" = rep(as.numeric(NA), n_rules),
        "Qstar_gcomp" = matrix(as.numeric(NA), nrow=n_ids, ncol=n_rules),
        "Qstar_iptw" = matrix(as.numeric(NA), nrow=1, ncol=n_rules),
        "Y" = rep(as.numeric(NA), n_ids)
      )
      tmle_contrast_bin <- list(
        "Qstar" = matrix(as.numeric(NA), nrow=n_ids, ncol=n_rules),
        "epsilon" = rep(as.numeric(NA), n_rules), 
        "Qstar_gcomp" = matrix(as.numeric(NA), nrow=n_ids, ncol=n_rules),
        "Qstar_iptw" = matrix(as.numeric(NA), nrow=1, ncol=n_rules),
        "Y" = rep(as.numeric(NA), n_ids)
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
      
      track_initial_data(current_y_preds, debug)
      
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
          current_t = t,
          debug = debug
        )
        
        track_tmle_results(result_multi, "pre-storage-multi", debug)
        
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
          current_t = t,
          debug = debug
        )
        
        track_tmle_results(result_bin, "pre-storage-bin", debug)
        
        # Store results
        store_results(tmle_contrast, result_multi, debug=debug)
        store_results(tmle_contrast_bin, result_bin, debug=debug)
        
      }, error = function(e) {
        if(debug) {
          debug_print(sprintf("\nError processing time point %d: %s\n", t, conditionMessage(e)))
          debug_print("Using default values\n")
        }
        use_default_values(tmle_contrast, ybound)
        use_default_values(tmle_contrast_bin, ybound)
      })
      
      # Calculate IPTW weights and means
      track_stored_results(tmle_contrast, "pre-iptw", debug)
      
      tryCatch({
        current_rules <- obs.rules[[min(t, length(obs.rules))]]
        
        # Multinomial IPTW
        iptw_result_multi <- calculate_iptw(current_g_preds, current_rules, 
                                            tmle_contrast$Qstar,  # Pass Qstar instead of Qstar_gcomp
                                            n_rules, gbound, debug)
        tmle_contrast$Qstar_iptw <- iptw_result_multi
        
        # Binary IPTW
        iptw_result_bin <- calculate_iptw(current_g_preds_bin, current_rules,
                                          tmle_contrast_bin$Qstar,  # Pass Qstar instead of Qstar_gcomp
                                          n_rules, gbound, debug)
        tmle_contrast_bin$Qstar_iptw <- iptw_result_bin
        
        track_stored_results(tmle_contrast, "post-iptw", debug)
        
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
      
      track_initial_data(current_y_preds, debug)
      
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
          current_t = t,
          debug = debug
        )
        
        track_tmle_results(result_multi, "pre-storage-multi", debug)
        
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
          current_t = t,
          debug = debug
        )
        
        track_tmle_results(result_bin, "pre-storage-bin", debug)
        
        # Store results
        store_results(tmle_contrast, result_multi)
        store_results(tmle_contrast_bin, result_bin)
        
        process_time_points_tracking(tmle_contrast, tmle_contrast_bin, t, debug=debug)
        
      }, error = function(e) {
        if(debug) {
          cat(sprintf("\nError processing time point %d: %s\n", t, conditionMessage(e)))
          cat("Using default values\n")
        }
        use_default_values(tmle_contrast, ybound)
        use_default_values(tmle_contrast_bin, ybound)
      })
      
      # Calculate IPTW weights and means
      track_stored_results(tmle_contrast, "pre-iptw", debug)
      tryCatch({
        current_rules <- obs.rules[[min(t, length(obs.rules))]]
        
        # Multinomial IPTW
        iptw_result_multi <- calculate_iptw(current_g_preds, current_rules, 
                                            tmle_contrast$Qstar, 
                                            n_rules, gbound, debug)
        tmle_contrast$Qstar_iptw <- iptw_result_multi
        
        # Binary IPTW
        iptw_result_bin <- calculate_iptw(current_g_preds_bin, current_rules,
                                          tmle_contrast_bin$Qstar,
                                          n_rules, gbound, debug)
        tmle_contrast_bin$Qstar_iptw <- iptw_result_bin
        
        process_time_points_tracking(tmle_contrast, tmle_contrast_bin, t, debug=debug)
        
        track_stored_results(tmle_contrast, "post-iptw", debug)
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
  
  if(debug) {
    cat("\nFinal time point processing summary:\n")
    for(t in 1:t_end) {
      cat("\nTime point", t, "summary:\n")
      cat("TMLE estimates:\n")
      print(colMeans(tmle_contrasts[[t]]$Qstar, na.rm=TRUE))
      cat("IPTW estimates:\n")
      print(tmle_contrasts[[t]]$Qstar_iptw)
    }
  }
  
  return(list(
    "multinomial" = tmle_contrasts,
    "binary" = tmle_contrasts_bin
  ))
}

process_time_points_tracking <- function(tmle_contrast, tmle_contrast_bin, t, debug=FALSE) {
  if(debug) {
    cat("\nTracking results at time", t, ":")
    cat("\nMultinomial Results:")
    if(!is.null(tmle_contrast)) {
      cat("\nQstar summary:\n")
      print(summary(as.vector(tmle_contrast$Qstar)))
      cat("\nQstar range:", paste(range(tmle_contrast$Qstar, na.rm=TRUE), collapse="-"))
      cat("\nQstar_iptw:\n") 
      print(tmle_contrast$Qstar_iptw)
    }
    cat("\nBinary Results:")
    if(!is.null(tmle_contrast_bin)) {
      cat("\nQstar summary:\n")
      print(summary(as.vector(tmle_contrast_bin$Qstar)))
      cat("\nQstar range:", paste(range(tmle_contrast_bin$Qstar, na.rm=TRUE), collapse="-"))
      cat("\nQstar_iptw:\n")
      print(tmle_contrast_bin$Qstar_iptw)
    }
  }
}

# Helper functions
process_g_preds <- function(preds_processed, t, n_ids, J, gbound, debug) {
  if(!is.null(preds_processed) && t <= length(preds_processed)) {
    preds <- preds_processed[[t]]
    
    if(is.null(preds)) {
      if(debug) cat("No predictions for time", t, "using uniform\n")
      return(matrix(1/J, nrow=n_ids, ncol=J))
    }
    
    # Ensure matrix format with J columns
    if(!is.matrix(preds)) {
      if(debug) cat("Converting predictions to matrix\n")
      preds <- matrix(preds, nrow=n_ids, ncol=J)
    }
    
    # Normalize probabilities
    if(debug) cat("Normalizing probabilities\n")
    preds <- t(apply(preds, 1, function(row) {
      if(any(!is.finite(row))) return(rep(1/J, J))
      bounded <- pmin(pmax(row, gbound[1]), gbound[2]) 
      # Add minimum floor to prevent too small values
      bounded <- pmax(bounded, 1e-4)
      bounded / sum(bounded)
    }))
    
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

# In get_y_preds function, adjust the scaling factor
get_y_preds <- function(initial_model_for_Y, t, n_ids, ybound, debug) {
  result <- tryCatch({
    if(is.list(initial_model_for_Y)) {
      if(!is.null(initial_model_for_Y$preds)) {
        preds <- initial_model_for_Y$preds
        if(is.matrix(preds)) {
          col_idx <- min(t, ncol(preds))
          preds <- preds[,col_idx]
        }
        
        # Adjust scaling factor to avoid pushing values towards bounds
        scale_factor <- mean(preds, na.rm=TRUE)
        preds <- preds * min(scale_factor, 0.5)  # Cap scaling at 0.5 to avoid extreme values
        
        # Add small random variation when needed
        if(var(preds) < 1e-6) {
          preds <- preds + rnorm(length(preds), 0, ybound[1] * 0.1)  # Smaller variation
        }
        
        # Tighter bounds
        matrix(pmin(pmax(preds, ybound[1]), ybound[2]), nrow=n_ids)
      } else {
        # Lower default range
        matrix(runif(n_ids, ybound[1], 0.5), nrow=n_ids)  # Use 0.5 instead of 0.025
      }
    } else {
      base_val <- as.numeric(initial_model_for_Y) * min(mean(initial_model_for_Y, na.rm=TRUE), 0.5)
      matrix(pmin(pmax(base_val + rnorm(n_ids, 0, ybound[1] * 0.1), ybound[1]), 0.5), nrow=n_ids)
    }
  }, error = function(e) {
    if(debug) cat("Error getting Y predictions:", conditionMessage(e), "\n")
    matrix(runif(n_ids, ybound[1], 0.5), nrow=n_ids)  # Use 0.5 instead of 0.025
  })
  
  if(debug) {
    cat("\nY predictions summary:\n")
    print(summary(as.vector(result)))
    cat("Variance:", var(as.vector(result)), "\n")
  }
  return(result)
}

# Track initial data loading
track_initial_data <- function(initial_model_for_Y_preds, debug=FALSE) {
  if(debug) {
    cat("\n=== Initial Data Loading ===")
    cat("\nInitial predictions summary:")
    print(summary(as.vector(initial_model_for_Y_preds)))
    if(is.list(initial_model_for_Y_preds)) {
      cat("\npreds component:")
      print(summary(as.vector(initial_model_for_Y_preds$preds)))
    }
  }
}

# Track TMLE results before storage
track_tmle_results <- function(result, stage="pre-storage", debug=FALSE) {
  if(debug) {
    cat(sprintf("\n=== TMLE Results (%s) ===", stage))
    if(!is.null(result)) {
      cat("\nQstar summary:")
      print(summary(as.vector(result$Qstar)))
      cat("\nQstar_gcomp summary:")
      print(summary(as.vector(result$Qstar_gcomp)))
      cat("\nY summary:")
      print(summary(as.vector(result$Y)))
    }
  }
}

# Track after storage
track_stored_results <- function(tmle_contrast, stage="post-storage", debug=FALSE) {
  if(debug) {
    cat(sprintf("\n=== Stored Results (%s) ===", stage))
    if(!is.null(tmle_contrast)) {
      cat("\nQstar summary:")
      print(summary(as.vector(tmle_contrast$Qstar)))
      cat("\nQstar_gcomp summary:")
      print(summary(as.vector(tmle_contrast$Qstar_gcomp)))
      cat("\nY summary:")
      print(summary(as.vector(tmle_contrast$Y)))
    }
  }
}

store_results <- function(tmle_contrast, result, debug=FALSE) {
  for(comp in c("Qstar", "Qstar_gcomp", "Qstar_iptw", "Y", "epsilon")) {
    if(!is.null(result[[comp]])) {
      if(is.null(tmle_contrast[[comp]])) {
        # Initialize with result
        tmle_contrast[[comp]] <- result[[comp]]
      } else {
        # Only update invalid values
        invalid_mask <- is.na(tmle_contrast[[comp]]) | 
          !is.finite(tmle_contrast[[comp]]) |
          tmle_contrast[[comp]] < ybound[1] | 
          tmle_contrast[[comp]] > ybound[2]
        
        tmle_contrast[[comp]][invalid_mask] <- result[[comp]][invalid_mask]
      }
    }
  }
}

use_default_values <- function(tmle_contrast, ybound) {
  # Use mean value between bounds as default
  default_val <- mean(ybound)
  tmle_contrast$Qstar[] <- default_val
  tmle_contrast$Qstar_gcomp[] <- default_val
  tmle_contrast$Y[] <- default_val
}

calculate_iptw <- function(g_preds, rules, predict_Qstar, n_rules, gbound, debug) {
  # Initialize means with original predictions
  iptw_means <- numeric(n_rules)
  
  for(rule_idx in 1:n_rules) {
    valid_idx <- !is.na(rules[,rule_idx]) & rules[,rule_idx] == 1
    
    if(any(valid_idx)) {
      outcomes <- predict_Qstar[,rule_idx]
      rule_probs <- g_preds[valid_idx, min(rule_idx, ncol(g_preds))]
      rule_probs <- pmin(pmax(rule_probs, gbound[1]), gbound[2])
      marginal_prob <- mean(valid_idx, na.rm=TRUE)
      
      weights <- rep(0, nrow(g_preds))
      weights[valid_idx] <- marginal_prob / rule_probs
      
      # Trim extreme weights
      max_weight <- quantile(weights[valid_idx], 0.95, na.rm=TRUE) 
      weights <- pmin(weights, max_weight)
      weights[valid_idx] <- weights[valid_idx] / sum(weights[valid_idx])
      
      # Calculate weighted mean
      iptw_means[rule_idx] <- weighted.mean(outcomes[valid_idx], 
                                            weights[valid_idx], 
                                            na.rm=TRUE)
    } else {
      iptw_means[rule_idx] <- mean(predict_Qstar[,rule_idx], na.rm=TRUE)
    }
  }
  
  matrix(iptw_means, nrow=1)
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
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size = 7, current_t,
                            debug = FALSE) {
  
  if(debug) {
    cat("\nInitial state:")
    cat("\nInput predictions summary:")
    print(summary(as.vector(initial_model_for_Y_preds)))
    cat("\nInput Qstar dimensions:", paste(dim(initial_model_for_Y_preds), collapse=" x"), "\n")
  }
  
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
    predicted_Y <- matrix(initial_model_for_Y_preds, nrow=n_ids, ncol=n_rules) 
    predict_Qstar <- predicted_Y
    epsilon <- rep(0, n_rules)
    
    observed_Y <- rep(NA, n_ids) 
    if("Y" %in% colnames(initial_model_for_Y_data)) {
      # Get unique IDs in order
      unique_ids <- unique(initial_model_for_Y_data$ID)
      
      # Use passed time point instead of t[1]
      if(debug) {
        cat("\nExtracting Y values for time", current_t)
        cat("\nColumns:", paste(colnames(initial_model_for_Y_data), collapse=", "))
        cat("\nUnique Y values before processing:", paste(unique(initial_model_for_Y_data$Y), collapse=", "))
      }
      
      # For each ID, get Y value at current time
      for(i in seq_along(unique_ids)) {
        id <- unique_ids[i]
        id_rows <- which(initial_model_for_Y_data$ID == id & 
                           initial_model_for_Y_data$t == current_t)
        if(length(id_rows) > 0) {
          y_val <- initial_model_for_Y_data$Y[id_rows[1]]
          # Only treat -1 as missing, preserve actual 0s and 1s
          if(y_val != -1) {
            observed_Y[i] <- y_val
          } else {
            # If -1, try to get Y value from previous time point
            prev_row <- which(initial_model_for_Y_data$ID == id & 
                                initial_model_for_Y_data$t == (current_t - 1))
            if(length(prev_row) > 0) {
              prev_y <- initial_model_for_Y_data$Y[prev_row[1]]
              if(prev_y != -1) observed_Y[i] <- prev_y
            }
          }
        }
      }
      
      if(debug) {
        cat("\nAfter matching at time", current_t, ":")
        cat("\nRange of observed_Y:", paste(range(observed_Y, na.rm=TRUE), collapse="-"))
        cat("\nMean of observed_Y:", mean(observed_Y, na.rm=TRUE))
        cat("\nNumber of NAs:", sum(is.na(observed_Y)))
        cat("\nDistribution of Y values:\n")
        print(table(observed_Y, useNA="ifany"))
      }
    }
    
    # Only fill NAs with 0.5 if absolutely necessary
    na_count <- sum(is.na(observed_Y))
    if(na_count > 0) {
      if(debug) cat("\nFilling", na_count, "NAs with predicted values\n")
      # Use predicted values for NAs instead of 0.5
      na_indices <- which(is.na(observed_Y))
      observed_Y[na_indices] <- initial_model_for_Y_preds[na_indices]
    }
    
    # Construct g_matrix based on case
    g_matrix <- if(is_binary_case) {
      # Binary case: construct P(A=1)
      probs <- g_preds_bounded[[1]]
      if(is.matrix(probs)) probs <- probs[,1]
      # Bound probabilities, but allow more extreme values
      probs <- pmin(pmax(probs, gbound[1]), gbound[2])
      matrix(probs, ncol=1)
    } else {
      # Multinomial case - add renormalization per timepoint
      g_mat <- matrix(0, nrow=n_ids, ncol=J)
      for(j in 1:J) {
        probs <- g_preds_bounded[[j]]
        if(is.matrix(probs)) probs <- probs[,1]
        g_mat[,j] <- pmin(pmax(probs, gbound[1]), gbound[2])
      }
      # Renormalize at each timepoint
      g_mat <- t(apply(g_mat, 1, function(row) {
        bounded <- pmin(pmax(row, gbound[1]), gbound[2])
        bounded / sum(bounded) 
      }))
      g_mat
    }
    
    # Process each rule
    for(i in seq_len(n_rules)) {
      rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
      valid_indices <- match(rule_result$ID, unique(initial_model_for_Y_data$ID))
      valid_indices <- valid_indices[!is.na(valid_indices)]
      
      if(length(valid_indices) > 0) {
        # Get initial predictions for this rule
        initial_preds <- initial_model_for_Y_preds[valid_indices]
        
        # Get rule-specific treatments
        rule_treatments <- as.numeric(rule_result$A0)
        
        # Set initial predictions with bounds
        predicted_Y[valid_indices, i] <- pmin(pmax(initial_preds, 0.1), 0.9)  # Use wider bounds
        predict_Qstar[valid_indices, i] <- predicted_Y[valid_indices, i]
        
        # Process rule indicators
        rule_indicators <- obs.rules[, i]
        valid_rules <- !is.na(rule_indicators) & rule_indicators == 1 & 
          !is.na(rule_treatments) & rule_treatments > 0 & rule_treatments <= J
        
        # Calculate clever covariate for valid rules only
        H <- numeric(n_ids)
        H[is.na(H)] <- 0  # Replace NaNs with 0
        # Inside rule processing loop
        if(any(valid_rules)) {
          # Calculate normalized weights first
          w <- rep(0, n_ids)
          for(idx in which(valid_rules)) {
            if(is_binary_case) {
              p1 <- g_matrix[idx, 1]
              p0 <- 1 - p1
              w[idx] <- if(rule_treatments[idx] == 1) 1/p1 else 1/p0
            } else {
              treatment_prob <- pmax(g_matrix[idx, rule_treatments[idx]], 1e-6)  # Add a small floor
              w[idx] <- 1/treatment_prob
            }
          }
          
          # Trim extreme weights
          w_valid <- w[valid_rules]
          w_quant <- quantile(w_valid[w_valid > 0], c(gbound[1], gbound[2]), na.rm=TRUE)
          w[w < w_quant[1]] <- w_quant[1]
          w[w > w_quant[2]] <- w_quant[2]
          
          # Normalize weights
          w[valid_rules] <- w[valid_rules] / sum(w[valid_rules])
          
          # Use normalized weights for H
          H <- w
          
          # Targeting step with weighted GLM
          valid_idx <- which(valid_rules)
          
          # Improve GLM stability when estimating epsilon
          # Inside rule processing loop where we do targeting
          if(length(valid_idx) >= 5) {
            # Get current predictions 
            current_preds <- predict_Qstar[valid_idx, i]
            
            # Create GLM data
            glm_data <- data.frame(
              y = observed_Y[valid_idx],
              offset = qlogis(pmin(pmax(current_preds, 0.001), 0.999))
            )
            
            # Add clever covariate
            H_cent <- H[valid_idx] - mean(H[valid_idx], na.rm=TRUE)
            if(sd(H_cent, na.rm=TRUE) > 0) {
              glm_data$h <- H_cent/sd(H_cent, na.rm=TRUE)
            } else {
              glm_data$h <- H_cent
            }
            
            # Add weights
            glm_data$w <- w[valid_idx]
            
            # Fit GLM
            fit <- tryCatch({
              glm(y ~ h + offset(offset),
                  family = quasibinomial(link = "logit"),
                  weights = w,
                  data = glm_data,
                  control = glm.control(maxit = 50, epsilon = 1e-6)
              )
            }, error = function(e) NULL)
            
            if(!is.null(fit)) {
              eps <- coef(fit)["h"]
              if(!is.na(eps)) {
                epsilon[i] <- eps
                
                # Calculate updates on logit scale
                current_logit <- qlogis(pmin(pmax(predict_Qstar[,i], 0.001), 0.999))
                H_update <- H - mean(H, na.rm=TRUE)
                if(sd(H_update, na.rm=TRUE) > 0) {
                  H_update <- H_update/sd(H_update, na.rm=TRUE)
                }
                
                # Update predictions
                updated_logit <- current_logit + eps * H_update
                predict_Qstar[,i] <- pmin(pmax(plogis(updated_logit), ybound[1]), ybound[2])
              }
            }
          }
        }
      }
    }
    
    if(debug) {
      cat("\nFinal results for rule", i, ":")
      cat("\nOriginal mean:", mean(initial_model_for_Y_preds, na.rm=TRUE))
      cat("\nUpdated mean:", mean(predict_Qstar[,i], na.rm=TRUE))
      cat("\nObserved Y mean:", mean(observed_Y[valid_rules], na.rm=TRUE))
      cat("\nProportion following rule:", mean(valid_rules, na.rm=TRUE))
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
      cat("\nTMLE Results Detail:\n")
      cat("Predicted Y range:", paste(range(predicted_Y, na.rm=TRUE), collapse="-"), "\n")
      cat("Qstar range:", paste(range(predict_Qstar, na.rm=TRUE), collapse="-"), "\n")
      cat("Qstar dimensions:", paste(dim(predict_Qstar), collapse=" x"), "\n")
      cat("Qstar means by rule:\n")  
      cat("Observed Y mean:", mean(observed_Y, na.rm=TRUE), "\n")
      cat("IPTW means:", paste(iptw_means, collapse=", "), "\n")
      cat("Rule-specific means:\n")
      for(i in 1:n_rules) {
        cat("Rule", i, "mean:", mean(predict_Qstar[,i], na.rm=TRUE), "\n")
        cat("Rule", i, "IPTW mean:", iptw_means[i], "\n")
      }
    }
    
    return(list(
      "Qstar" = predict_Qstar,
      "epsilon" = epsilon,
      "Qstar_gcomp" = predicted_Y,
      "Qstar_iptw" = matrix(iptw_means, nrow=1),
      "Y" = observed_Y
    ))
  }, error = function(e) {
    if(debug) {
      cat("Error in getTMLELongLSTM:", e$message, "\n")
      cat("Traceback:\n")
      print(e)
    }
    # Return safe defaults
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
  
  if(debug) {
    cat("\nFinal results summary:")
    if(!is.null(final_result$Qstar)) {
      cat("\nQstar range:", paste(range(final_result$Qstar, na.rm=TRUE), collapse="-"))
      cat("\nQstar means by rule:")
      print(colMeans(final_result$Qstar, na.rm=TRUE))
    }
    if(!is.null(final_result$Qstar_iptw)) {
      cat("\nIPTW means:", paste(final_result$Qstar_iptw, collapse=", "))
    }
    if(!is.null(final_result$Qstar_gcomp)) {
      cat("\nGcomp range:", paste(range(final_result$Qstar_gcomp, na.rm=TRUE), collapse="-"))
    }
  }
  
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