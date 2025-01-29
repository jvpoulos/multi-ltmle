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
      
      # Initialize results for this time point 
      tmle_contrast <- list(
        "Qstar" = matrix(NA, nrow = n_ids, ncol = n_rules),
        "epsilon" = rep(NA, n_rules),
        "Qstar_gcomp" = matrix(NA, nrow = n_ids, ncol = n_rules), 
        "Qstar_iptw" = matrix(NA, nrow = 1, ncol = n_rules),
        "Y" = rep(NA, n_ids)
      )
      tmle_contrast_bin <- list(
        "Qstar" = matrix(NA, nrow = n_ids, ncol = n_rules),
        "epsilon" = rep(NA, n_rules),
        "Qstar_gcomp" = matrix(NA, nrow = n_ids, ncol = n_rules),
        "Qstar_iptw" = matrix(NA, nrow = 1, ncol = n_rules), 
        "Y" = rep(NA, n_ids)
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
      
      if(debug) {
        cat("\nPre-Assignment Dimensions:")
        cat("\nresult_multi$Qstar:", paste(dim(result_multi$Qstar), collapse=" x "))
        cat("\nresult_multi$Qstar_gcomp:", paste(dim(result_multi$Qstar_gcomp), collapse=" x "))
      }
      
      # Directly assign results
      tmle_contrast <- result_multi
      tmle_contrast_bin <- result_bin
      
      if(debug) {
        cat("\nPost-Assignment Dimensions:")
        cat("\ntmle_contrast$Qstar:", paste(dim(tmle_contrast$Qstar), collapse=" x "))
        cat("\ntmle_contrast$Qstar_gcomp:", paste(dim(tmle_contrast$Qstar_gcomp), collapse=" x "))
      }
      
      if(debug) {
        cat("\nStored Results:")
        cat("\nMultinomial Qstar range:", paste(range(tmle_contrast$Qstar, na.rm=TRUE), collapse="-"))
        cat("\nBinary Qstar range:", paste(range(tmle_contrast_bin$Qstar, na.rm=TRUE), collapse="-"))
      }
      
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
      
      if(debug) {
        cat("\nPreparing to get Y predictions")
        cat("\ninitial_model_for_Y type:", class(initial_model_for_Y))
        cat("\nTime point:", t)
        cat("\nExpected n_ids:", n_ids)
      }
      
      current_c_preds <- get_c_preds(C_preds_processed, t, n_ids, gbound)
      current_y_preds <- get_y_preds(initial_model_for_Y, t, n_ids, ybound, debug)
      
      track_initial_data(current_y_preds, debug)
      
      # Process both cases
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
      
      # Directly assign results
      tmle_contrast <- result_multi
      tmle_contrast_bin <- result_bin
      
      if(debug) {
        cat("\nStored Results:")
        cat("\nMultinomial Qstar range:", paste(range(tmle_contrast$Qstar, na.rm=TRUE), collapse="-"))
        cat("\nBinary Qstar range:", paste(range(tmle_contrast_bin$Qstar, na.rm=TRUE), collapse="-"))
      }
      
      process_time_points_tracking(tmle_contrast, tmle_contrast_bin, t, debug=debug)
      
      
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
    # Deep copy of components
    tmle_contrasts[[t]] <- list(
      "Qstar" = results[[t]]$multinomial$Qstar,
      "epsilon" = results[[t]]$multinomial$epsilon,
      "Qstar_gcomp" = results[[t]]$multinomial$Qstar_gcomp,
      "Qstar_iptw" = results[[t]]$multinomial$Qstar_iptw,
      "Y" = results[[t]]$multinomial$Y
    )
    tmle_contrasts_bin[[t]] <- list(
      "Qstar" = results[[t]]$binary$Qstar,
      "epsilon" = results[[t]]$binary$epsilon, 
      "Qstar_gcomp" = results[[t]]$binary$Qstar_gcomp,
      "Qstar_iptw" = results[[t]]$binary$Qstar_iptw,
      "Y" = results[[t]]$binary$Y
    )
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
  
  if(debug) {
    cat("\nTMLE Result Dimensions:")
    cat("\npredict_Qstar:", paste(dim(predict_Qstar), collapse=" x "))
    cat("\npredicted_Y:", paste(dim(predicted_Y), collapse=" x "))
    cat("\nQstar_iptw:", paste(dim(matrix(iptw_means, nrow=1)), collapse=" x "))
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

get_y_preds <- function(initial_model_for_Y, t, n_ids, ybound, debug) {
  if(debug) {
    cat("\nEntering get_y_preds")
    cat("\nReceived initial_model_for_Y type:", class(initial_model_for_Y))
    cat("\nExpected n_ids:", n_ids)  
    if(is.vector(initial_model_for_Y)) {
      cat("\nReceived vector length:", length(initial_model_for_Y))
    } else if(is.matrix(initial_model_for_Y)) {
      cat("\nReceived matrix dims:", paste(dim(initial_model_for_Y), collapse=" x "))
    } else if(is.list(initial_model_for_Y)) {
      cat("\nReceived list with components:", paste(names(initial_model_for_Y), collapse=", "))
    }
  }
  
  result <- tryCatch({
    if(is.null(initial_model_for_Y)) {
      if(debug) cat("\nNULL initial_model_for_Y, returning default matrix")
      return(matrix(0.5, nrow=n_ids, ncol=1))
    }
    
    if(is.list(initial_model_for_Y)) {
      if(!is.null(initial_model_for_Y$preds)) {
        preds <- initial_model_for_Y$preds
        if(is.matrix(preds)) {
          if(debug) cat("\nProcessing matrix predictions with dims:", paste(dim(preds), collapse=" x "))
          col_idx <- min(t, ncol(preds))
          preds <- preds[,col_idx, drop=TRUE]
        }
        if(length(preds) != n_ids) {
          if(debug) cat("\nLength mismatch: preds=", length(preds), " n_ids=", n_ids)
          # Attempt to expand or truncate to match n_ids
          if(length(preds) > n_ids) {
            preds <- preds[1:n_ids]
          } else {
            preds <- rep(preds, length.out=n_ids)
          }
        }
        result_matrix <- matrix(preds, nrow=n_ids)
      } else {
        if(debug) cat("\nNo preds in list, using default matrix")
        result_matrix <- matrix(0.5, nrow=n_ids, ncol=1)
      }
    } else if(is.vector(initial_model_for_Y)) {
      if(debug) cat("\nProcessing vector input of length:", length(initial_model_for_Y))
      if(length(initial_model_for_Y) != n_ids) {
        if(debug) cat("\nLength mismatch: vector=", length(initial_model_for_Y), " n_ids=", n_ids)
        # Attempt to expand or truncate to match n_ids
        initial_model_for_Y <- rep(initial_model_for_Y, length.out=n_ids)
      }
      result_matrix <- matrix(initial_model_for_Y, nrow=n_ids)
    } else if(is.matrix(initial_model_for_Y)) {
      if(debug) cat("\nProcessing matrix input with dims:", paste(dim(initial_model_for_Y), collapse=" x "))
      if(nrow(initial_model_for_Y) != n_ids) {
        if(debug) cat("\nRow count mismatch: matrix=", nrow(initial_model_for_Y), " n_ids=", n_ids)
        # Attempt to expand or truncate to match n_ids
        result_matrix <- matrix(initial_model_for_Y[1:min(nrow(initial_model_for_Y), n_ids),], nrow=n_ids, ncol=ncol(initial_model_for_Y))
      } else {
        result_matrix <- initial_model_for_Y
      }
    } else {
      if(debug) cat("\nUnhandled input type, using default matrix")
      result_matrix <- matrix(0.5, nrow=n_ids, ncol=1)
    }
    
    # Ensure proper dimensions
    if(ncol(result_matrix) > 1) {
      if(debug) cat("\nTaking first column of multi-column matrix")
      result_matrix <- result_matrix[,1,drop=FALSE]
    }
    
    # Bound values
    result_matrix <- pmin(pmax(result_matrix, ybound[1]), ybound[2])
    
    if(debug) {
      cat("\nFinal matrix dimensions:", paste(dim(result_matrix), collapse=" x "))
      cat("\nValue summary:", paste(range(result_matrix), collapse="-"))
    }
    
    return(result_matrix)
    
  }, error = function(e) {
    if(debug) {
      cat("\nError in get_y_preds:", conditionMessage(e))
      cat("\nReturning default matrix")
    }
    matrix(0.5, nrow=n_ids, ncol=1)
  })
  
  if(debug) {
    cat("\nget_y_preds returning matrix with dims:", paste(dim(result), collapse=" x "), "\n")
    cat("Y predictions summary:\n")
    print(summary(as.vector(result)))
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

    # Initialize matrices with survival probabilities 
    predicted_Y <- matrix(initial_model_for_Y_preds, nrow=n_ids, ncol=n_rules) 
    predict_Qstar <- predicted_Y
    epsilon <- rep(0, n_rules)
    
    if(debug) {
      cat("\nPredicted Y (survival probabilities) after initialization:")
      print(summary(as.vector(predicted_Y)))
    }
    
    # In getTMLELongLSTM, modify the Y extraction:
    
    observed_Y <- rep(NA, n_ids)
    if("Y" %in% colnames(initial_model_for_Y_data)) {
      unique_ids <- unique(initial_model_for_Y_data$ID)
      
      if(debug) {
        cat("\nExtracting Y values for time", current_t)
        cat("\nUnique Y values before processing:", 
            paste(unique(initial_model_for_Y_data$Y), collapse=", "))
      }
      
      # Extract and convert Y values to survival probabilities
      for(i in seq_along(unique_ids)) {
        id <- unique_ids[i]
        id_rows <- which(initial_model_for_Y_data$ID == id & 
                           initial_model_for_Y_data$t == current_t)
        
        if(length(id_rows) > 0) {
          y_val <- initial_model_for_Y_data$Y[id_rows[1]]
          if(y_val != -1) {
            # Convert from event indicator (1=event) to survival probability (1=survived)
            observed_Y[i] <- 1 - y_val  # 1 becomes 0 (event), 0 becomes 1 (survived)
          } else {
            # Look back up to 3 time points for a valid Y value
            for(back_t in 1:3) {
              prev_row <- which(initial_model_for_Y_data$ID == id & 
                                  initial_model_for_Y_data$t == (current_t - back_t))
              if(length(prev_row) > 0) {
                prev_y <- initial_model_for_Y_data$Y[prev_row[1]]
                if(prev_y != -1) {
                  observed_Y[i] <- 1 - prev_y  # Convert to survival probability
                  break
                }
              }
            }
          }
        }
      }
      
      if(debug) {
        cat("\nAfter matching at time", current_t, ":")
        cat("\nRange of observed_Y (survival probabilities):", 
            paste(range(observed_Y, na.rm=TRUE), collapse="-"))
        cat("\nMean survival probability:", mean(observed_Y, na.rm=TRUE))
        cat("\nNumber of NAs:", sum(is.na(observed_Y)))
      }
    }
    
    # Fill NAs with predicted values (already survival probabilities)
    na_count <- sum(is.na(observed_Y))
    if(na_count > 0) {
      if(debug) cat("\nFilling", na_count, "NAs with predicted values\n")
      na_indices <- which(is.na(observed_Y))
      observed_Y[na_indices] <- initial_model_for_Y_preds[na_indices]
    }
    
    if(debug) {
      cat("\nFinal observed Y summary (survival probabilities):")
      print(summary(observed_Y))
    }
    
    # Construct g_matrix with improved bounds
    g_matrix <- if(is_binary_case) {
      probs <- g_preds_bounded[[1]]
      if(is.matrix(probs)) probs <- probs[,1]
      probs <- pmin(pmax(probs, gbound[1]), gbound[2])
      matrix(probs, ncol=1)
    } else {
      g_mat <- matrix(0, nrow=n_ids, ncol=J)
      for(j in 1:J) {
        probs <- g_preds_bounded[[j]]
        if(is.matrix(probs)) probs <- probs[,1]
        g_mat[,j] <- pmin(pmax(probs, gbound[1]), gbound[2])
      }
      # Improved normalization with minimum probability floor
      g_mat <- t(apply(g_mat, 1, function(row) {
        bounded <- pmin(pmax(row, gbound[1]), gbound[2])
        bounded <- pmax(bounded, 1e-4)  # Minimum floor
        bounded / sum(bounded)
      }))
      g_mat
    }
    
    # Process each rule with improved stability
    for(i in seq_len(n_rules)) {
      rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
      valid_indices <- match(rule_result$ID, unique(initial_model_for_Y_data$ID))
      valid_indices <- valid_indices[!is.na(valid_indices)]
      
      if(length(valid_indices) > 0) {
        initial_preds <- initial_model_for_Y_preds[valid_indices]
        rule_treatments <- as.numeric(rule_result$A0)
        
        # Use wider bounds for initial predictions
        predicted_Y[valid_indices, i] <- pmin(pmax(initial_preds, ybound[1]), ybound[2])
        predict_Qstar[valid_indices, i] <- predicted_Y[valid_indices, i]
        
        if(debug) {
          cat("\nPredictions after bounding transform for rule", i, ":")
          cat("\nMean prediction:", mean(predict_Qstar[valid_indices, i], na.rm=TRUE))
          cat("\nRange:", paste(range(predict_Qstar[valid_indices, i], na.rm=TRUE), collapse=" - "))
        }
        
        rule_indicators <- obs.rules[, i]
        valid_rules <- !is.na(rule_indicators) & rule_indicators == 1 & 
          !is.na(rule_treatments) & rule_treatments > 0 & rule_treatments <= J
        
        # Initialize H and w vectors with zeros
        H <- rep(0, n_ids)
        w <- rep(0, n_ids)
        
        # Calculate clever covariates for valid rules
        n_valid <- sum(valid_rules)
        if(n_valid >= 5) {
          for(idx in which(valid_rules)) {
            if(idx <= length(rule_treatments)) {
              treatment <- rule_treatments[idx]
              if(!is.na(treatment) && treatment > 0 && treatment <= J) {
                if(is_binary_case) {
                  # Binary case
                  p1 <- pmin(pmax(g_matrix[idx, 1], 1e-6), 1-1e-6)
                  p0 <- 1 - p1
                  
                  if(treatment == 1) {
                    w[idx] <- 1 / p1
                    H[idx] <- 1 / p1
                  } else {
                    w[idx] <- 1 / p0
                    H[idx] <- -1 / p0
                  }
                } else {
                  # Multinomial case
                  p_treat <- pmin(pmax(g_matrix[idx, treatment], 1e-6), 1-1e-6)
                  w[idx] <- 1 / p_treat
                  H[idx] <- 1 / p_treat
                  
                  # Calculate contrast term as average of other probabilities
                  contrast_sum <- 0
                  n_others <- 0
                  for(j in 1:J) {
                    if(j != treatment) {
                      p_j <- pmin(pmax(g_matrix[idx, j], 1e-6), 1-1e-6)
                      contrast_sum <- contrast_sum + (1 / p_j)
                      n_others <- n_others + 1
                    }
                  }
                  H[idx] <- H[idx] - (contrast_sum / n_others)
                }
              }
            }
          }
          
          # Center and scale H values, with additional checks
          H_valid <- H[valid_rules]
          h_sum <- sum(H_valid)
          if(h_sum != 0 && !is.na(h_sum)) {  # Only proceed if we have non-zero, non-NA values
            H_mean <- mean(H_valid, na.rm=TRUE)
            H_sd <- sd(H_valid, na.rm=TRUE)
            
            if(!is.na(H_sd) && H_sd > 1e-8) {  # Use small threshold instead of exactly 0
              H <- (H - H_mean) / H_sd
            } else {
              # If variance too small, just center
              H <- H - H_mean
            }
          } else {
            # If all zeros/NAs, create small random variation
            H[valid_rules] <- rnorm(n_valid, mean=0, sd=0.1)
          }
        }
        
        if(debug) {
          cat("\nClever covariate calculation:")
          cat("\nNumber of valid rules:", sum(valid_rules))
          cat("\nRange of H values:", paste(range(H, na.rm=TRUE), collapse=" - "))
          cat("\nMean of H:", mean(H, na.rm=TRUE))
          cat("\nSD of H:", sd(H, na.rm=TRUE))
        }
        
        # Targeting step with improved stability
        valid_idx <- which(valid_rules)
        if(length(valid_idx) >= 5) {
          current_preds <- predict_Qstar[valid_idx, i]
          current_preds <- pmin(pmax(current_preds, 0.001), 0.999)
          
          # Create GLM data with additional check for infinite/NA values
          offset_vals <- qlogis(current_preds)
          valid_data <- !is.infinite(offset_vals) & !is.na(offset_vals) & 
            !is.infinite(H[valid_idx]) & !is.na(H[valid_idx])
          
          if(sum(valid_data) >= 5) {
            glm_data <- data.frame(
              y = observed_Y[valid_idx][valid_data],
              offset = offset_vals[valid_data],
              h = H[valid_idx][valid_data],
              w = w[valid_idx][valid_data]
            )
            
            # Fit GLM with improved stability controls
            fit <- tryCatch({
              glm(y ~ h + offset(offset),
                  family = quasibinomial(link = "logit"),
                  weights = w,
                  data = glm_data,
                  control = glm.control(maxit = 100, epsilon = 1e-8)
              )
            }, error = function(e) NULL)
            
            if(!is.null(fit)) {
              coefs <- coef(fit)
              eps <- if("h" %in% names(coefs)) {
                val <- coefs["h"]
                if(is.na(val) || abs(val) > 10) 0 else val
              } else 0
              
              epsilon[i] <- eps
              
              if(eps != 0) {
                # Calculate update with improved stability
                logit_current <- qlogis(pmin(pmax(predict_Qstar[,i], 0.001), 0.999))
                delta <- eps * H
                delta <- pmin(pmax(delta, -5), 5)  # Wider bounds for update
                updated_logit <- logit_current + delta
                predict_Qstar[,i] <- pmin(pmax(plogis(updated_logit), ybound[1]), ybound[2])
              }
            }
            
            if(debug) {
              cat("\nGLM fit details:")
              if(!is.null(fit)) {
                cat("\nCoefficients:\n")
                print(coef(fit))
                cat("\nSummary:\n")
                print(summary(fit))
              }
              cat("\nEpsilon value after processing:", eps)
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
              p <- if(rule_treatments[idx] == 1) {
                pmin(pmax(g_matrix[idx, 1], 1e-6), 1-1e-6)
              } else {
                pmin(pmax(1 - g_matrix[idx, 1], 1e-6), 1-1e-6)
              }
            } else {
              p <- pmin(pmax(g_matrix[idx, rule_treatments[idx]], 1e-6), 1-1e-6)
            }
            weights[idx] <- marginal_prob / p
          }
        }
        
        # Improved weight trimming for IPTW
        weights <- pmin(weights, quantile(weights[weights > 0], 0.95, na.rm=TRUE))
        
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
    
    list(
      "Qstar" = predict_Qstar,
      "epsilon" = epsilon,
      "Qstar_gcomp" = predicted_Y,
      "Qstar_iptw" = matrix(iptw_means, nrow=1),
      "Y" = observed_Y
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