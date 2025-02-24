###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

# Safe prediction getter function
safe_get_preds <- function(preds_list, t, n_ids = n) {
  if(is.null(n_ids) || n_ids <= 0) {
    stop("Invalid n_ids value")
  }
  
  # Handle out of bounds time index
  if(t > length(preds_list)) {
    print(paste("Time", t, "exceeds predictions list length", length(preds_list), "using last available"))
    t <- length(preds_list)
  }
  
  # Get predictions
  preds <- preds_list[[t]]
  if(is.null(preds) || length(preds) == 0) {
    print(paste("No predictions for time", t, "using default"))
    return(matrix(0.5, nrow=n_ids, ncol=1))
  }
  
  # Convert to matrix and ensure numeric
  if(!is.matrix(preds)) {
    if(is.data.frame(preds)) {
      preds <- as.matrix(preds)
    } else {
      preds <- matrix(as.numeric(preds), ncol=1)
    }
  }
  
  # Ensure numeric matrix
  mode(preds) <- "numeric"
  
  # Handle dimensions with validation
  if(nrow(preds) != n_ids) {
    original_rows <- nrow(preds)
    if(nrow(preds) < n_ids) {
      # If too short, repeat the last value
      padding_rows <- n_ids - nrow(preds)
      if(ncol(preds) > 0) {
        padding <- matrix(rep(tail(preds, ncol(preds)), 
                              length.out=padding_rows * ncol(preds)), 
                          nrow=padding_rows)
        preds <- rbind(preds, padding)
      } else {
        print("Warning: preds has 0 columns, creating default matrix")
        preds <- matrix(0.5, nrow=n_ids, ncol=1)
      }
    } else {
      # If too long, truncate
      preds <- preds[1:n_ids, , drop=FALSE]
    }
    print(paste("Adjusted matrix dimensions from", original_rows, "to", nrow(preds), "rows"))
  }
  
  # Final dimension check
  if(!identical(dim(preds)[1], n_ids)) {
    print(paste("Warning: Final dimensions", paste(dim(preds), collapse="x"), 
                "don't match expected n_ids", n_ids))
  }
  
  return(preds)
}

# Safe cumulative prediction handler
safe_get_cuml_preds <- function(preds, n_ids = n) {
  cuml_preds <- vector("list", length(preds))
  
  for(t in seq_along(preds)) {
    if(t == 1) {
      cuml_preds[[t]] <- safe_get_preds(list(preds[[t]]), 1, n_ids)
    } else {
      prev_preds <- cuml_preds[[t-1]]
      curr_preds <- safe_get_preds(list(preds[[t]]), 1, n_ids)
      # Scale probabilities before multiplication
      scaled_prev <- (prev_preds + 1) / 2  # Scale to [0.5, 1]
      cuml_preds[[t]] <- curr_preds * scaled_prev
    }
  }
  
  return(cuml_preds)
}

# Update process_predictions function signature
process_predictions <- function(slice, type="A", t=NULL, t_end=NULL, n_ids=NULL, J=NULL, 
                                ybound=NULL, gbound=NULL, debug=FALSE) {
  # Get total dimensions from input
  n_total_samples <- if(is.null(slice)) 0 else nrow(slice)
  samples_per_time <- if(!is.null(t_end)) n_total_samples %/% (t_end + 1) else 1
  
  # Get time slice if t is provided and all required parameters are present
  if(!is.null(t) && !is.null(samples_per_time) && !is.null(n_total_samples)) {
    slice <- get_time_slice(
      preds_r = slice,
      t = t,
      samples_per_time = samples_per_time,
      n_total_samples = n_total_samples,
      debug = debug
    )
  }
  
  # Handle invalid slice
  if(is.null(slice)) {
    if(debug) cat("Creating default predictions\n")
    if(is.null(n_ids) || is.null(J)) {
      warning("Missing n_ids or J for default predictions")
      return(NULL)
    }
    return(matrix(
      if(type == "A") 1/J else 0,
      nrow=n_ids,
      ncol=if(type == "A") J else 1
    ))
  }
  
  # Ensure matrix format and proper dimensions
  if(!is.matrix(slice)) {
    if(is.null(J)) {
      J <- if(type == "A") ncol(slice) else 1
    }
    slice <- matrix(slice, ncol=if(type == "A") J else 1)
  }
  
  # Interpolate if needed and n_ids is provided
  if(!is.null(n_ids) && nrow(slice) != n_ids) {
    new_slice <- matrix(0, nrow=n_ids, ncol=ncol(slice))
    for(j in seq_len(ncol(slice))) {
      x_old <- seq(0, 1, length.out=nrow(slice))
      x_new <- seq(0, 1, length.out=n_ids)
      new_slice[,j] <- approx(x_old, slice[,j], x_new)$y
    }
    slice <- new_slice
  }
  
  # Process based on type
  result <- switch(type,
                   "Y" = {
                     if(is.null(ybound)) {
                       warning("Missing ybound for Y predictions")
                       slice
                     } else {
                       pmin(pmax(slice, ybound[1]), ybound[2])
                     }
                   },
                   "C" = {
                     if(is.null(gbound)) {
                       warning("Missing gbound for C predictions")
                       slice
                     } else {
                       pmin(pmax(slice, gbound[1]), gbound[2])
                     }
                   },
                   "A" = {
                     if(is.null(gbound) || is.null(J)) {
                       warning("Missing gbound or J for A predictions")
                       slice
                     } else {
                       # For treatment predictions, ensure proper probabilities
                       t(apply(slice, 1, function(row) {
                         if(any(is.na(row)) || any(!is.finite(row))) return(rep(1/J, J))
                         bounded <- pmax(row, gbound[1])
                         bounded / sum(bounded)
                       }))
                     }
                   }
  )
  
  # Add column names
  colnames(result) <- switch(type,
                             "Y" = "Y",
                             "C" = "C",
                             "A" = paste0("A", 1:J)
  )
  
  return(result)
}

# Helper function to get time slice
get_time_slice <- function(preds_r, t, samples_per_time, n_total_samples, debug=FALSE) {
  # Calculate slice indices
  start_idx <- ((t-1) * samples_per_time) + 1
  end_idx <- min(t * samples_per_time, n_total_samples)
  
  # Validate indices
  if(start_idx > n_total_samples || end_idx < start_idx) {
    if(debug) cat(sprintf("Invalid slice indices [%d:%d]\n", start_idx, end_idx))
    return(NULL)
  }
  
  # Extract and validate slice
  slice <- preds_r[start_idx:end_idx, , drop=FALSE]
  if(is.null(slice) || nrow(slice) == 0 || ncol(slice) == 0) {
    if(debug) cat("Empty slice extracted\n")
    return(NULL)
  }
  
  if(debug) {
    cat(sprintf("\nTime %d slice [%d:%d]:\n", t-1, start_idx, end_idx))
    cat("Shape:", paste(dim(slice), collapse=" x "), "\n")
    cat("Range:", paste(range(slice, na.rm=TRUE), collapse=" - "), "\n")
  }
  
  return(slice)
}

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
    grep("^A\\.[0-9]+", colnames(tmle_dat), value=TRUE),
    grep("^A[0-9]+$", colnames(tmle_dat), value=TRUE)
  )
  print(paste("Found", length(A_cols), "treatment columns:", paste(A_cols, collapse=", ")))
  
  # If no A columns found, try to get from treatment history
  if(length(A_cols) == 0) {
    print("No direct treatment columns found, checking alternatives...")
    
    # Check for treatment columns in specific order of preference
    target_cols <- grep("^target", colnames(tmle_dat), value=TRUE)
    if(length(target_cols) == 0) {
      target_cols <- grep("^treatment\\.", colnames(tmle_dat), value=TRUE)
    }
    
    if(length(target_cols) > 0) {
      print(paste("Found", length(target_cols), "target columns:", paste(target_cols, collapse=", ")))
      # Extract treatment matrix
      treatment_matrix <- as.matrix(tmle_dat[target_cols])
      # Create time-specific A columns
      for(t in 0:t.end) {
        col_name <- paste0("A.", t)
        tmle_dat[[col_name]] <- treatment_matrix[, min(t + 1, ncol(treatment_matrix))]
      }
      # Set base A column from first time point
      tmle_dat$A <- tmle_dat[[paste0("A.", 0)]]
      print("Created treatment columns from target columns")
    } else {
      # Check for diagnosis-based treatment assignment
      diagnoses <- c("mdd", "bipolar", "schiz") 
      has_diagnoses <- any(diagnoses %in% colnames(tmle_dat))
      
      if(has_diagnoses) {
        print("Creating treatment assignments based on diagnoses")
        # Initialize based on diagnoses if available
        tmle_dat$A <- ifelse(tmle_dat$schiz == 1, 2,
                             ifelse(tmle_dat$bipolar == 1, 4,
                                    ifelse(tmle_dat$mdd == 1, 1, 5)))
        # Create time-specific treatment columns
        for(t in 0:t.end) {
          tmle_dat[[paste0("A.", t)]] <- tmle_dat$A
        }
      } else {
        warning("No treatment or diagnosis information found - using default treatment assignment")
        tmle_dat$A <- 5  # Default to treatment 5
        for(t in 0:t.end) {
          tmle_dat[[paste0("A.", t)]] <- 5
        }
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

process_time_points <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, g_preds_bin_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size, n_ids, output_dir,
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
      "lstm",
      "getTMLELongLSTM",
      "process_g_preds",
      "get_c_preds", 
      "get_y_preds",
      "calculate_iptw",
      "log_iptw_error",
      "track_initial_data",  # Add tracking functions
      "track_tmle_results",
      "track_stored_results",
      "process_time_points_tracking",
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
    
    # Update cluster environment setup with debug functions
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
      ybound = ybound,
      output_dir = output_dir,
      track_initial_data = track_initial_data,
      track_tmle_results = track_tmle_results,
      track_stored_results = track_stored_results,
      process_time_points_tracking = process_time_points_tracking
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
        output_dir = output_dir,
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
        output_dir = output_dir,
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
        output_dir = output_dir,
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
        output_dir = output_dir,
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
      cat("Observed Y mean:\n")
      print(mean(tmle_contrasts[[t]]$Y, na.rm=TRUE))
      cat("Number of valid rules:\n")
      print(sapply(1:ncol(obs.rules[[t]]), function(i) sum(obs.rules[[t]][,i], na.rm=TRUE)))
    }
    
    # Add dimension checks for final output
    cat("\nFinal output dimensions:")
    cat("\nNumber of time points:", length(tmle_contrasts))
    for(t in 1:t_end) {
      cat(sprintf("\nTime point %d:", t))
      cat("\n  Qstar:", paste(dim(tmle_contrasts[[t]]$Qstar), collapse=" x "))
      cat("\n  Qstar_gcomp:", paste(dim(tmle_contrasts[[t]]$Qstar_gcomp), collapse=" x "))
      cat("\n  Qstar_iptw:", paste(dim(tmle_contrasts[[t]]$Qstar_iptw), collapse=" x "))
      cat("\n  Y:", paste(dim(matrix(tmle_contrasts[[t]]$Y)), collapse=" x "))
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

get_y_preds <- function(initial_model_for_Y, t, n_ids, ybound, debug) {
  if(debug) {
    cat("\nEntering get_y_preds")
    cat("\nReceived initial_model_for_Y type:", class(initial_model_for_Y))
    cat("\nExpected n_ids:", n_ids)  
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
        
        # Identify censored values
        is_censored <- preds == -1 | is.na(preds)
        if(any(is_censored)) {
          if(debug) cat("\nHandling", sum(is_censored), "censored values")
          # Keep censored values as -1
          preds[is_censored] <- -1
        }
        
        if(length(preds) != n_ids) {
          if(debug) cat("\nLength mismatch: preds=", length(preds), " n_ids=", n_ids)
          # Match dimensions preserving censoring
          if(length(preds) > n_ids) {
            preds <- preds[1:n_ids]
            is_censored <- is_censored[1:n_ids]
          } else {
            preds <- rep(preds, length.out=n_ids)
            is_censored <- rep(is_censored, length.out=n_ids)
          }
        }
        result_matrix <- matrix(preds, nrow=n_ids)
      } else {
        if(debug) cat("\nNo preds in list, using default matrix")
        result_matrix <- matrix(0.5, nrow=n_ids, ncol=1)
      }
    } else if(is.vector(initial_model_for_Y)) {
      if(debug) cat("\nProcessing vector input of length:", length(initial_model_for_Y))
      
      # Identify censored values
      is_censored <- initial_model_for_Y == -1 | is.na(initial_model_for_Y)
      
      # Apply bounds only to non-censored values
      temp_vector <- initial_model_for_Y
      temp_vector[!is_censored] <- pmin(pmax(temp_vector[!is_censored], ybound[1]), ybound[2])
      
      if(length(temp_vector) != n_ids) {
        temp_vector <- rep(temp_vector, length.out=n_ids)
        is_censored <- rep(is_censored, length.out=n_ids)
      }
      result_matrix <- matrix(temp_vector, nrow=n_ids)
      
    } else if(is.matrix(initial_model_for_Y)) {
      if(debug) cat("\nProcessing matrix input with dims:", paste(dim(initial_model_for_Y), collapse=" x "))
      
      # Identify censored values
      is_censored <- initial_model_for_Y == -1 | is.na(initial_model_for_Y)
      
      if(nrow(initial_model_for_Y) != n_ids) {
        if(debug) cat("\nRow count mismatch: matrix=", nrow(initial_model_for_Y), " n_ids=", n_ids)
        # Match dimensions preserving censoring
        result_matrix <- matrix(initial_model_for_Y[1:min(nrow(initial_model_for_Y), n_ids),], 
                                nrow=n_ids, ncol=ncol(initial_model_for_Y))
        is_censored <- matrix(is_censored[1:min(nrow(is_censored), n_ids),],
                              nrow=n_ids, ncol=ncol(is_censored))
      } else {
        result_matrix <- initial_model_for_Y
      }
      
      # Apply bounds only to non-censored values
      result_matrix[!is_censored] <- pmin(pmax(result_matrix[!is_censored], ybound[1]), ybound[2])
    } else {
      if(debug) cat("\nUnhandled input type, using default matrix")
      result_matrix <- matrix(0.5, nrow=n_ids, ncol=1)
    }
    
    # Final censoring check
    if(exists("is_censored")) {
      result_matrix[is_censored] <- -1
    }
    
    # Ensure proper dimensions
    if(ncol(result_matrix) > 1) {
      if(debug) cat("\nTaking first column of multi-column matrix")
      result_matrix <- result_matrix[,1,drop=FALSE]
    }
    
    if(debug) {
      cat("\nFinal matrix dimensions:", paste(dim(result_matrix), collapse=" x "))
      is_censored <- result_matrix == -1 | is.na(result_matrix)
      cat("\nValue summary (non-censored):", 
          paste(range(result_matrix[!is_censored], na.rm=TRUE), collapse="-"))
      cat("\nCensored values:", sum(is_censored))
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
    is_censored <- result == -1 | is.na(result)
    cat("Y predictions summary:\n")
    print(summary(as.vector(result[!is_censored])))
    cat("\nCensored values:", sum(is_censored))
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
      
      # Get valid weights that match valid outcomes
      valid_weights <- weights[valid_idx]
      valid_outcomes <- outcomes[valid_idx]
      
      # Check if vectors have matching lengths and are non-empty
      if(length(valid_weights) > 0 && length(valid_outcomes) > 0) {
        # If lengths don't match, truncate to the shorter length
        if(length(valid_weights) != length(valid_outcomes)) {
          min_len <- min(length(valid_weights), length(valid_outcomes))
          valid_weights <- valid_weights[1:min_len]
          valid_outcomes <- valid_outcomes[1:min_len]
        }
        
        # Ensure weights sum to 1
        valid_weights <- valid_weights / sum(valid_weights, na.rm=TRUE)
        
        # Calculate weighted mean with matching vectors
        iptw_means[rule_idx] <- weighted.mean(valid_outcomes, valid_weights, na.rm=TRUE)
      } else {
        iptw_means[rule_idx] <- mean(predict_Qstar[,rule_idx], na.rm=TRUE)  # Fallback
      }
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

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded,
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size,
                            current_t, output_dir, debug=FALSE) {
  
  if(debug) {
    cat("\nStarting getTMLELongLSTM")
    cat("\nInitial data dimensions:", paste(dim(initial_model_for_Y_data), collapse=" x "))
    cat("\nobs.rules dimensions:", paste(dim(obs.rules), collapse=" x "))
    cat("\nTime point:", current_t, "of", t_end)
  }
  
  # Get dimensions
  n_ids <- nrow(obs.rules)
  
  # Extract data for current window with validation
  Y <- if(!is.null(initial_model_for_Y_data$Y)) {
    # Use actual values from input data
    y_vals <- initial_model_for_Y_data$Y
    if(all(is.na(y_vals)) || length(y_vals) == 0) {
      # Get future outcomes for this timestep
      sapply(1:n_ids, function(i) {
        id <- initial_model_for_Y_data$ID[i]
        
        # Get data row for this ID
        data_idx <- match(id, data$ID)
        if(is.na(data_idx)) return(-1)
        
        # Calculate future time index
        future_idx <- current_t + window_size + 1
        if(future_idx > ncol(target_matrix)) return(-1)
        
        # Get value
        val <- target_matrix[data_idx, future_idx]
        if(is.na(val)) return(-1)
        
        # Return actual value
        val
      })
    } else {
      # Use existing y_vals
      y_vals
    }
  } else {
    warning("No Y values found in initial_model_for_Y_data")
    # Use actual Y values from data if available
    if("Y" %in% colnames(data)) {
      data$Y
    } else {
      # Fallback to target values
      data_long$target
    }
  }
  
  # Validate Y values
  print("Y value validation:")
  print(paste("Y range:", paste(range(Y, na.rm=TRUE), collapse=" - ")))
  print(paste("NA count:", sum(is.na(Y))))
  print("Y value distribution:")
  print(table(Y, useNA="ifany"))
  C <- initial_model_for_Y_data$C
  
  # Identify censoring status
  is_censored <- Y == -1 | is.na(Y) | C == 1
  valid_rows <- !is_censored
  
  if(debug) {
    cat("\nCensoring summary:")
    cat("\n  Total observations:", length(Y))
    cat("\n  Censored:", sum(is_censored))
    cat("\n  Valid:", sum(valid_rows))
  }
  
  # Run LSTM predictions for each rule
  rule_predictions <- vector("list", length(tmle_rules))
  names(rule_predictions) <- names(tmle_rules)
  
  # Process each treatment rule
  for(rule in names(tmle_rules)) {
    if(debug) cat("\nProcessing rule:", rule)
    
    # Get rule-specific treatments
    shifted_data <- switch(rule,
                           "static" = static_mtp_lstm(initial_model_for_Y_data),
                           "dynamic" = dynamic_mtp_lstm(initial_model_for_Y_data), 
                           "stochastic" = stochastic_mtp_lstm(initial_model_for_Y_data)
    )
    
    # Create rule-specific dataset with proper A columns
    rule_data <- initial_model_for_Y_data
    
    # Set treatment columns based on shifted data 
    for(t in 0:t_end) {
      col_name <- paste0("A.", t)
      rule_data[[col_name]] <- shifted_data$A0[match(rule_data$ID, shifted_data$ID)]
    }
    
    # Base A column also needs to be set
    rule_data$A <- rule_data[[paste0("A.", 0)]]
    
    # Ensure tmle_covars_Y includes A columns
    tmle_covars_Y_with_A <- unique(c(
      tmle_covars_Y,
      grep("^A\\.", colnames(rule_data), value=TRUE),
      "A"
    ))
    
    # Get LSTM predictions with inference=TRUE
    if(debug) cat("\nRunning LSTM for rule:", rule)
    
    lstm_preds <- lstm(
      data = rule_data,
      outcome = "Y",
      covariates = tmle_covars_Y_with_A,
      t_end = t_end,
      window_size = window_size,
      out_activation = "sigmoid",
      loss_fn = "binary_crossentropy",
      output_dir = output_dir,
      J = 1,
      ybound = ybound,
      gbound = gbound,
      inference = TRUE,
      debug = debug
    )
    
    # Store predictions for this rule
    rule_predictions[[rule]] <- lstm_preds
  }
  
  # Initialize matrices for predictions 
  Qs <- matrix(NA, nrow=n_ids, ncol=length(tmle_rules))
  colnames(Qs) <- names(tmle_rules)
  
  # Process predictions for each rule
  for(i in seq_along(rule_predictions)) {
    rule <- names(tmle_rules)[i]
    preds <- rule_predictions[[rule]]
    
    if(is.null(preds)) {
      # Default prediction if LSTM failed
      Qs[,i] <- rep(mean(Y[valid_rows], na.rm=TRUE), n_ids)
    } else {
      # Use predictions for current timepoint
      t_preds <- preds[[min(current_t + 1, length(preds))]]
      if(is.null(t_preds)) {
        Qs[,i] <- rep(mean(Y[valid_rows], na.rm=TRUE), n_ids)
      } else {
        # Ensure matching dimensions and respect bounds
        t_preds <- rep(t_preds, length.out=n_ids)
        Qs[,i] <- pmin(pmax(t_preds, ybound[1]), ybound[2])
      }
    }
  }
  
  # Process initial predictions to ensure proper format
  initial_preds <- matrix(initial_model_for_Y_preds, nrow=n_ids)
  
  # Create QAW matrix 
  QAW <- cbind(QA = initial_preds, Qs)
  colnames(QAW) <- c("QA", names(tmle_rules))
  
  # Apply bounds to QAW
  QAW <- pmin(pmax(QAW, ybound[1]), ybound[2])
  
  # Process treatment predictions
  g_matrix <- if(is.list(g_preds_bounded)) {
    do.call(cbind, lapply(seq_len(ncol(obs.treatment)), function(j) {
      if(j <= length(g_preds_bounded) && !is.null(g_preds_bounded[[j]])) {
        pred <- matrix(g_preds_bounded[[j]], nrow=n_ids)
        if(ncol(pred) > 1) pred[,1] else pred
      } else {
        rep(1/ncol(obs.treatment), n_ids)
      }
    }))
  } else if(is.matrix(g_preds_bounded)) {
    if(nrow(g_preds_bounded) != n_ids) {
      matrix(rep(g_preds_bounded, length.out=n_ids*ncol(g_preds_bounded)), 
             ncol=ncol(g_preds_bounded))
    } else {
      g_preds_bounded 
    }
  } else {
    matrix(1/ncol(obs.treatment), nrow=n_ids, ncol=ncol(obs.treatment))
  }
  
  # Calculate clever covariates with proper dimensions
  clever_covariates <- matrix(0, nrow=n_ids, ncol=ncol(obs.rules))
  is_censored_adj <- rep(is_censored, length.out=n_ids)
  
  for(i in seq_len(ncol(obs.rules))) {
    clever_covariates[,i] <- obs.rules[,i] * (!is_censored_adj)
  }
  
  # Calculate censoring-adjusted weights
  weights <- tryCatch({
    # Adjust treatment probabilities for censoring
    C_matrix <- matrix(rep(C_preds_bounded, ncol(g_matrix)), 
                       nrow=nrow(C_preds_bounded),
                       ncol=ncol(g_matrix))
    
    # Joint probability of treatment and not being censored
    probs <- g_matrix * (1 - C_matrix)
    bounded_probs <- pmin(pmax(probs, gbound[1]), gbound[2])
    
    # Calculate weights per rule
    weights_matrix <- matrix(0, nrow=n_ids, ncol=ncol(obs.rules))
    for(i in seq_len(ncol(obs.rules))) {
      valid_idx <- clever_covariates[,i] > 0
      if(any(valid_idx)) {
        # Calculate stabilized weights
        treatment_probs <- rowSums(obs.treatment[valid_idx,] * bounded_probs[valid_idx,], na.rm=TRUE)
        treatment_probs[treatment_probs < gbound[1]] <- gbound[1]
        
        # IPCW weights
        cens_weights <- 1 / (1 - C_matrix[valid_idx,1])
        weights_matrix[valid_idx,i] <- cens_weights / treatment_probs
        
        # Normalize and trim extreme weights
        rule_weights <- weights_matrix[valid_idx,i]
        max_weight <- quantile(rule_weights, 0.99, na.rm=TRUE)
        weights_matrix[valid_idx,i] <- pmin(rule_weights, max_weight)
        weights_matrix[valid_idx,i] <- weights_matrix[valid_idx,i] / sum(weights_matrix[valid_idx,i])
      }
    }
    weights_matrix
    
  }, error = function(e) {
    if(debug) cat("\nWeight calculation error:", e$message)
    matrix(1, nrow=n_ids, ncol=ncol(obs.rules))
  })
  
  # Fit targeting models
  updated_models <- vector("list", ncol(clever_covariates))
  
  for(i in seq_len(ncol(clever_covariates))) {
    if(debug) cat("\nFitting model for rule", i)
    
    # Create rule-specific model data
    model_data <- data.frame(
      y = if(current_t < t_end) QAW[,"QA"] else Y,
      offset = qlogis(pmax(pmin(QAW[,i+1], 0.9999), 0.0001)),  # Bound values for qlogis
      weights = weights[,i]
    )
    
    # Add validation before model fitting to ensure sufficient data
    valid_rows <- complete.cases(model_data) &  # All columns present and not NA 
      is.finite(model_data$y) &  # y values are finite
      is.finite(model_data$offset) &  # offset values are finite 
      is.finite(model_data$weights) & # weights are finite
      model_data$y != -1 & # not censored
      model_data$weights > 0 & # positive weights
      !is.infinite(qlogis(model_data$y)) # y can be logit transformed
    
    # Remove invalid rows
    model_data <- model_data[valid_rows, , drop=FALSE]
    
    if(nrow(model_data) > 0 && 
       any(!is.na(model_data$y)) && 
       any(!is.na(model_data$offset)) &&
       any(model_data$weights > 0)) {
      
      updated_models[[i]] <- tryCatch({
        # Ensure proper column names 
        colnames(model_data) <- c("y", "offset", "weights")
        
        # Fit GLM
        glm(y ~ 1 + offset(offset),
            weights = weights,
            family = quasibinomial(),
            data = model_data)
      }, error = function(e) {
        if(debug) cat("\nGLM error:", e$message)
        NULL
      })
    } else {
      if(debug) cat("\nInsufficient valid data for rule", i)
      updated_models[[i]] <- NULL
    }
  }
  
  # Generate predictions with proper error handling
  Qstar <- do.call(cbind, lapply(seq_along(updated_models), function(i) {
    if(is.null(updated_models[[i]])) {
      # Use default prediction when model is NULL
      rep(mean(Y[valid_rows], na.rm=TRUE), n_ids)
    } else {
      tryCatch({
        preds <- predict(updated_models[[i]], type="response")
        # Ensure proper length
        rep(preds, length.out=n_ids)
      }, error = function(e) {
        if(debug) cat("\nPrediction error for rule", i, ":", e$message)
        rep(mean(Y[valid_rows], na.rm=TRUE), n_ids)
      })
    }
  }))
  
  # Ensure proper column names 
  if(ncol(Qstar) == ncol(obs.rules)) {
    colnames(Qstar) <- colnames(obs.rules)
  } else {
    warning("Qstar dimensions don't match obs.rules dimensions")
  }
  
  # Calculate IPTW estimates with weight length check
  Qstar_iptw <- matrix(sapply(1:ncol(clever_covariates), function(i) {
    valid_idx <- clever_covariates[,i] > 0 & !is_censored_adj
    if(any(valid_idx)) {
      # Ensure weights and Y have same length
      w <- weights[valid_idx,i]
      y <- Y[valid_idx]
      if(length(w) == length(y)) {
        weighted.mean(y, w, na.rm=TRUE)
      } else {
        if(debug) cat(sprintf("\nWeight length mismatch for rule %d: weights=%d, Y=%d", 
                              i, length(w), length(y)))
        mean(Y[valid_rows], na.rm=TRUE)
      }
    } else {
      mean(Y[valid_rows], na.rm=TRUE)
    }
  }), nrow=1)
  colnames(Qstar_iptw) <- colnames(obs.rules)
  
  # Calculate G-computation estimates
  Qstar_gcomp <- matrix(QAW[,-1], ncol=ncol(obs.rules))
  colnames(Qstar_gcomp) <- colnames(obs.rules)
  
  # Get epsilons
  epsilon <- sapply(updated_models, function(mod) {
    if(is.null(mod)) 0 else coef(mod)[1]
  })
  
  return(list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = clever_covariates,
    "weights" = weights,
    "updated_model_for_Y" = updated_models,
    "Qstar" = Qstar,
    "epsilon" = epsilon,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qstar_gcomp,
    "Y" = Y,
    "ID" = initial_model_for_Y_data$ID
  ))
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