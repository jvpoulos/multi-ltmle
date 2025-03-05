###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

verify_reticulate <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Reticulate package is not available. Please install with: install.packages('reticulate')")
  }
  
  # Try to initialize Python
  reticulate::py_available(initialize = TRUE)
  
  # Check if Python is actually accessible
  tryCatch({
    py <- reticulate::py_run_string("x = 1+1")
    return(TRUE)
  }, error = function(e) {
    stop("Python is not properly configured: ", e$message)
  })
}

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
      # First time point - just get the predictions directly
      cuml_preds[[t]] <- safe_get_preds(list(preds[[t]]), 1, n_ids)
    } else {
      # Get previous and current predictions
      prev_preds <- cuml_preds[[t-1]]
      curr_preds <- safe_get_preds(list(preds[[t]]), 1, n_ids)
      
      # Verify both have compatible dimensions
      if(is.null(prev_preds) || is.null(curr_preds)) {
        # Handle null predictions by using defaults
        cuml_preds[[t]] <- matrix(0.5, nrow=n_ids, ncol=1)
        next
      }
      
      # Ensure we have matrices with matching dimensions
      if(!is.matrix(prev_preds)) prev_preds <- matrix(prev_preds, ncol=1)
      if(!is.matrix(curr_preds)) curr_preds <- matrix(curr_preds, ncol=1)
      
      # Make dimensions match
      if(ncol(prev_preds) != ncol(curr_preds)) {
        # Adjust columns to match
        max_cols <- max(ncol(prev_preds), ncol(curr_preds))
        if(ncol(prev_preds) < max_cols) {
          prev_preds <- cbind(prev_preds, matrix(0.5, nrow=nrow(prev_preds), ncol=max_cols-ncol(prev_preds)))
        }
        if(ncol(curr_preds) < max_cols) {
          curr_preds <- cbind(curr_preds, matrix(0.5, nrow=nrow(curr_preds), ncol=max_cols-ncol(curr_preds)))
        }
      }
      
      # Match row counts
      if(nrow(prev_preds) != nrow(curr_preds)) {
        min_rows <- min(nrow(prev_preds), nrow(curr_preds))
        if(min_rows < n_ids) {
          # Expand to n_ids
          prev_preds <- matrix(rep(prev_preds, length.out=n_ids*ncol(prev_preds)), nrow=n_ids)
          curr_preds <- matrix(rep(curr_preds, length.out=n_ids*ncol(curr_preds)), nrow=n_ids)
        } else {
          # Truncate to match
          prev_preds <- prev_preds[1:min_rows,, drop=FALSE]
          curr_preds <- curr_preds[1:min_rows,, drop=FALSE]
        }
      }
      
      # Force numeric type for both matrices
      storage.mode(prev_preds) <- "numeric"
      storage.mode(curr_preds) <- "numeric"
      
      # Replace NAs with 0.5
      prev_preds[is.na(prev_preds)] <- 0.5
      curr_preds[is.na(curr_preds)] <- 0.5
      
      # CORRECT SCALING: Scale to [0.5, 1] range to prevent underflow
      # For probabilities in [0,1], this maps to [0.5, 1]
      scaled_prev <- 0.5 + (prev_preds / 2)
      
      # Perform multiplication with error handling
      cuml_preds[[t]] <- tryCatch({
        # Element-wise multiplication
        result <- curr_preds * scaled_prev
        
        # Check for invalid values and replace them
        result[!is.finite(result)] <- 0.5
        result
      }, 
      error = function(e) {
        # If error occurs, return a valid default matrix
        message("Error in cumulative prediction calculation: ", e$message)
        matrix(0.5, nrow=nrow(curr_preds), ncol=ncol(curr_preds))
      })
    }
  }
  
  return(cuml_preds)
}

# Optimized version of process_predictions in tmle_fns_lstm.R

# Ensure process_predictions is in global environment
process_predictions <- function(slice, type="A", t=NULL, t_end=NULL, n_ids=NULL, J=NULL, 
                               ybound=NULL, gbound=NULL, debug=FALSE) {
  # Ensure numeric types for calculations
  if (!is.null(t_end)) t_end <- as.integer(t_end)
  if (!is.null(t)) t <- as.integer(t)
  if (!is.null(n_ids)) n_ids <- as.integer(n_ids)
  if (!is.null(J)) J <- as.integer(J)
  
  # Get total dimensions from input with type safety
  n_total_samples <- if(is.null(slice)) 0 else as.integer(nrow(slice))
  
  # Calculate samples per time with validation - optimize division
  samples_per_time <- 1  # Default value
  if(!is.null(t_end) && !is.null(n_total_samples)) {
    # Proper integer division to avoid overflow
    if(t_end > 0) {
      # Use direct calculation instead of ceiling
      samples_per_time <- as.integer((n_total_samples + t_end) / (t_end + 1))
      
      # Validate the result is reasonable
      if(samples_per_time <= 0 || samples_per_time > n_total_samples) {
        # Default to something reasonable
        samples_per_time <- min(14000, max(1, as.integer(n_total_samples / 37)))
        if(debug) cat("Corrected samples_per_time to:", samples_per_time, "\n")
      }
    }
  }
  
  # Get time slice if t is provided and all required parameters are present
  if(!is.null(t) && !is.null(samples_per_time) && !is.null(n_total_samples)) {
    # Calculate slice indices directly instead of calling a separate function
    # FIXED CALCULATION
    chunk_size <- as.integer((n_total_samples + t_end) / (t_end + 1))
    start_idx <- ((t-1) * chunk_size) + 1
    end_idx <- min(t * chunk_size, n_total_samples)
    
    # Safety check for indices
    if(start_idx > n_total_samples || start_idx < 1 || end_idx < start_idx) {
      if(debug) cat(sprintf("Invalid slice indices [%d:%d] (n_total_samples=%d), using fallback\n", 
                            start_idx, end_idx, n_total_samples))
      
      # Simplify fallback by calculating directly
      chunks <- min(37, t_end + 1)  # Limit to standard number of time points
      chunk_size <- as.integer((n_total_samples + chunks - 1) / chunks)
      
      # Ensure t is within valid range
      t_adjusted <- min(t, chunks)
      
      # Calculate indices using adjusted approach
      start_idx <- ((t_adjusted-1) * chunk_size) + 1
      end_idx <- min(t_adjusted * chunk_size, n_total_samples)
    }
    
    # Extract slice based on type of input
    if(is.vector(slice)) {
      if(start_idx <= length(slice) && end_idx <= length(slice)) {
        slice <- matrix(slice[start_idx:end_idx], ncol=1)
      } else {
        if(debug) cat("Vector indices out of bounds\n")
        slice <- NULL
      }
    } else if(is.matrix(slice)) {
      if(start_idx <= nrow(slice) && end_idx <= nrow(slice)) {
        slice <- slice[start_idx:end_idx, , drop=FALSE]
      } else {
        if(debug) cat("Matrix indices out of bounds\n")
        slice <- NULL
      }
    } else if(is.data.frame(slice)) {
      if(start_idx <= nrow(slice) && end_idx <= nrow(slice)) {
        slice <- as.matrix(slice[start_idx:end_idx, , drop=FALSE])
      } else {
        if(debug) cat("Data frame indices out of bounds\n")
        slice <- NULL
      }
    } else {
      if(debug) cat("Unsupported slice type:", class(slice), "\n")
      slice <- NULL
    }
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
  
  # Ensure matrix format and proper dimensions - direct conversion
  if(!is.matrix(slice)) {
    if(is.null(J)) {
      J <- if(type == "A") ncol(slice) else 1
    }
    slice <- matrix(slice, ncol=if(type == "A") J else 1)
  }
  
  # Interpolate if needed and n_ids is provided
  if(!is.null(n_ids) && nrow(slice) != n_ids) {
    # Optimize interpolation with direct approx call
    if(nrow(slice) > 1) {
      x_old <- seq(0, 1, length.out=nrow(slice))
      x_new <- seq(0, 1, length.out=n_ids)
      
      # Preallocate matrix
      new_slice <- matrix(0, nrow=n_ids, ncol=ncol(slice))
      
      # Interpolate each column
      for(j in seq_len(ncol(slice))) {
        new_slice[,j] <- approx(x_old, slice[,j], x_new)$y
      }
      slice <- new_slice
    } else {
      # If only one row, just repeat it
      slice <- matrix(rep(slice, n_ids), nrow=n_ids, ncol=ncol(slice), byrow=TRUE)
    }
  }
  
  # Process based on type with more efficient code
  if(type == "Y") {
    if(!is.null(ybound)) {
      # Apply bounds in a single vectorized operation
      result <- pmin(pmax(slice, ybound[1]), ybound[2])
    } else {
      warning("Missing ybound for Y predictions")
      result <- slice
    }
  } else if(type == "C") {
    if(!is.null(gbound)) {
      # Apply bounds in a single vectorized operation
      result <- pmin(pmax(slice, gbound[1]), gbound[2])
    } else {
      warning("Missing gbound for C predictions")
      result <- slice
    }
  } else { # type == "A"
    if(!is.null(gbound) && !is.null(J)) {
      # Preallocate result matrix
      result <- matrix(0, nrow=nrow(slice), ncol=ncol(slice))
      
      # Optimize row operations with apply
      result <- t(apply(slice, 1, function(row) {
        if(any(is.na(row)) || any(!is.finite(row))) return(rep(1/J, J))
        # Use direct vector operations
        bounded <- pmax(row, gbound[1])
        bounded / sum(bounded)
      }))
    } else {
      warning("Missing gbound or J for A predictions")
      result <- slice
    }
  }
  
  # Add column names
  colnames(result) <- switch(type,
                             "Y" = "Y",
                             "C" = "C",
                             "A" = paste0("A", 1:J)
  )
  
  return(result)
}

get_time_slice <- function(preds_r, t, samples_per_time, n_total_samples, t_end = 36, debug=FALSE) {
  # Safety check for input parameters
  if(is.null(preds_r) || !is.numeric(t) || t < 1) {
    if(debug) cat("Invalid input parameters to get_time_slice\n")
    return(NULL)
  }
  
  # Validate and adjust samples_per_time if needed
  if(is.null(samples_per_time) || !is.numeric(samples_per_time) || samples_per_time <= 0) {
    # Calculate a reasonable default
    samples_per_time <- ceiling(n_total_samples / (t_end + 1))
    if(debug) cat("Using calculated samples_per_time:", samples_per_time, "\n")
  }
  
  # For safety, cap samples_per_time to a reasonable value
  if(samples_per_time > n_total_samples) {
    samples_per_time <- n_total_samples
    if(debug) cat("Capped samples_per_time to n_total_samples:", samples_per_time, "\n")
  }
  
  # Calculate slice indices - FIXED CALCULATION
  chunk_size <- ceiling(n_total_samples / (t_end + 1))
  start_idx <- ((t-1) * chunk_size) + 1
  end_idx <- min(t * chunk_size, n_total_samples)
  
  # Safety check for indices
  if(start_idx > n_total_samples || start_idx < 1 || end_idx < start_idx) {
    if(debug) cat(sprintf("Invalid slice indices [%d:%d] (n_total_samples=%d), using fallback method\n", 
                          start_idx, end_idx, n_total_samples))
    
    # Use simplified, more reliable fallback approach
    chunks <- min(37, t_end + 1)  # Limit to standard number of time points
    chunk_size <- ceiling(n_total_samples / chunks)
    
    # Ensure t is within valid range
    t_adjusted <- min(t, chunks)
    
    # Calculate indices using adjusted approach
    start_idx <- ((t_adjusted-1) * chunk_size) + 1
    end_idx <- min(t_adjusted * chunk_size, n_total_samples)
    
    if(debug) cat(sprintf("Fallback indices: [%d:%d]\n", start_idx, end_idx))
  }
  
  # Extract and validate slice with error handling
  slice <- tryCatch({
    if(is.vector(preds_r)) {
      # Handle vector case
      if(start_idx <= length(preds_r) && end_idx <= length(preds_r)) {
        slice_vec <- preds_r[start_idx:end_idx]
        if(length(slice_vec) > 0) {
          matrix(slice_vec, ncol=1)
        } else {
          NULL
        }
      } else {
        if(debug) cat("Vector indices out of bounds\n")
        NULL
      }
    } else if(is.matrix(preds_r)) {
      # Handle matrix case - ensure indices are valid
      if(start_idx <= nrow(preds_r) && end_idx <= nrow(preds_r)) {
        preds_r[start_idx:end_idx, , drop=FALSE]
      } else {
        if(debug) cat("Matrix indices out of bounds\n")
        NULL
      }
    } else if(is.data.frame(preds_r)) {
      # Handle data frame case
      if(start_idx <= nrow(preds_r) && end_idx <= nrow(preds_r)) {
        as.matrix(preds_r[start_idx:end_idx, , drop=FALSE])
      } else {
        if(debug) cat("Data frame indices out of bounds\n")
        NULL
      }
    } else {
      # Handle other cases
      if(debug) cat("Unsupported preds_r type:", class(preds_r), "\n")
      NULL
    }
  }, error = function(e) {
    if(debug) cat("Error extracting slice:", e$message, "\n")
    NULL
  })
  
  # Validate extracted slice
  if(is.null(slice) || length(slice) == 0 || 
     (is.matrix(slice) && (nrow(slice) == 0 || ncol(slice) == 0))) {
    if(debug) cat("Empty or invalid slice extracted\n")
    return(NULL)
  }
  
  # Ensure numeric mode
  storage.mode(slice) <- "numeric"
  
  if(debug) {
    cat(sprintf("\nTime %d slice [%d:%d]:\n", t-1, start_idx, end_idx))
    cat("Shape:", paste(dim(slice), collapse=" x "), "\n")
    if(all(is.finite(slice))) {
      cat("Range:", paste(range(slice, na.rm=TRUE), collapse=" - "), "\n")
    } else {
      cat("Contains non-finite values\n")
    }
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
  
  # Precompute these once instead of for each timepoint
  n_rules <- length(tmle_rules)
  base_covariates <- unique(gsub("\\.[0-9]+$", "", tmle_covars_Y))
  
  # Initialize Python variables just once outside the loop
  if (!exists("py", envir = .GlobalEnv, inherits = FALSE)) {
    py <- reticulate::py_run_string("x = 1+1")
    assign("py", py, envir = .GlobalEnv)
  }
  
  # Set Python variables once, not in each iteration
  py <- reticulate::py
  py$feature_cols <- base_covariates
  py$window_size <- window_size
  py$output_dir <- output_dir
  py$t_end <- t_end
  py$gbound <- gbound
  py$ybound <- ybound
  # Get J directly from the data when possible
  actual_J <- if(!is.null(g_preds_processed) && !is.null(g_preds_processed[[1]])) {
    if(is.matrix(g_preds_processed[[1]])) {
      ncol(g_preds_processed[[1]])
    } else if(is.list(g_preds_processed[[1]]) && !is.null(g_preds_processed[[1]][[1]])) {
      if(is.matrix(g_preds_processed[[1]][[1]])) {
        ncol(g_preds_processed[[1]][[1]])
      } else {
        6  # Default to 6 if structure is unexpected
      }
    } else {
      6  # Default to 6 if structure is unexpected
    }
  } else {
    6  # Default to 6 treatments if no data available
  }
  
  # Then update py$J with the correct value
  py$J <- actual_J
  
  # Preallocate full result matrices to avoid repeated memory allocation
  results <- vector("list", t_end)
  
  # Process time points
  time_points <- 1:t_end
  
  # Pre-process data before the loop for all timepoints if possible
  # Process shared data that doesn't change per timepoint
  
  # Use lapply instead of a for loop for better performance
  results <- lapply(time_points, function(t) {
    if(debug) cat(sprintf("\nProcessing time point %d/%d\n", t, t_end))
    time_start <- Sys.time()
    
    # Preallocate result matrices with the right dimensions
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
    
    # Process predictions in one step for all rules to avoid redundant processing
    current_g_preds <- process_g_preds(g_preds_processed, t, n_ids, py$J, gbound, debug)
    current_g_preds_bin <- process_g_preds(g_preds_bin_processed, t, n_ids, py$J, gbound, debug)
    current_c_preds <- get_c_preds(C_preds_processed, t, n_ids, gbound)
    current_y_preds <- get_y_preds(initial_model_for_Y, t, n_ids, ybound, debug)
    
    # Create treatment probability lists only once
    current_g_preds_list <- lapply(1:py$J, function(j) matrix(current_g_preds[,j], ncol=1))
    current_g_preds_bin_list <- lapply(1:py$J, function(j) matrix(current_g_preds_bin[,j], ncol=1))
    
    # Process both cases with optimized TMLE
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
    
    # Binary case - reuse matrices where possible
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
    
    # Calculate IPTW only once using vectorized approach
    current_rules <- obs.rules[[min(t, length(obs.rules))]]
    
    # Single IPTW calculation for both models
    tryCatch({
      # Multinomial IPTW
      tmle_contrast$Qstar_iptw <- calculate_iptw(current_g_preds, current_rules, 
                                                 tmle_contrast$Qstar, n_rules, gbound, debug)
      
      # Binary IPTW
      tmle_contrast_bin$Qstar_iptw <- calculate_iptw(current_g_preds_bin, current_rules,
                                                     tmle_contrast_bin$Qstar, n_rules, gbound, debug)
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
    
    # Return both models in a single list to reduce memory copying
    list(multinomial = tmle_contrast, binary = tmle_contrast_bin)
  })
  
  # Restructure results once at the end
  tmle_contrasts <- vector("list", t_end)
  tmle_contrasts_bin <- vector("list", t_end)
  
  for(t in 1:t_end) {
    # Use direct assignment instead of copying
    tmle_contrasts[[t]] <- results[[t]]$multinomial
    tmle_contrasts_bin[[t]] <- results[[t]]$binary
  }
  
  # Only do final debug output if needed
  if(debug) {
    cat("\nFinal time point processing summary:\n")
    # Include only essential summary metrics
    for(t in 1:t_end) {
      cat("\nTime point", t, "summary:")
      cat("\nTMLE estimates:", colMeans(tmle_contrasts[[t]]$Qstar, na.rm=TRUE))
      cat("\nIPTW estimates:", tmle_contrasts[[t]]$Qstar_iptw)
      cat("\nObserved Y mean:", mean(tmle_contrasts[[t]]$Y, na.rm=TRUE))
    }
  }
  
  return(list("multinomial" = tmle_contrasts, "binary" = tmle_contrasts_bin))
}

# This optimized version of process_time_points uses batch processing
# to make a single LSTM call for all time points and treatment rules
process_time_points_batch <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, g_preds_bin_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size, n_ids, output_dir,
                                cores = 1, debug = FALSE) {
  
  # Precompute these once instead of for each timepoint
  n_rules <- length(tmle_rules)
  base_covariates <- unique(gsub("\\.[0-9]+$", "", tmle_covars_Y))
  
  # Initialize Python variables just once outside the loop
  if (!exists("py", envir = .GlobalEnv, inherits = FALSE)) {
    py <- reticulate::py_run_string("x = 1+1")
    assign("py", py, envir = .GlobalEnv)
  }
  
  # Set Python variables once, not in each iteration
  py <- reticulate::py
  py$feature_cols <- base_covariates
  py$window_size <- window_size
  py$output_dir <- output_dir
  py$t_end <- t_end
  py$gbound <- gbound
  py$ybound <- ybound
  # Get J directly from the data when possible
  actual_J <- if(!is.null(g_preds_processed) && !is.null(g_preds_processed[[1]])) {
    if(is.matrix(g_preds_processed[[1]])) {
      ncol(g_preds_processed[[1]])
    } else if(is.list(g_preds_processed[[1]]) && !is.null(g_preds_processed[[1]][[1]])) {
      if(is.matrix(g_preds_processed[[1]][[1]])) {
        ncol(g_preds_processed[[1]][[1]])
      } else {
        6  # Default to 6 if structure is unexpected
      }
    } else {
      6  # Default to 6 if structure is unexpected
    }
  } else {
    6  # Default to 6 treatments if no data available
  }
  
  # Then update py$J with the correct value
  py$J <- actual_J
  
  # Reset LSTM model cache between full runs
  if (!exists("model_loaded_for_run", envir = .GlobalEnv)) {
    assign("model_loaded_for_run", FALSE, envir = .GlobalEnv)
    # Clear any existing cached models
    if (exists("cached_models", envir = .GlobalEnv)) {
      assign("cached_models", list(), envir = .GlobalEnv)
    }
    # Reset first_lstm_call flag
    if (exists("first_lstm_call", envir = .GlobalEnv)) {
      assign("first_lstm_call", TRUE, envir = .GlobalEnv)
    }
  }
  
  # We need to prepare datasets for all the rules to use batch processing
  # This only needs to be done once for all time points
  # Pre-compute rule-specific datasets for all time points
  rule_data_master <- vector("list", length(tmle_rules))
  names(rule_data_master) <- names(tmle_rules)
  
  if(debug) cat("\nPreparing rule datasets for batch processing...\n")
  
  # Create rule-specific datasets for all rules
  for(rule in names(tmle_rules)) {
    if(debug) cat(paste("Creating dataset for rule:", rule, "\n"))
    
    # Get rule-specific treatments
    shifted_data <- switch(rule,
                          "static" = static_mtp_lstm(initial_model_for_Y_data),
                          "dynamic" = dynamic_mtp_lstm(initial_model_for_Y_data),
                          "stochastic" = stochastic_mtp_lstm(initial_model_for_Y_data))
    
    # Create dataset with rule-specific treatments
    rule_data <- initial_model_for_Y_data
    id_mapping <- match(rule_data$ID, shifted_data$ID)
    
    # Set treatment columns efficiently
    for(t in 0:t_end) {
      col_name <- paste0("A.", t)
      rule_data[[col_name]] <- shifted_data$A0[id_mapping]
    }
    
    # Set base A column also
    rule_data$A <- rule_data[[paste0("A.", 0)]]
    
    # Store in master list
    rule_data_master[[rule]] <- rule_data
  }
  
  # Add A columns to covariates once
  tmle_covars_Y_with_A <- unique(c(
    tmle_covars_Y,
    grep("^A\\.", colnames(rule_data_master[[1]]), value=TRUE),
    "A"
  ))
  
  # Do LSTM model loading once for all time points
  # This is what loads the model and caches it
  if(!model_loaded_for_run) {
    if(debug) cat("\nLoading LSTM model once for all time points...\n")
    
    # Load the model once for all time points by running first rule
    rule <- names(tmle_rules)[1]
    lstm_preds <- lstm(
      data = rule_data_master[[rule]],
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
      debug = FALSE,
      batch_models = TRUE  # This caches the model
    )
    
    assign("model_loaded_for_run", TRUE, envir = .GlobalEnv)
    if(debug) cat("LSTM model loaded and cached.\n")
  }
  
  # Process all rules in batch to get predictions for all rules
  if(debug) cat("\nGenerating predictions for all rules...\n")
  all_lstm_preds <- lstm(
    data = NULL,  # Not used in batch mode
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
    debug = FALSE,
    batch_models = TRUE,
    batch_rules = rule_data_master  # Pass all rules at once
  )
  
  # Preallocate full result matrices to avoid repeated memory allocation
  results <- vector("list", t_end)
  
  # Process time points
  time_points <- 1:t_end
  
  # Use lapply instead of a for loop for better performance
  results <- lapply(time_points, function(t) {
    if(debug) cat(sprintf("\nProcessing time point %d/%d\n", t, t_end))
    time_start <- Sys.time()
    
    # Preallocate result matrices with the right dimensions
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
    
    # Process predictions in one step for all rules to avoid redundant processing
    current_g_preds <- process_g_preds(g_preds_processed, t, n_ids, py$J, gbound, debug)
    current_g_preds_bin <- process_g_preds(g_preds_bin_processed, t, n_ids, py$J, gbound, debug)
    current_c_preds <- get_c_preds(C_preds_processed, t, n_ids, gbound)
    current_y_preds <- get_y_preds(initial_model_for_Y, t, n_ids, ybound, debug)
    
    # Create treatment probability lists only once
    current_g_preds_list <- lapply(1:py$J, function(j) matrix(current_g_preds[,j], ncol=1))
    current_g_preds_bin_list <- lapply(1:py$J, function(j) matrix(current_g_preds_bin[,j], ncol=1))
    
    # Extract data efficiently by avoiding loops where possible
    Y <- initial_model_for_Y_data$Y
    C <- initial_model_for_Y_data$C
    
    # Vectorize censoring status calculation - one operation instead of multiple checks
    is_censored <- Y == -1 | is.na(Y) | C == 1
    valid_rows <- !is_censored
    
    # Preallocate rule predictions matrix - one allocation instead of growing
    Qs <- matrix(NA_real_, nrow=n_ids, ncol=n_rules) 
    colnames(Qs) <- names(tmle_rules)
    
    # Process all rules using the cached predictions
    for(i in seq_along(tmle_rules)) {
      rule <- names(tmle_rules)[i]
      
      # Get the cached predictions for this rule
      lstm_preds <- all_lstm_preds[[rule]]
      
      # Process predictions for each rule
      if(is.null(lstm_preds)) {
        # Use vectorized assignment for default case
        Qs[,i] <- mean(Y[valid_rows], na.rm=TRUE)
      } else {
        # Get time-specific predictions
        t_preds <- lstm_preds[[min(t + 1, length(lstm_preds))]]
        
        if(is.null(t_preds)) {
          # Use vectorized assignment for default case
          Qs[,i] <- mean(Y[valid_rows], na.rm=TRUE)
        } else {
          # Ensure proper dimensions with vectorized operations
          t_preds <- rep(t_preds, length.out=n_ids)
          # Bound values in one operation
          Qs[,i] <- pmin(pmax(t_preds, ybound[1]), ybound[2])
        }
      }
    }
    
    # Process initial predictions to ensure proper format
    initial_preds <- matrix(current_y_preds, nrow=n_ids)
    
    # Create QAW matrix efficiently
    QAW <- cbind(QA = initial_preds, Qs)
    colnames(QAW) <- c("QA", names(tmle_rules))
    
    # Apply bounds in one vectorized operation instead of multiple checks
    QAW <- pmin(pmax(QAW, ybound[1]), ybound[2])
    
    # Process treatment predictions in one step
    # Optimize g_matrix creation
    g_matrix <- if(is.list(current_g_preds_list)) {
      # Pre-allocate matrix with correct dimensions
      g_mat <- matrix(0, nrow=n_ids, ncol=ncol(treatments[[min(t + 1, length(treatments))]]))
      
      # Fill matrix efficiently by column
      for(j in seq_len(ncol(g_mat))) {
        if(j <= length(current_g_preds_list) && !is.null(current_g_preds_list[[j]])) {
          pred <- matrix(current_g_preds_list[[j]], nrow=n_ids)
          g_mat[,j] <- if(ncol(pred) > 1) pred[,1] else pred
        } else {
          g_mat[,j] <- rep(1/ncol(g_mat), n_ids)
        }
      }
      g_mat
    } else if(is.matrix(current_g_preds_list)) {
      # Efficiently handle matrix format
      if(nrow(current_g_preds_list) != n_ids) {
        matrix(rep(current_g_preds_list, length.out=n_ids*ncol(current_g_preds_list)), 
               ncol=ncol(current_g_preds_list))
      } else {
        current_g_preds_list 
      }
    } else {
      # Default uniform probabilities
      matrix(1/ncol(treatments[[min(t + 1, length(treatments))]]), 
             nrow=n_ids, ncol=ncol(treatments[[min(t + 1, length(treatments))]]))
    }
    
    # Get current treatment and rules
    current_obs_treatment <- treatments[[min(t + 1, length(treatments))]]
    current_obs_rules <- obs.rules[[min(t, length(obs.rules))]]
    
    # Create clever covariates in one step - preallocate for efficiency
    clever_covariates <- matrix(0, nrow=n_ids, ncol=ncol(current_obs_rules))
    is_censored_adj <- rep(is_censored, length.out=n_ids)
    
    # Vectorized operation for all rules
    for(i in seq_len(ncol(current_obs_rules))) {
      clever_covariates[,i] <- current_obs_rules[,i] * (!is_censored_adj)
    }
    
    # Calculate censoring-adjusted weights efficiently
    weights <- matrix(0, nrow=n_ids, ncol=ncol(current_obs_rules))
    
    # Calculate censoring matrix once instead of in the loop
    C_matrix <- matrix(rep(current_c_preds, ncol(g_matrix)), 
                       nrow=nrow(current_c_preds),
                       ncol=ncol(g_matrix))
    
    # Joint probability calculation - one operation instead of multiple
    probs <- g_matrix * (1 - C_matrix)
    bounded_probs <- pmin(pmax(probs, gbound[1]), gbound[2])
    
    # Calculate weights for all rules at once
    for(i in seq_len(ncol(current_obs_rules))) {
      valid_idx <- clever_covariates[,i] > 0
      if(any(valid_idx)) {
        # Calculate treatment probabilities for all valid rows at once
        treatment_probs <- rowSums(current_obs_treatment[valid_idx,] * bounded_probs[valid_idx,], na.rm=TRUE)
        treatment_probs[treatment_probs < gbound[1]] <- gbound[1]
        
        # IPCW weights
        cens_weights <- 1 / (1 - C_matrix[valid_idx,1])
        weights[valid_idx,i] <- cens_weights / treatment_probs
        
        # Optimize trimming and normalization
        rule_weights <- weights[valid_idx,i]
        if(length(rule_weights) > 0) {
          # Calculate quantile once and use for all
          max_weight <- quantile(rule_weights, 0.99, na.rm=TRUE)
          weights[valid_idx,i] <- pmin(rule_weights, max_weight) / 
            sum(pmin(rule_weights, max_weight), na.rm=TRUE)
        }
      }
    }
    
    # Preallocate modeling components
    updated_models <- vector("list", ncol(clever_covariates))
    
    # Optimize GLM fitting - only run when sufficient data
    for(i in seq_len(ncol(clever_covariates))) {
      # Create model data efficiently - single data.frame creation
      model_data <- data.frame(
        y = if(t < t_end) QAW[,"QA"] else Y,
        offset = qlogis(pmax(pmin(QAW[,i+1], 0.9999), 0.0001)),
        weights = weights[,i]
      )
      
      # Filter valid rows in one operation 
      valid_rows <- complete.cases(model_data) &
        is.finite(model_data$y) &
        is.finite(model_data$offset) &
        is.finite(model_data$weights) &
        model_data$y != -1 &
        model_data$weights > 0 &
        !is.infinite(qlogis(model_data$y))
      
      # Only fit model if sufficient data
      if(sum(valid_rows) > 10) {
        # Subset data once
        model_data <- model_data[valid_rows, , drop=FALSE]
        
        # Only fit if we have data with non-zero weights
        if(nrow(model_data) > 0 && any(model_data$weights > 0)) {
          # Optimize GLM fit with limited iterations
          updated_models[[i]] <- tryCatch({
            glm(
              y ~ 1 + offset(offset),
              weights = weights,
              family = quasibinomial(),
              data = model_data,
              control = list(maxit = 25)  # Limit iterations for speed
            )
          }, error = function(e) {
            # Return NULL on error rather than stopping
            if(debug) cat("\nGLM error:", e$message)
            NULL
          })
        }
      }
    }
    
    # Generate Qstar predictions efficiently
    # Pre-allocate results
    Qstar <- matrix(NA_real_, nrow=n_ids, ncol=length(updated_models))
    
    # Fill with predictions - use faster approach for NULL models
    for(i in seq_along(updated_models)) {
      if(is.null(updated_models[[i]])) {
        Qstar[,i] <- mean(Y[valid_rows], na.rm=TRUE)
      } else {
        # Get predictions directly
        preds <- predict(updated_models[[i]], type="response", newdata=NULL)
        # Expand to proper length if needed
        Qstar[,i] <- rep(preds, length.out=n_ids)
      }
    }
    
    # Set column names once
    if(ncol(Qstar) == ncol(current_obs_rules)) {
      colnames(Qstar) <- colnames(current_obs_rules)
    }
    
    # Create multinomial TMLE contrast
    tmle_contrast <- list(
      "Qs" = Qs,
      "QAW" = QAW,
      "clever_covariates" = clever_covariates,
      "weights" = weights,
      "updated_model_for_Y" = updated_models,
      "Qstar" = Qstar,
      "epsilon" = sapply(updated_models, function(mod) {
        if(is.null(mod)) 0 else tryCatch(coef(mod)[1], error=function(e) 0)
      }),
      "Qstar_gcomp" = QAW[,-1],
      "Y" = Y,
      "ID" = initial_model_for_Y_data$ID
    )
    
    # For binary case - process separately using current_g_preds_bin
    # Similar processing as above but with binary treatment predictions
    g_matrix_bin <- if(is.list(current_g_preds_bin_list)) {
      g_mat <- matrix(0, nrow=n_ids, ncol=ncol(current_obs_treatment))
      for(j in seq_len(ncol(g_mat))) {
        if(j <= length(current_g_preds_bin_list) && !is.null(current_g_preds_bin_list[[j]])) {
          pred <- matrix(current_g_preds_bin_list[[j]], nrow=n_ids)
          g_mat[,j] <- if(ncol(pred) > 1) pred[,1] else pred
        } else {
          g_mat[,j] <- rep(1/ncol(g_mat), n_ids)
        }
      }
      g_mat
    } else if(is.matrix(current_g_preds_bin_list)) {
      if(nrow(current_g_preds_bin_list) != n_ids) {
        matrix(rep(current_g_preds_bin_list, length.out=n_ids*ncol(current_g_preds_bin_list)), 
               ncol=ncol(current_g_preds_bin_list))
      } else {
        current_g_preds_bin_list 
      }
    } else {
      matrix(1/ncol(current_obs_treatment), nrow=n_ids, ncol=ncol(current_obs_treatment))
    }
    
    # Calculate IPTW for both models at once
    tryCatch({
      # Multinomial IPTW
      tmle_contrast$Qstar_iptw <- calculate_iptw(current_g_preds, current_obs_rules, 
                                               tmle_contrast$Qstar, n_rules, gbound, debug)
      
      # Binary IPTW
      binary_iptw <- calculate_iptw(current_g_preds_bin, current_obs_rules,
                                   tmle_contrast$Qstar, n_rules, gbound, debug)
    }, error = function(e) {
      if(debug) log_iptw_error(e, current_g_preds, current_obs_rules)
      tmle_contrast$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
      binary_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
    })
    
    # Create binary TMLE contrast - copy from multinomial and update relevant parts
    tmle_contrast_bin <- tmle_contrast
    tmle_contrast_bin$Qstar_iptw <- binary_iptw
    
    if(debug) {
      time_end <- Sys.time()
      cat(sprintf("\nTime point %d completed in %.2f s\n", 
                  t, as.numeric(difftime(time_end, time_start, units="secs"))))
    }
    
    # Return both models in a single list to reduce memory copying
    list(multinomial = tmle_contrast, binary = tmle_contrast_bin)
  })
  
  # Restructure results once at the end
  tmle_contrasts <- vector("list", t_end)
  tmle_contrasts_bin <- vector("list", t_end)
  
  for(t in 1:t_end) {
    # Use direct assignment instead of copying
    tmle_contrasts[[t]] <- results[[t]]$multinomial
    tmle_contrasts_bin[[t]] <- results[[t]]$binary
  }
  
  # Only do final debug output if needed
  if(debug) {
    cat("\nFinal time point processing summary:\n")
    # Include only essential summary metrics
    for(t in 1:t_end) {
      cat("\nTime point", t, "summary:")
      cat("\nTMLE estimates:", colMeans(tmle_contrasts[[t]]$Qstar, na.rm=TRUE))
      cat("\nIPTW estimates:", tmle_contrasts[[t]]$Qstar_iptw)
      cat("\nObserved Y mean:", mean(tmle_contrasts[[t]]$Y, na.rm=TRUE))
    }
  }
  
  return(list("multinomial" = tmle_contrasts, "binary" = tmle_contrasts_bin))
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

process_g_preds <- function(preds_processed, t, n_ids, J, gbound, debug) {
  if(!is.null(preds_processed) && t <= length(preds_processed)) {
    preds <- preds_processed[[t]]
    
    if(is.null(preds)) {
      if(debug) cat("No predictions for time", t, "using uniform\n")
      return(matrix(1/J, nrow=n_ids, ncol=J))
    }
    
    # Debug info about what we're processing
    if(debug) {
      cat("Processing predictions for time", t, "\n")
      cat("Prediction class:", class(preds), "\n")
      if(is.matrix(preds)) {
        cat("Matrix dimensions:", paste(dim(preds), collapse="x"), "\n")
      } else {
        cat("Length:", length(preds), "\n")
      }
    }
    
    # Ensure we have a numeric matrix
    if(!is.matrix(preds)) {
      if(debug) cat("Converting predictions to matrix\n")
      tryCatch({
        if(is.data.frame(preds)) {
          preds <- as.matrix(preds)
          mode(preds) <- "numeric"
        } else if(is.list(preds)) {
          numeric_values <- unlist(lapply(preds, function(x) {
            if(is.numeric(x)) return(x)
            as.numeric(as.character(x))
          }))
          preds <- matrix(numeric_values, ncol=ncol(preds))
        } else {
          preds <- as.numeric(as.character(preds))
          preds <- matrix(preds, ncol=J)
        }
      }, error = function(e) {
        if(debug) cat("Error converting to matrix:", e$message, "\n")
        return(matrix(1/J, nrow=n_ids, ncol=J))
      })
    }
    
    # Check if conversion was successful
    if(!is.matrix(preds) || !is.numeric(preds)) {
      if(debug) cat("Conversion failed, using uniform probs\n")
      return(matrix(1/J, nrow=n_ids, ncol=J))
    }
    
    # IMPORTANT FIX: Use actual column count from the matrix if available
    actual_J <- ncol(preds)
    if(actual_J > 0 && actual_J != J) {
      if(debug) cat("Using actual column count:", actual_J, "instead of specified J:", J, "\n")
      J <- actual_J  # Use the actual column count from the data
    }
    
    # Rest of the function remains the same...
    # Handle dimension mismatches
    if(ncol(preds) != J) {
      if(debug) cat("Column count mismatch:", ncol(preds), "vs", J, "\n")
      if(ncol(preds) < J) {
        # Add columns if needed
        preds <- cbind(preds, matrix(1/J, nrow=nrow(preds), ncol=J-ncol(preds)))
      } else {
        # Truncate if too many columns
        preds <- preds[, 1:J, drop=FALSE]
      }
    }
    
    # Adjust number of rows if needed
    if(nrow(preds) != n_ids) {
      if(debug) cat("Row count mismatch:", nrow(preds), "vs", n_ids, "\n")
      if(nrow(preds) < n_ids) {
        # Repeat rows to match n_ids
        repeats <- ceiling(n_ids / nrow(preds))
        preds <- preds[rep(1:nrow(preds), repeats), , drop=FALSE]
        preds <- preds[1:n_ids, , drop=FALSE]
      } else {
        # Truncate if too many rows
        preds <- preds[1:n_ids, , drop=FALSE]
      }
    }
    
    # Replace NAs with uniform probabilities
    na_indices <- is.na(preds)
    if(any(na_indices)) {
      if(debug) cat("Replacing", sum(na_indices), "NAs with uniform values\n")
      preds[na_indices] <- 1/J
    }
    
    # Normalize rows to sum to 1
    if(debug) cat("Normalizing probabilities\n")
    preds <- t(apply(preds, 1, function(row) {
      if(any(!is.finite(row)) || sum(row) == 0) {
        return(rep(1/J, J))
      }
      bounded <- pmin(pmax(row, gbound[1]), gbound[2])
      bounded / sum(bounded)
    }))
    
    return(preds)
  } else {
    if(debug) cat("No predictions available for time", t, "using uniform\n") 
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
  # Pre-allocate result vector for efficiency
  iptw_means <- numeric(n_rules)
  
  # Process all rules in one go if possible
  for(rule_idx in 1:n_rules) {
    # Vectorized valid index calculation
    valid_idx <- !is.na(rules[,rule_idx]) & rules[,rule_idx] == 1
    
    if(any(valid_idx)) {
      # Get outcomes for this rule efficiently
      outcomes <- predict_Qstar[,rule_idx]
      
      # Get probabilities efficiently
      rule_probs <- g_preds[valid_idx, min(rule_idx, ncol(g_preds))]
      rule_probs <- pmin(pmax(rule_probs, gbound[1]), gbound[2])
      
      # Calculate weights with vector operations
      marginal_prob <- mean(valid_idx, na.rm=TRUE)
      
      # Vectorized weight calculation
      weights <- rep(0, length(valid_idx))
      weights[valid_idx] <- marginal_prob / rule_probs
      
      # Efficient weight trimming
      max_weight <- quantile(weights[valid_idx], 0.95, na.rm=TRUE)
      weights <- pmin(weights, max_weight)
      
      # Extract vectors efficiently
      valid_weights <- weights[valid_idx]
      valid_outcomes <- outcomes[valid_idx]
      
      # Verify lengths and compute weighted mean
      if(length(valid_weights) > 0 && length(valid_outcomes) > 0) {
        if(length(valid_weights) != length(valid_outcomes)) {
          # Trim to common length
          min_len <- min(length(valid_weights), length(valid_outcomes))
          valid_weights <- valid_weights[1:min_len]
          valid_outcomes <- valid_outcomes[1:min_len]
        }
        
        # Normalize weights once
        valid_weights <- valid_weights / sum(valid_weights, na.rm=TRUE)
        
        # Calculate weighted mean efficiently
        iptw_means[rule_idx] <- sum(valid_outcomes * valid_weights, na.rm=TRUE)
      } else {
        iptw_means[rule_idx] <- mean(predict_Qstar[,rule_idx], na.rm=TRUE)
      }
    } else {
      iptw_means[rule_idx] <- mean(predict_Qstar[,rule_idx], na.rm=TRUE)
    }
  }
  
  # Return result as matrix
  matrix(iptw_means, nrow=1)
}

log_iptw_error <- function(e, g_preds, rules) {
  cat("Error calculating IPTW:\n")
  cat(conditionMessage(e), "\n")
  cat("Dimensions:\n")
  cat("g_preds:", paste(dim(g_preds), collapse=" x "), "\n") 
  cat("rules:", paste(dim(rules), collapse=" x "), "\n")
}

# Optimized version of getTMLELongLSTM function from tmle_fns_lstm.R

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded,
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size,
                            current_t, output_dir, debug=FALSE) {
  
  # Minimize debug logging
  if(debug) cat("\nStarting getTMLELongLSTM for time", current_t)
  
  # Get dimensions once
  n_ids <- nrow(obs.rules)
  n_rules <- length(tmle_rules)
  
  # Extract data efficiently by avoiding loops where possible
  Y <- initial_model_for_Y_data$Y
  C <- initial_model_for_Y_data$C
  
  # Vectorize censoring status calculation - one operation instead of multiple checks
  is_censored <- Y == -1 | is.na(Y) | C == 1
  valid_rows <- !is_censored
  
  # Preallocate rule predictions matrix - one allocation instead of growing
  Qs <- matrix(NA_real_, nrow=n_ids, ncol=n_rules) 
  colnames(Qs) <- names(tmle_rules)
  
  # Use efficient rule processing by combining rule operations
  # Precompute shifted data for all rules at once to avoid redundant processing
  shifted_data_list <- lapply(names(tmle_rules), function(rule) {
    # Get rule-specific treatments using existing functions
    switch(rule,
           "static" = static_mtp_lstm(initial_model_for_Y_data),
           "dynamic" = dynamic_mtp_lstm(initial_model_for_Y_data),
           "stochastic" = stochastic_mtp_lstm(initial_model_for_Y_data))
  })
  names(shifted_data_list) <- names(tmle_rules)
  
  # Create rule-specific datasets once efficiently
  # Use pre-computed ID mapping to avoid redundant lookups
  id_mapping_master <- match(initial_model_for_Y_data$ID, unique(initial_model_for_Y_data$ID))
  
  rule_data_list <- lapply(names(tmle_rules), function(rule) {
    # Start with shared data structure - avoid duplicating large objects
    rule_data <- initial_model_for_Y_data
    
    # Set treatment columns efficiently using pre-computed mapping
    shifted_data <- shifted_data_list[[rule]]
    id_mapping <- match(rule_data$ID, shifted_data$ID)
    
    # Vectorized assignment for all time points - single operation per time point
    for(t in 0:t_end) {
      col_name <- paste0("A.", t)
      rule_data[[col_name]] <- shifted_data$A0[id_mapping]
    }
    
    # Base A column also needs to be set
    rule_data$A <- rule_data[[paste0("A.", 0)]]
    
    rule_data
  })
  names(rule_data_list) <- names(tmle_rules)
  
  # Add A columns to covariates once - don't repeat this operation
  tmle_covars_Y_with_A <- unique(c(
    tmle_covars_Y,
    grep("^A\\.", colnames(rule_data_list[[1]]), value=TRUE),
    "A"
  ))
  
  # Use batch processing for all rules
  # First time, run with batch_models=TRUE to cache model
  if (exists("first_lstm_call", envir = .GlobalEnv) && first_lstm_call) {
    # Run LSTM for the first rule to cache model (only runs the Python script once)
    rule <- names(tmle_rules)[1]
    lstm_preds <- lstm(
      data = rule_data_list[[rule]],
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
      debug = FALSE,  # Disable debug for better performance
      batch_models = TRUE  # Enable model caching
    )
    # Set first_lstm_call to FALSE so we don't do this again
    assign("first_lstm_call", FALSE, envir = .GlobalEnv)
  } else if (!exists("first_lstm_call", envir = .GlobalEnv)) {
    # Initialize first_lstm_call if it doesn't exist
    assign("first_lstm_call", TRUE, envir = .GlobalEnv)
  }
  
  # Process all rules in batch
  all_lstm_preds <- lstm(
    data = NULL,  # Not used in batch mode
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
    debug = FALSE,
    batch_models = TRUE,
    batch_rules = rule_data_list  # Pass all rules at once
  )
  
  # Process predictions for all rules
  for(i in seq_along(tmle_rules)) {
    rule <- names(tmle_rules)[i]
    
    # Get predictions for this rule from batch results
    lstm_preds <- all_lstm_preds[[rule]]
    
    # Process predictions efficiently
    if(is.null(lstm_preds)) {
      # Use vectorized assignment for default case
      Qs[,i] <- mean(Y[valid_rows], na.rm=TRUE)
    } else {
      # Get time-specific predictions
      t_preds <- lstm_preds[[min(current_t + 1, length(lstm_preds))]]
      
      if(is.null(t_preds)) {
        # Use vectorized assignment for default case
        Qs[,i] <- mean(Y[valid_rows], na.rm=TRUE)
      } else {
        # Ensure proper dimensions with vectorized operations
        t_preds <- rep(t_preds, length.out=n_ids)
        # Bound values in one operation
        Qs[,i] <- pmin(pmax(t_preds, ybound[1]), ybound[2])
      }
    }
  }
  
  # Process initial predictions to ensure proper format
  initial_preds <- matrix(initial_model_for_Y_preds, nrow=n_ids)
  
  # Create QAW matrix efficiently
  QAW <- cbind(QA = initial_preds, Qs)
  colnames(QAW) <- c("QA", names(tmle_rules))
  
  # Apply bounds in one vectorized operation instead of multiple checks
  QAW <- pmin(pmax(QAW, ybound[1]), ybound[2])
  
  # Process treatment predictions in one step
  # Optimize g_matrix creation
  g_matrix <- if(is.list(g_preds_bounded)) {
    # Pre-allocate matrix with correct dimensions
    g_mat <- matrix(0, nrow=n_ids, ncol=ncol(obs.treatment))
    
    # Fill matrix efficiently by column
    for(j in seq_len(ncol(obs.treatment))) {
      if(j <= length(g_preds_bounded) && !is.null(g_preds_bounded[[j]])) {
        pred <- matrix(g_preds_bounded[[j]], nrow=n_ids)
        g_mat[,j] <- if(ncol(pred) > 1) pred[,1] else pred
      } else {
        g_mat[,j] <- rep(1/ncol(obs.treatment), n_ids)
      }
    }
    g_mat
  } else if(is.matrix(g_preds_bounded)) {
    # Efficiently handle matrix format
    if(nrow(g_preds_bounded) != n_ids) {
      matrix(rep(g_preds_bounded, length.out=n_ids*ncol(g_preds_bounded)), 
             ncol=ncol(g_preds_bounded))
    } else {
      g_preds_bounded 
    }
  } else {
    # Default uniform probabilities
    matrix(1/ncol(obs.treatment), nrow=n_ids, ncol=ncol(obs.treatment))
  }
  
  # Create clever covariates in one step - preallocate for efficiency
  clever_covariates <- matrix(0, nrow=n_ids, ncol=ncol(obs.rules))
  is_censored_adj <- rep(is_censored, length.out=n_ids)
  
  # Vectorized operation for all rules
  for(i in seq_len(ncol(obs.rules))) {
    clever_covariates[,i] <- obs.rules[,i] * (!is_censored_adj)
  }
  
  # Calculate censoring-adjusted weights efficiently
  weights <- matrix(0, nrow=n_ids, ncol=ncol(obs.rules))
  
  # Calculate censoring matrix once instead of in the loop
  C_matrix <- matrix(rep(C_preds_bounded, ncol(g_matrix)), 
                     nrow=nrow(C_preds_bounded),
                     ncol=ncol(g_matrix))
  
  # Joint probability calculation - one operation instead of multiple
  probs <- g_matrix * (1 - C_matrix)
  bounded_probs <- pmin(pmax(probs, gbound[1]), gbound[2])
  
  # Calculate weights for all rules at once
  for(i in seq_len(ncol(obs.rules))) {
    valid_idx <- clever_covariates[,i] > 0
    if(any(valid_idx)) {
      # Calculate treatment probabilities for all valid rows at once
      treatment_probs <- rowSums(obs.treatment[valid_idx,] * bounded_probs[valid_idx,], na.rm=TRUE)
      treatment_probs[treatment_probs < gbound[1]] <- gbound[1]
      
      # IPCW weights
      cens_weights <- 1 / (1 - C_matrix[valid_idx,1])
      weights[valid_idx,i] <- cens_weights / treatment_probs
      
      # Optimize trimming and normalization
      rule_weights <- weights[valid_idx,i]
      if(length(rule_weights) > 0) {
        # Calculate quantile once and use for all
        max_weight <- quantile(rule_weights, 0.99, na.rm=TRUE)
        weights[valid_idx,i] <- pmin(rule_weights, max_weight) / 
          sum(pmin(rule_weights, max_weight), na.rm=TRUE)
      }
    }
  }
  
  # Preallocate modeling components
  updated_models <- vector("list", ncol(clever_covariates))
  
  # Optimize GLM fitting - only run when sufficient data
  for(i in seq_len(ncol(clever_covariates))) {
    # Create model data efficiently - single data.frame creation
    model_data <- data.frame(
      y = if(current_t < t_end) QAW[,"QA"] else Y,
      offset = qlogis(pmax(pmin(QAW[,i+1], 0.9999), 0.0001)),
      weights = weights[,i]
    )
    
    # Pre-transform y values to avoid qlogis warnings
    # Only valid values between 0 and 1 should be passed to qlogis
    model_data$y_bounded <- pmin(pmax(model_data$y, 0.0001), 0.9999)
    
    # Filter valid rows in one operation 
    valid_rows <- complete.cases(model_data) &
      is.finite(model_data$y) &
      is.finite(model_data$offset) &
      is.finite(model_data$weights) &
      model_data$y != -1 &
      model_data$weights > 0
    
    # Only fit model if sufficient data
    if(sum(valid_rows) > 10) {
      # Subset data once
      model_data <- model_data[valid_rows, , drop=FALSE]
      
      # Only fit if we have data with non-zero weights
      if(nrow(model_data) > 0 && any(model_data$weights > 0)) {
        # Optimize GLM fit with limited iterations
        # Use the bounded y values to avoid qlogis warnings
        updated_models[[i]] <- tryCatch({
          glm(
            y_bounded ~ 1 + offset(offset),
            weights = weights,
            family = quasibinomial(),
            data = model_data,
            control = list(maxit = 25)  # Limit iterations for speed
          )
        }, error = function(e) {
          # Return NULL on error rather than stopping
          if(debug) cat("\nGLM error:", e$message)
          NULL
        })
      }
    }
  }
  
  # Generate Qstar predictions efficiently
  # Pre-allocate results
  Qstar <- matrix(NA_real_, nrow=n_ids, ncol=length(updated_models))
  
  # Fill with predictions - use faster approach for NULL models
  for(i in seq_along(updated_models)) {
    if(is.null(updated_models[[i]])) {
      # Ensure we have a valid default value, especially for time point t_end
      default_val <- mean(Y[valid_rows], na.rm=TRUE)
      if(is.na(default_val) || !is.finite(default_val)) {
        default_val <- 0.5  # Use a reasonable default if no valid data
      }
      Qstar[,i] <- default_val
    } else {
      # Get predictions directly
      preds <- predict(updated_models[[i]], type="response", newdata=NULL)
      # Ensure predictions are valid
      if(length(preds) == 0 || any(is.na(preds)) || any(!is.finite(preds))) {
        preds <- rep(0.5, length(preds))  # Use reasonable defaults for invalid predictions
      }
      # Expand to proper length if needed
      Qstar[,i] <- rep(preds, length.out=n_ids)
    }
  }
  
  # Set column names once
  if(ncol(Qstar) == ncol(obs.rules)) {
    colnames(Qstar) <- colnames(obs.rules)
  }
  
  # Calculate IPTW estimates with vectorized operations
  Qstar_iptw <- matrix(sapply(1:ncol(clever_covariates), function(i) {
    valid_idx <- clever_covariates[,i] > 0 & !is_censored_adj
    if(any(valid_idx)) {
      w <- weights[valid_idx,i]
      y <- Y[valid_idx]
      # Check lengths match
      if(length(w) == length(y)) {
        # Use weighted.mean with non-NA values
        valid_ys <- !is.na(y) & is.finite(y) & y != -1
        w_clean <- w[valid_ys]
        y_clean <- y[valid_ys]
        if(length(w_clean) > 0 && sum(w_clean) > 0) {
          weighted.mean(y_clean, w_clean, na.rm=TRUE) 
        } else {
          val <- mean(Y[valid_rows], na.rm=TRUE)
          if(is.na(val) || !is.finite(val)) val <- 0.5  # Fallback for last time point
          val
        }
      } else {
        val <- mean(Y[valid_rows], na.rm=TRUE)
        if(is.na(val) || !is.finite(val)) val <- 0.5  # Fallback for last time point
        val
      }
    } else {
      val <- mean(Y[valid_rows], na.rm=TRUE)
      if(is.na(val) || !is.finite(val)) val <- 0.5  # Fallback for last time point
      val
    }
  }), nrow=1)
  colnames(Qstar_iptw) <- colnames(obs.rules)
  
  # Calculate G-computation estimates directly
  Qstar_gcomp <- matrix(QAW[,-1], ncol=ncol(obs.rules))
  colnames(Qstar_gcomp) <- colnames(obs.rules)
  
  # Get epsilons efficiently
  epsilon <- sapply(updated_models, function(mod) {
    if(is.null(mod)) {
      0 
    } else {
      tryCatch(coef(mod)[1], error=function(e) 0)
    }
  })
  
  # Return the result list - minimal copies
  list(
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
  )
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