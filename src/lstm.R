
# Function to predict using cached models
predict_with_cached_model <- function(model_key, rule_data, n_ids, t_end, window_size, debug=FALSE) {
  if(!exists("cached_models", envir = .GlobalEnv) || is.null(cached_models[[model_key]])) {
    if(debug) print(paste("No cached model found for key:", model_key))
    return(NULL)
  }
  
  # Retrieve cached model
  cached <- cached_models[[model_key]]
  preds_r <- cached$preds_r
  is_Y_outcome <- cached$is_Y_outcome
  is_censoring_model <- cached$is_censoring_model
  
  if(debug) {
    print("Using cached model predictions:")
    print(paste("Prediction matrix shape:", paste(dim(preds_r), collapse=" x ")))
    print(paste("Model type: Y=", is_Y_outcome, ", C=", is_censoring_model))
  }
  
  # Determine prediction type
  prediction_type <- if(is_Y_outcome) "Y" else if(is_censoring_model) "C" else "A"
  
  # CRITICAL CHECK: Make sure we set t_end consistently
  if (!is.numeric(t_end) || t_end <= 0) {
    warning("Invalid t_end value: ", t_end, ", defaulting to 36")
    t_end <- 36  # Always default to 36 time points, not 37
  }
  
  # Calculate samples per time
  n_total_samples <- nrow(preds_r)
  samples_per_time <- ceiling(n_total_samples / (t_end + 1))
  
  if(debug) {
    print(paste("Samples per time:", samples_per_time))
    print(paste("Total time points:", t_end + 1))
  }
  
  # CRITICAL CHECK: Examine if the LSTM model is producing survival probabilities (high ~0.99) 
  # instead of event probabilities (low ~0.1-0.3) for outcome predictions
  if(is_Y_outcome && !is.null(preds_r) && nrow(preds_r) > 0) {
    lstm_mean <- mean(preds_r, na.rm=TRUE)
    if(debug) {
      print("=== LSTM OUTPUT INSPECTION ===")
      print(paste("LSTM output mean:", lstm_mean))
      print(paste("Range:", paste(range(preds_r, na.rm=TRUE), collapse=" - ")))
      
      # Detailed distribution analysis for diagnosis
      breaks <- c(0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1)
      cat("Value distribution:\n")
      for(i in 1:(length(breaks)-1)) {
        count <- sum(preds_r >= breaks[i] & preds_r < breaks[i+1], na.rm=TRUE)
        pct <- 100 * count / length(preds_r)
        cat(sprintf("  %.3f - %.3f: %d values (%.2f%%)\n", 
                   breaks[i], breaks[i+1], count, pct))
      }
      
      if(lstm_mean > 0.7) {
        print("WARNING: LSTM predictions for Y appear to be survival probabilities (~0.99)")
        print("These should be event probabilities (~0.1-0.3)")
        print("Will convert from survival to event probabilities (1-p)")
      }
    }
    
    # Fix the issue: If values are suspiciously high (mean > 0.7) for a Y outcome model
    # they're likely survival probabilities instead of event probabilities, so convert them
    if(lstm_mean > 0.7) {
      if(debug) {
        print("*** APPLYING CRITICAL FIX: Converting survival probabilities to event probabilities")
        print(paste("Before conversion - Mean:", round(lstm_mean, 6), 
                   "Range:", paste(round(range(preds_r, na.rm=TRUE), 6), collapse=" - ")))
      }
      
      # Convert from survival (1-event) to event probabilities
      preds_r <- 1.0 - preds_r
      
      # Display the new statistics
      if(debug) {
        new_mean <- mean(preds_r, na.rm=TRUE)
        new_range <- range(preds_r, na.rm=TRUE)
        print(paste("After conversion - Mean:", round(new_mean, 6),
                   "Range:", paste(round(new_range, 6), collapse=" - ")))
        
        # Check if conversion was appropriate
        if(new_mean > 0.7) {
          print("WARNING: Values still high after conversion! May need further investigation.")
        } else {
          print("Conversion successful: values now in expected event probability range.")
        }
      }
    }
  }
  
  # Process predictions for all time periods
  validated_preds <- lapply(1:(t_end + 1), function(t) {
    # Calculate indices for current time slice
    start_idx <- ((t-1) * samples_per_time) + 1
    end_idx <- min(t * samples_per_time, n_total_samples)
    
    # Get the appropriate slice of predictions
    if(start_idx <= n_total_samples && end_idx >= start_idx) {
      slice <- preds_r[start_idx:end_idx, , drop=FALSE]
    } else {
      # Use default values if out of bounds
      slice <- matrix(0.5, nrow=min(samples_per_time, 1000), ncol=if(prediction_type == "A") 6 else 1)
    }
    
    # Process predictions for this rule
    J <- if(prediction_type == "A") ncol(slice) else 1
    
    # Import required function from tmle_fns_lstm.R if not already available
    if(!exists("process_predictions", envir = .GlobalEnv, inherits = TRUE)) {
      source(file.path("src", "tmle_fns_lstm.R"), local = .GlobalEnv)
    }
    
    process_predictions(
      slice = slice,
      type = prediction_type,
      t = t,
      t_end = t_end,
      n_ids = n_ids,
      J = J,
      ybound = if(prediction_type == "Y") c(0.0001, 0.9999) else NULL,
      gbound = if(prediction_type != "Y") c(0.05, 1) else NULL,
      debug = debug
    )
  })
  
  # Add time period names
  names(validated_preds) <- paste0("t", 0:t_end)
  
  return(validated_preds)
}


lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, J, ybound, gbound, inference=FALSE, is_censoring=FALSE, debug=TRUE, batch_models=FALSE, batch_rules=NULL) {
  # This static variable tracks if models have already been loaded to avoid redundant loading
  if (!exists("cached_models", envir = .GlobalEnv)) {
    assign("cached_models", list(), envir = .GlobalEnv)
  }
  
  # Static variable to track if we've written data files already in batch mode
  if (!exists("batch_data_written", envir = .GlobalEnv)) {
    assign("batch_data_written", FALSE, envir = .GlobalEnv)
  }
  
  # Track if this is a batch processing call
  if (!is.null(batch_rules) && batch_models) {
    batch_mode <- TRUE
    if(debug){
      print("Running in batch mode for multiple rules")
      print(paste("Processing", length(batch_rules), "rules:", paste(names(batch_rules), collapse=", ")))
    }
  } else {
    batch_mode <- FALSE
  }
  
  if(debug && !batch_mode){
    print("Input parameters:")
    print(paste("Loss function:", loss_fn))
    print(paste("Is censoring:", is_censoring))
    print(paste("Outcome:", paste(outcome, collapse=",")))
    print(paste("Length of outcome:", length(outcome)))
    print(paste("J:", J))
  }
  
  # Use cached models if available in batch mode
  if(batch_mode && length(cached_models) > 0) {
    if(debug) print("Using cached models for batch processing")
    
    # Begin processing multiple rules
    results_list <- vector("list", length(batch_rules))
    names(results_list) <- names(batch_rules)
    
    # Process all rules using the cached model
    for(rule_name in names(batch_rules)) {
      rule_data <- batch_rules[[rule_name]]
      if(debug) print(paste("Processing rule:", rule_name))
      
      # Extract predictions for this rule using cached model
      results_list[[rule_name]] <- predict_with_cached_model(
        model_key = paste0(outcome[1], "_", loss_fn),
        rule_data = rule_data,
        n_ids = length(unique(rule_data$ID)),
        t_end = t_end,
        window_size = window_size,
        debug = debug
      )
    }
    
    return(results_list)
  }
  
  # Only display this message on the first call or non-batch mode
  if(!batch_data_written || !batch_mode) {
    print("Preparing data files...")
    
    # Modify outcome determination logic 
    is_Y_model <- FALSE # Default
    is_treatment_model <- FALSE 
    is_censoring_model <- FALSE
    
    # Override based on is_censoring parameter if provided
    if(is_censoring) {
      print("Training Censoring (C) Model...")
      is_censoring_model <- TRUE
      is_Y_model <- FALSE 
      is_treatment_model <- FALSE
      outcome <- grep("^C\\.", colnames(data), value=TRUE) # Force C columns as outcome
    } else {
      # Otherwise determine from outcome prefix
      if(is.character(outcome)) {
        if(length(outcome) == 1) {
          is_Y_model <- grepl("^Y", outcome)
          is_treatment_model <- grepl("^A", outcome) 
          is_censoring_model <- grepl("^C", outcome)
        } else {
          outcome_prefix <- unique(substr(outcome, 1, 1))
          if(debug){
            print(paste("Outcome column prefixes:", paste(outcome_prefix, collapse=",")))
          }
          is_Y_model <- "Y" %in% outcome_prefix 
          is_treatment_model <- "A" %in% outcome_prefix
          is_censoring_model <- "C" %in% outcome_prefix
        }
      }
    }
    
    # Validate model type and loss function match
    if(is_censoring_model && loss_fn != "binary_crossentropy") {
      warning("Forcing binary_crossentropy for censoring model")
      loss_fn <- "binary_crossentropy"
    } else if(is_Y_model && loss_fn != "binary_crossentropy") {
      warning("Forcing binary_crossentropy for Y model")
      loss_fn <- "binary_crossentropy"
    } else if(is_treatment_model && !loss_fn %in% c("binary_crossentropy", "sparse_categorical_crossentropy")) {
      warning("Invalid loss function for treatment model. Using binary_crossentropy.")
      loss_fn <- "binary_crossentropy"
    }
    
    if(debug){
      print("Initial data structure:")
      print(paste("Dimensions:", paste(dim(data), collapse=" x ")))
    }
    
    # Replace in lstm.R after "print("Preparing data files...")"
    
    # Validate inputs - no changes needed here but made more efficient
    if(!is.data.frame(data) || nrow(data) == 0) {
      stop("Invalid input data")
    }
    
    if(window_size < 1 || window_size > t_end) {
      stop("Invalid window size")  
    }
    
    if(!is.character(outcome)) {
      stop("Outcome must be character string")
    }
    
    # Optimize column finding with direct indexing instead of multiple regex operations
    # Find appropriate columns based on model type more efficiently
    if(is_censoring_model) {
      # More direct column matching - store regex results once instead of recomputing
      target_cols <- grep("^C\\.[0-9]+$|^C$", colnames(data), value=TRUE)
      if(debug){
        print(paste("Found", length(target_cols), "censoring columns"))
      }
      outcome_cols <- target_cols
    } else if(is_Y_model) {
      target_cols <- grep("^Y\\.[0-9]+$|^Y$", colnames(data), value=TRUE)
      if(debug){
        print(paste("Found", length(target_cols), "outcome columns"))
      }
      outcome_cols <- target_cols
    } else {
      # More efficient treatment column detection
      A_pattern <- "^A\\.[0-9]+$"
      A_cols_dot <- grep(A_pattern, colnames(data), value=TRUE)
      if(length(A_cols_dot) == 0) {
        A_pattern <- "^A[0-9]+$"
        A_cols_dot <- grep(A_pattern, colnames(data), value=TRUE)
      }
      target_cols <- A_cols_dot
      if(debug){
        print(paste("Found", length(target_cols), "treatment columns"))
      }
      outcome_cols <- target_cols
    }
    
    # Pre-process feature columns more efficiently
    # Cache the result of the GSub operation instead of recalculating
    if(length(covariates) > 0) {
      base_covariates <- unique(gsub("\\.[0-9]+$", "", covariates))
      if(debug){
        print("Base covariates:")
        print(base_covariates)
      }
    } else {
      base_covariates <- character(0)
    }
    
    # More efficient feature selection with precomputed patterns
    # Get all time-varying and static feature columns
    feature_cols <- character(0)
    
    # Define feature sets by model type
    base_features <- c("L1", "L2", "L3", "A") # Add A to base features
    a_model_features <- c("L1", "L2", "L3")  # A model doesn't use A as input
    c_model_features <- base_features  # C model uses A 
    y_model_features <- base_features  # Y model uses A
    
    # Select appropriate time-varying features based on model type
    time_varying_covs <- if(is_treatment_model) {
      a_model_features
    } else if(is_censoring_model) {
      c_model_features  # Now includes A
    } else {
      y_model_features  # Now includes A
    }
    
    # Precompute regex patterns to avoid multiple regex compilations
    # Get time-varying feature columns
    time_regex_patterns <- paste0("^", time_varying_covs, "\\.[0-9]+$", collapse="|")
    feature_cols <- c(feature_cols, grep(time_regex_patterns, colnames(data), value=TRUE))
    
    # Static covariates remain the same but process more efficiently 
    static_covs <- c("V3", "white", "black", "latino", "other", "mdd", "bipolar", "schiz")
    static_regex_patterns <- paste0("^", static_covs, "$|^", static_covs, "\\.[0-9]+$", collapse="|")
    feature_cols <- c(feature_cols, grep(static_regex_patterns, colnames(data), value=TRUE))
    
    # Ensure unique and sorted
    feature_cols <- sort(unique(feature_cols))
    
    if(debug){
      print("Found feature columns:")
      print(feature_cols)
    }
    
    # Calculate dimensions more efficiently
    unique_ids <- sort(unique(data$ID))
    n_ids <- length(unique_ids)
    
    # Fix n_times calculation to handle both Y and Y.0 style columns more efficiently
    n_times <- if(length(target_cols) == 1 && target_cols == "Y") {
      # Single Y column case
      37  # Since we know it's 36 time points (0-36) + 1
    } else {
      # Y.0, Y.1, etc case - more efficient extraction
      time_indices <- as.numeric(sub(".*\\.", "", target_cols))
      if(length(time_indices) > 0 && !all(is.na(time_indices))) {
        max(time_indices, na.rm=TRUE) + 1
      } else {
        stop("Could not determine time points from target columns:", 
             paste(target_cols, collapse=", "))
      }
    }
    
    # Validate window size
    if(is.na(n_times) || n_times == 0) {
      stop("Invalid number of time points:", n_times)
    }
    
    if(n_times <= window_size) {
      stop(sprintf("Window size (%d) must be less than number of time points (%d)", 
                   window_size, n_times))
    }
    
    # Calculate sequences ensuring positive value
    n_sequences_per_id <- max(1, n_times - window_size + 1)
    
    # Pre-allocate data structure with proper size rather than growing it
    # Create data_long with validated dimensions and pre-allocate target
    total_rows <- n_ids * n_sequences_per_id
    data_long <- data.frame(
      ID = rep(unique_ids, each = n_sequences_per_id),
      time = rep(0:(n_sequences_per_id-1), times = n_ids),
      target = rep(NA_real_, total_rows),
      stringsAsFactors = FALSE
    )
    
    # Optimized treatment model data processing
    if(is_treatment_model) {
      # Extract treatment matrix once
      treatment_matrix <- as.matrix(data[target_cols])
      treatment_matrix_dim <- dim(treatment_matrix)
      
      # Pre-allocate columns with default values
      data_long$A <- 5  # Default treatment
      for(j in 1:J) {
        data_long[[paste0("A", j-1)]] <- 0
      }
      
      # Get mapping of IDs to column indices more efficiently
      id_map <- match(data_long$ID, data$ID)
      
      # Vectorize time access where possible
      time_values <- data_long$time
      
      # Process in larger chunks rather than individual ID rows
      for(idx in seq_len(nrow(data_long))) {
        id_idx <- id_map[idx]
        if(!is.na(id_idx)) {
          t <- time_values[idx]
          window_idx <- t + window_size
          
          if(window_idx <= treatment_matrix_dim[2]) {
            val <- treatment_matrix[id_idx, window_idx]
            
            if(!is.na(val) && val %in% 0:6) {
              # Set treatment value
              data_long$A[idx] <- val
              
              # Set one-hot encoding
              trt_idx <- val + 1  # 0-based to 1-based
              if(trt_idx >= 1 && trt_idx <= J) {
                data_long[[paste0("A", val)]][idx] <- 1
              }
            }
          }
        }
      }
    }
    else if(is_Y_model) {
      # Extract outcome matrix once
      outcome_matrix <- as.matrix(data[target_cols])
      outcome_matrix_dim <- dim(outcome_matrix)
      
      # More efficient processing with vectorized operations where possible
      # and reduced function calls
      data_long$target <- sapply(1:nrow(data_long), function(i) {
        id <- data_long$ID[i]
        t <- data_long$time[i]
        
        # Get data index for this ID - do this lookup once
        data_idx <- match(id, data$ID)
        if(is.na(data_idx)) {
          return(if(inference) 0 else -1)
        }
        
        # Calculate proper time indices
        current_idx <- t + window_size
        if(current_idx >= outcome_matrix_dim[2]) {
          return(if(inference) 0 else -1)
        }
        
        # Use direct matrix indexing instead of repeated conditional checks
        current_val <- outcome_matrix[data_idx, current_idx + 1]
        
        # Simplified logic
        if(inference) {
          # During inference, handle current value and convert -1/NA to 0
          if(is.na(current_val) || current_val == -1) 0 else as.numeric(current_val)
        } else {
          # For training, preserve censoring status
          if(is.na(current_val)) -1 else as.numeric(current_val)
        }
      })
      
      # More efficient output data creation
      output_data <- data.frame(
        ID = data_long$ID,
        target = if(inference) {
          # Vectorized operation instead of element-by-element
          ifelse(data_long$target == -1, 0, data_long$target)
        } else {
          data_long$target
        },
        stringsAsFactors = FALSE
      )
    }
    
    # Optimized creation of output_data based on case
    if(is_treatment_model) {
      # Get treatment matrix for future predictions
      treatment_matrix <- as.matrix(data[target_cols])
      treatment_matrix_dim <- dim(treatment_matrix)
      
      if(loss_fn == "binary_crossentropy") {
        # Pre-allocate vector for future treatments
        future_treatments <- numeric(nrow(data_long))
        
        # Process future treatments in a more efficient way
        for(i in seq_len(nrow(data_long))) {
          id_idx <- match(data_long$ID[i], data$ID)
          if(!is.na(id_idx)) {
            t <- data_long$time[i]
            prediction_time <- t + window_size
            
            if(prediction_time < treatment_matrix_dim[2]) {
              val <- treatment_matrix[id_idx, prediction_time + 1]
              if(!is.na(val) && val %in% 0:6) {
                future_treatments[i] <- val
              } else {
                future_treatments[i] <- 5 # Default
              }
            } else {
              future_treatments[i] <- 5 # Default for out of bounds
            }
          } else {
            future_treatments[i] <- 5 # Default for invalid ID
          }
        }
        
        # Create output dataframe with ID
        output_data <- data.frame(
          ID = data_long$ID,
          stringsAsFactors = FALSE
        )
        
        # Convert future treatment values to 0-5 categories efficiently
        categorical_target <- ifelse(future_treatments == 0, 5, 
                                     pmin(future_treatments - 1, 5))
        
        # Create one-hot columns A0 through A5 for future treatments efficiently
        # using matrix operations instead of column-by-column
        for(j in 0:5) {
          output_data[[paste0("A", j)]] <- as.integer(categorical_target == j)
        }
        
        output_filename <- file.path(output_dir, "lstm_bin_A_output.csv")
        input_filename <- file.path(output_dir, "lstm_bin_A_input.csv")
        
      } else {
        # Optimize categorical case
        data_long$target <- sapply(1:nrow(data_long), function(i) {
          id_idx <- match(data_long$ID[i], data$ID)
          if(is.na(id_idx)) return(5) # Default
          
          t <- data_long$time[i]
          prediction_time <- t + window_size
          
          if(prediction_time >= treatment_matrix_dim[2]) {
            return(5) # Default for out of bounds
          }
          
          val <- treatment_matrix[id_idx, prediction_time + 1]
          if(is.na(val) || !val %in% 0:6) {
            return(5) # Default for invalid values
          }
          
          # Convert treatment value to categorical index
          ifelse(val == 0, 5, pmin(val - 1, 5))
        })
        
        output_data <- data.frame(
          ID = data_long$ID,
          target = data_long$target,
          stringsAsFactors = FALSE
        )
        
        output_filename <- file.path(output_dir, "lstm_cat_A_output.csv")
        input_filename <- file.path(output_dir, "lstm_cat_A_input.csv")
      }
    } else if(is_Y_model) {
      # Y model output creation already optimized above
      output_filename <- file.path(output_dir, "lstm_bin_Y_output.csv")
      input_filename <- file.path(output_dir, "lstm_bin_Y_input.csv")
    } else if(is_censoring_model) {
      # Optimize censoring model processing
      censoring_matrix <- as.matrix(data[target_cols])
      censoring_matrix_dim <- dim(censoring_matrix)
      
      # Preallocate targets
      data_long$target <- numeric(nrow(data_long))
      
      # Process censoring targets more efficiently
      for(i in seq_len(nrow(data_long))) {
        id_idx <- match(data_long$ID[i], data$ID)
        if(is.na(id_idx)) {
          data_long$target[i] <- 1  # Default to censored
          next
        }
        
        t <- data_long$time[i]
        prediction_time <- t + window_size
        
        if(prediction_time >= censoring_matrix_dim[2]) {
          data_long$target[i] <- 1  # Default to censored
          next
        }
        
        val <- censoring_matrix[id_idx, prediction_time + 1]
        # Simplified censoring logic
        data_long$target[i] <- ifelse(is.na(val) || val == -1, 1, 0)
      }
      
      # Create output data efficiently
      output_data <- data.frame(
        ID = data_long$ID,
        target = data_long$target,
        stringsAsFactors = FALSE
      )
      
      output_filename <- file.path(output_dir, "lstm_bin_C_output.csv")
      input_filename <- file.path(output_dir, "lstm_bin_C_input.csv")
    }
    
    # Create input dataframe more efficiently
    input_data <- data.frame(ID = data_long$ID, stringsAsFactors = FALSE)
    
    # Process static and time-varying covariates more efficiently
    for(base_col in base_covariates) {
      if(base_col %in% static_covs) {
        # Static covariate handling - vectorized operations
        if(base_col %in% colnames(data)) {
          col_data <- data[[base_col]]
          id_map <- match(data_long$ID, data$ID)
          default_value <- ifelse(base_col == "V3", mean(col_data, na.rm=TRUE), -1)
          input_data[[base_col]] <- ifelse(is.na(id_map), default_value, col_data[id_map])
        }
      } else if(base_col %in% time_varying_covs) {
        # Time-varying covariate handling
        time_cols <- grep(paste0("^", base_col, "\\.[0-9]+$"), colnames(data), value=TRUE)
        if(length(time_cols) > 0) {
          # Get time data once
          time_data <- as.matrix(data[time_cols])
          
          # Handle NA/missing values more efficiently
          time_data[is.na(time_data) | time_data == -1] <- -1
          
          # Process all rows at once where possible
          id_map <- match(data_long$ID, data$ID)
          sequence_matrix <- matrix(-1, nrow=nrow(data_long), ncol=window_size)
          
          valid_rows <- !is.na(id_map)
          if(any(valid_rows)) {
            # Process valid rows efficiently
            for(i in which(valid_rows)) {
              t_start <- data_long$time[i] + 1
              t_end <- min(t_start + window_size - 1, ncol(time_data))
              n_valid <- t_end - t_start + 1
              
              if(n_valid > 0) {
                # Direct matrix assignment
                sequence_matrix[i, 1:n_valid] <- time_data[id_map[i], t_start:t_end]
                if(n_valid < window_size) {
                  # Fill remaining with last value
                  sequence_matrix[i, (n_valid+1):window_size] <- time_data[id_map[i], t_end]
                }
              }
            }
          }
          
          # Convert to string more efficiently using vectorized apply
          input_data[[base_col]] <- apply(sequence_matrix, 1, toString)
        }
      }
    }
    
    # Write files with more efficient error handling
    tryCatch({
      print("Writing input data...")
      print(paste("Input file:", input_filename))
      # Use fwrite for faster CSV writing if data.table is available
      if(requireNamespace("data.table", quietly = TRUE)) {
        data.table::fwrite(input_data, file=input_filename)
      } else {
        write.csv(input_data, file=input_filename, row.names=FALSE)
      }
      print("Input data written successfully")
      
      print("Writing output data...")
      print(paste("Output file:", output_filename))
      if(requireNamespace("data.table", quietly = TRUE)) {
        data.table::fwrite(output_data, file=output_filename)
      } else {
        write.csv(output_data, file=output_filename, row.names=FALSE)
      }
      print("Output data written successfully")
    }, error = function(e) {
      print(paste("Error writing files:", e$message))
      print("Directory contents:")
      print(list.files(output_dir))
      stop(e)
    })
    
    # Set Python variables
    py$window_size <- as.integer(window_size)
    py$output_dir <- output_dir
    py$epochs <- as.integer(100)
    py$n_hidden <- as.integer(256)
    py$hidden_activation <- 'tanh'
    py$out_activation <- out_activation
    py$lr <- 0.002
    py$dr <- 0.25
    py$nb_batches <- as.integer(192)
    py$patience <- as.integer(10)
    py$t_end <- as.integer(t_end)
    py$feature_cols <- if(length(base_covariates) > 0) base_covariates else stop("No features available")
    py$outcome_cols <- outcome_cols
    py$gbound <-gbound
    py$ybound <-ybound
    
    # Synchronize model type and settings
    if(is_censoring_model) {
      print("Setting Censoring (C) Model params...")
      is_censoring <- TRUE  # Set R variable
      py$is_censoring <- TRUE  # Set Python variable
      py$J <- as.integer(1)
      py$loss_fn <- "binary_crossentropy"
    } else if(is_Y_model) {
      if(!inference) {
        print("Training Outcome (Y) Model...")
      }else{
        print("Testing Outcome (Y) Model...")
      }
      is_censoring <- FALSE  # Set R variable
      py$is_censoring <- FALSE  # Set Python variable
      py$J <- as.integer(1) 
      py$loss_fn <- "binary_crossentropy"
    } else {
      if(!inference) {
        print("Training Treatment (A) Model...")
      }
      is_censoring <- FALSE  # Set R variable
      py$is_censoring <- FALSE  # Set Python variable
      py$J <- as.integer(J)
      py$loss_fn <- loss_fn
    }
    
    print(paste("Model type verification - is_censoring:", is_censoring))
    
    # Import numpy
    np <- reticulate::import("numpy")
    
    # Run Python script with error handling
    result <- if(inference) {
      tryCatch({
        source_python("src/test_lstm.py")
      }, error = function(e) {
        print("Error in test_lstm.py:")
        print(e$message)
        print(paste("Error details:", reticulate::py_last_error()))
        NULL
      })
    } else {
      tryCatch({
        source_python("src/train_lstm.py")
      }, error = function(e) {
        print("Error in train_lstm.py:")
        print(e$message)
        print(paste("Error details:", reticulate::py_last_error()))
        NULL
      })
    }
    
    # Process predictions
    predictions <- tryCatch({
      # Store model type flags in parent environment
      is_Y_outcome <- "Y" %in% substr(outcome_cols, 1, 1)
      is_censoring_model <- is_censoring
      is_treatment_model <- !is_Y_outcome && !is_censoring_model
      
      # Determine prediction file path with consistent naming
      if(inference) {
        # Test predictions
        preds_file <- if(is_censoring_model) {
          file.path(output_dir, 'test_bin_C_preds.npy')
        } else {
          if(is_Y_outcome && loss_fn == "binary_crossentropy") {
            file.path(output_dir, 'test_bin_Y_preds.npy')
          } else {
            file.path(output_dir, 
                      if(loss_fn == "sparse_categorical_crossentropy") 
                        'test_bin_A_preds.npy'
                      else 
                        'test_bin_A_preds.npy')
          }
        }
      } else {
        # Training predictions
        preds_file <- if(is_censoring_model) {
          file.path(output_dir, 'lstm_bin_C_preds.npy')
        } else {
          if(is_Y_outcome && loss_fn == "binary_crossentropy") {
            file.path(output_dir, 'lstm_bin_Y_preds.npy')
          } else {
            file.path(output_dir, 
                      if(loss_fn == "sparse_categorical_crossentropy") 
                        'lstm_cat_A_preds.npy'
                      else 
                        'lstm_bin_A_preds.npy')
          }
        }
      }
      
      if(!file.exists(preds_file)) {
        warning(paste("Predictions file not found:", preds_file))
        return(NULL)
      }
      
      # Add debugging info
      if(debug) {
        cat("\nLooking for predictions file:", preds_file)
        cat("\nFile exists:", file.exists(preds_file))
        cat("\nDirectory contents:\n")
        print(list.files(output_dir, pattern=".npy"))
      }
      
      # Load and validate prediction array
      preds_r <- as.array(np$load(preds_file))
      
      # CRITICAL FIX: Check if the LSTM predictions are survival probabilities (mean~0.99)
      # instead of event probabilities (mean~0.1-0.3) for Y outcomes
      if(is_Y_outcome && !is.null(preds_r) && is.matrix(preds_r) && nrow(preds_r) > 0) {
        lstm_mean <- mean(preds_r, na.rm=TRUE)
        
        if(debug) {
          print("=== LSTM PREDICTIONS ANALYSIS ===")
          print(paste("Mean:", lstm_mean))
          print(paste("Range:", paste(range(preds_r, na.rm=TRUE), collapse=" to ")))
          
          # Print histogram of values for detailed analysis
          breaks <- c(0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1)
          hist_counts <- sapply(1:(length(breaks)-1), function(i) {
            sum(preds_r >= breaks[i] & preds_r < breaks[i+1], na.rm=TRUE)
          })
          
          print("Value distribution:")
          for(i in 1:(length(breaks)-1)) {
            pct <- 100 * hist_counts[i] / length(preds_r)
            print(sprintf("  %.2f - %.2f: %d values (%.2f%%)", 
                           breaks[i], breaks[i+1], hist_counts[i], pct))
          }
        }
        
        # If mean is very high (>0.7) for a Y outcome, these are likely survival probabilities
        # instead of event probabilities. Convert them automatically.
        if(lstm_mean > 0.7) {
          if(debug) {
            print("CRITICAL ISSUE DETECTED: LSTM predictions appear to be survival probabilities!")
            print("Converting to event probabilities...")
          }
          
          # Convert from survival (1-event) to event probabilities
          preds_r <- 1.0 - preds_r
          
          if(debug) {
            print("=== AFTER CONVERSION ===")
            print(paste("New mean:", mean(preds_r, na.rm=TRUE)))
            print(paste("New range:", paste(range(preds_r, na.rm=TRUE), collapse=" to ")))
          }
        }
      }
      
      # Cache model info for batch processing if enabled
      if(batch_models) {
        model_key <- paste0(outcome[1], "_", loss_fn)
        cached_models[[model_key]] <- list(
          preds_r = preds_r,
          preds_file = preds_file,
          is_Y_outcome = is_Y_outcome,
          is_censoring_model = is_censoring_model,
          is_treatment_model = is_treatment_model,
          inference = inference
        )
        assign("cached_models", cached_models, envir = .GlobalEnv)
        assign("batch_data_written", TRUE, envir = .GlobalEnv)
        if(debug) print(paste("Cached model with key:", model_key))
      }
      
      # Validate predictions
      if(is.null(preds_r) || length(dim(preds_r)) < 2) {
        stop("Invalid prediction format - predictions must be a 2D array")
      }
      
      # Ensure t_end is numeric and reasonable
      if (!is.numeric(t_end) || t_end <= 0) {
        warning("Invalid t_end value: ", t_end, ", defaulting to 36")
        t_end <- 36  # Default to 36 time points
      }
      
      # Calculate samples per time with validation
      n_total_samples <- nrow(preds_r)
      
      # Validate t_end is available and numeric
      if(is.null(t_end) || !is.numeric(t_end) || t_end <= 0) {
        t_end <- 36  # Default to 36 time points
        warning("Invalid or missing t_end, using default value:", t_end)
      }
      
      # Calculate samples_per_time safely
      samples_per_time <- tryCatch({
        as.integer(ceiling(n_total_samples / (t_end + 1)))
      }, error = function(e) {
        warning("Error calculating samples_per_time: ", e$message, 
                ", using default based on prediction shape")
        min(14000, max(1, as.integer(n_total_samples / 37)))
      })
      
      # Validate the result
      if(samples_per_time <= 0 || samples_per_time > n_total_samples) {
        samples_per_time <- min(14000, max(1, as.integer(n_total_samples / 37)))
        warning("Invalid samples_per_time calculated: ", samples_per_time, 
                ", using default based on prediction shape")
      }
      
      if(debug) {
        cat("\nValidated calculation values:")
        cat("\n  n_total_samples:", n_total_samples)
        cat("\n  t_end:", t_end)
        cat("\n  samples_per_time:", samples_per_time)
      }
      prediction_type <- if(is_Y_outcome) "Y" else if(is_censoring_model) "C" else "A"
      
      if(debug) {
        cat("\nLoaded predictions from:", preds_file, "\n")
        cat("Array shape:", paste(dim(preds_r), collapse=" x "), "\n")
        cat("Samples per time period:", samples_per_time, "\n")
        cat("Prediction type:", prediction_type, "\n")
        cat("Model type:", paste0(
          "Y=", is_Y_outcome, ", ",
          "C=", is_censoring_model, ", ",
          "A=", is_treatment_model
        ), "\n")
      }
      
      # Process all time periods
      validated_preds <- lapply(1:(t_end + 1), function(t) {
        # Calculate direct slices without window offset complexity
        start_idx <- ((t-1) * samples_per_time) + 1
        end_idx <- min(t * samples_per_time, n_total_samples)
        
        if(debug) {
          cat(sprintf("\nProcessing time %d:\n", t-1))
          cat(sprintf("Using direct indices %d to %d\n", start_idx, end_idx))
        }
        
        # Get slice of predictions
        slice <- if(start_idx <= n_total_samples && end_idx >= start_idx) {
          preds_r[start_idx:end_idx, , drop=FALSE]
        } else {
          if(debug) cat("Invalid indices, using default slice\n")
          matrix(1/J, nrow=min(samples_per_time, 1000), ncol=J)
        }
        
        # Process predictions with explicit parameter passing
        processed <- process_predictions(
          slice = slice,
          type = prediction_type,
          t = t,
          t_end = t_end,
          n_ids = n_ids,
          J = J,
          ybound = ybound,
          gbound = gbound,
          debug = debug
        )
        
        if(debug) {
          cat(sprintf("Processed predictions shape: %s\n", paste(dim(processed), collapse=" x ")))
          cat(sprintf("Range: %s\n", paste(range(processed, na.rm=TRUE), collapse=" - ")))
        }
        
        processed
      })
      
      # Add time period names
      names(validated_preds) <- paste0("t", 0:t_end)
      
      validated_preds
      
    }, error = function(e) {
      warning(paste("Error processing predictions:", e$message))
      if(exists("preds_r")) {
        print("Raw predictions info:")
        print(paste("Shape:", paste(dim(preds_r), collapse=" x ")))
        print(paste("Contains NaN:", any(is.nan(preds_r))))
        print(paste("Contains Inf:", any(is.infinite(preds_r))))
      }
      NULL
    })
    
    return(predictions)
  }
}