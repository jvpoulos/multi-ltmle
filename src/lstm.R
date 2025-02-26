lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, J, ybound, gbound, inference=FALSE, is_censoring=FALSE, debug=TRUE) {
  if(debug){
    print("Input parameters:")
    print(paste("Loss function:", loss_fn))
    print(paste("Is censoring:", is_censoring))
    print(paste("Outcome:", paste(outcome, collapse=",")))
    print(paste("Length of outcome:", length(outcome)))
    print(paste("J:", J))
  }
  
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

  # Validate inputs
  if(!is.data.frame(data) || nrow(data) == 0) {
    stop("Invalid input data")
  }
  
  if(window_size < 1 || window_size > t_end) {
    stop("Invalid window size")  
  }
  
  if(!is.character(outcome)) {
    stop("Outcome must be character string")
  }
  
  # Find appropriate columns based on model type
  if(is_censoring_model) {
    target_cols <- grep("^C\\.[0-9]+$|^C$", colnames(data), value=TRUE)
    if(debug){
      print(paste("Found", length(target_cols), "censoring columns"))
      print("Censoring columns:")
      print(target_cols)
    }
    outcome_cols <- target_cols
  } else if(is_Y_model) {
    target_cols <- grep("^Y\\.[0-9]+$|^Y$", colnames(data), value=TRUE)
    if(debug){
      print(paste("Found", length(target_cols), "outcome columns"))
      print("Outcome columns:")
      print(target_cols)
    }
    outcome_cols <- target_cols
  } else {
    # Treatment columns
    A_cols_dot <- grep("^A\\.[0-9]+$", colnames(data), value=TRUE)
    A_cols_plain <- grep("^A[0-9]+$", colnames(data), value=TRUE)
    target_cols <- if(length(A_cols_dot) > 0) A_cols_dot else A_cols_plain
    if(debug){
      print(paste("Found", length(target_cols), "treatment columns"))
      print("Treatment columns:")
      print(target_cols)
    }
    outcome_cols <- target_cols
  }
  
  # Pre-process feature columns
  base_covariates <- unique(gsub("\\.[0-9]+$", "", covariates))
  if(debug){
    print("Base covariates:")
    print(base_covariates)
  }

  # Get all time-varying and static feature columns
  feature_cols <- c()
  
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
  
  # Get time-varying feature columns
  for(base_col in time_varying_covs) {
    time_cols <- grep(paste0("^", base_col, "\\.[0-9]+$"), 
                      colnames(data), value=TRUE)
    feature_cols <- c(feature_cols, time_cols)
  }
  
  # Static covariates remain the same
  static_covs <- c("V3", "white", "black", "latino", "other", "mdd", "bipolar", "schiz")
  
  # Get static feature columns 
  for(base_col in static_covs) {
    base_cols <- grep(paste0("^", base_col, "$|^", base_col, "\\.[0-9]+$"),
                      colnames(data), value=TRUE)
    feature_cols <- c(feature_cols, base_cols)
  }
  
  # Ensure unique and sorted
  feature_cols <- sort(unique(feature_cols))
  
  if(debug){
    print("Found feature columns:")
    print(feature_cols)
  }
  
  # Calculate dimensions
  unique_ids <- sort(unique(data$ID))
  n_ids <- length(unique_ids)
  
  # Fix n_times calculation to handle both Y and Y.0 style columns
  n_times <- if(length(target_cols) == 1 && target_cols == "Y") {
    # Single Y column case
    37  # Since we know it's 36 time points (0-36) + 1
  } else {
    # Y.0, Y.1, etc case
    time_indices <- as.numeric(gsub(".*\\.", "", target_cols))
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
  
  # Create data_long with validated dimensions and pre-allocate target
  data_long <- data.frame(
    ID = rep(sort(unique(data$ID)), each = n_sequences_per_id),
    time = rep(0:(n_sequences_per_id-1), times = n_ids),
    target = NA  # Initialize target column upfront
  )
  
  if(is_treatment_model) {
    treatment_matrix <- as.matrix(data[target_cols])
    
    # Pre-allocate columns with default values
    data_long$A <- 5  # Default treatment
    for(j in 1:J) {
      data_long[[paste0("A", j-1)]] <- 0
    }
    
    # Pre-calculate mappings
    all_ids <- sort(unique(data$ID))
    
    # Print debug info
    if(debug) {
      cat("\nTreatment matrix info:")
      cat("\nDimensions:", paste(dim(treatment_matrix), collapse=" x "))
      cat("\nSample values:", paste(head(treatment_matrix[1,]), collapse=", "))
      cat("\nUnique treatments:", paste(sort(unique(as.vector(treatment_matrix))), collapse=", "))
    }
    
    # Process each ID directly
    for(id in all_ids) {
      # Get all rows for this ID
      id_mask <- data_long$ID == id
      id_rows <- which(id_mask)
      
      if(length(id_rows) > 0) {
        # Get data row for this ID
        data_idx <- match(id, data$ID)
        
        if(!is.na(data_idx)) {
          # Get all treatment values for this ID
          all_treatments <- treatment_matrix[data_idx,]
          
          # Process each time point
          for(i in seq_along(id_rows)) {
            t <- data_long$time[id_rows[i]]
            window_idx <- t + window_size
            
            if(window_idx <= length(all_treatments)) {
              val <- all_treatments[window_idx]
              
              if(!is.na(val) && val %in% 0:6) {
                # Set treatment value
                data_long$A[id_rows[i]] <- val
                
                # Set one-hot encoding
                trt_idx <- val + 1  # 0-based to 1-based
                if(trt_idx >= 1 && trt_idx <= J) {
                  data_long[[paste0("A", val)]][id_rows[i]] <- 1
                }
              }
            }
          }
        }
      }
    }
    
    if(debug) {
      cat("\nTreatment assignment results:")
      cat("\nFinal treatment distribution:\n")
      print(table(data_long$A, useNA="ifany"))
      cat("\nOne-hot encoding summary:\n")
      for(j in 0:5) {
        cat(paste0("A", j, " sum: ", sum(data_long[[paste0("A", j)]]), "\n"))
      }
    }
  }
  else if(is_Y_model) {
    outcome_matrix <- as.matrix(data[target_cols])
    data_long$target <- sapply(1:nrow(data_long), function(i) {
      id <- data_long$ID[i]
      t <- data_long$time[i]
      
      # Get data index for this ID
      data_idx <- match(id, data$ID)
      if(is.na(data_idx)) {
        if(inference) return(0) else return(-1)  # During inference, default to 0 not -1
      }
      
      # Calculate proper time indices
      current_idx <- t + window_size
      if(current_idx >= ncol(outcome_matrix)) {
        if(inference) return(0) else return(-1)  # During inference, default to 0 not -1
      }
      
      # For targeting step (inference), use current value 
      val <- if(inference) {
        # During inference we want to predict all values, no censoring
        current_val <- outcome_matrix[data_idx, current_idx + 1]
        if(is.na(current_val) || current_val == -1) 0 else as.numeric(current_val)
      } else {
        # For training, use next time point and preserve censoring
        future_idx <- current_idx + 1
        if(future_idx > ncol(outcome_matrix)) return(-1)
        outcome_matrix[data_idx, future_idx]
      }
      
      # During inference, convert -1 to 0 as we don't want any censoring
      if(inference && (is.na(val) || val == -1)) {
        return(0)
      }
      
      as.numeric(val)
    })
    
    # For Y model, output is binary outcome 
    output_data <- data.frame(
      ID = data_long$ID,
      target = if(inference) {
        # During inference, convert all -1 to 0
        ifelse(data_long$target == -1, 0, data_long$target)
      } else {
        data_long$target  # Keep censoring during training
      },
      stringsAsFactors = FALSE
    )
    
    if(debug) {
      cat("\nY model target distribution:")
      print(table(data_long$target))
      cat("\nCensored: ", sum(data_long$target == -1))
      cat("\nZeros: ", sum(data_long$target == 0))
      cat("\nOnes: ", sum(data_long$target == 1))
    }
  }
  
  # Create output_data based on case
  if(is_treatment_model) {
    # Get treatment matrix for future predictions
    treatment_matrix <- as.matrix(data[target_cols])
    
    if(loss_fn == "binary_crossentropy") {
      # First get future treatment values
      data_long$future_treatment <- sapply(1:nrow(data_long), function(i) {
        id <- data_long$ID[i]
        t <- data_long$time[i]
        
        # Get the treatment for the time step AFTER the window
        prediction_time <- t + window_size
        if(prediction_time >= ncol(treatment_matrix)) {
          return(5)  # Default for out of bounds
        }
        
        # Get the actual future treatment
        val <- treatment_matrix[id, prediction_time + 1]  # +1 to predict next step
        if(is.na(val) || !val %in% 0:6) {
          return(5)  # Default for invalid values
        }
        return(as.numeric(val))
      })
      
      # Create output dataframe with ID
      output_data <- data.frame(
        ID = data_long$ID,
        stringsAsFactors = FALSE
      )
      
      # Convert future treatment values to 0-5 categories
      categorical_target <- ifelse(data_long$future_treatment == 0, 5, 
                                   pmin(data_long$future_treatment - 1, 5))
      
      # Create one-hot columns A0 through A5 for future treatments
      for(j in 0:5) {
        col_name <- paste0("A", j)
        output_data[[col_name]] <- as.integer(categorical_target == j)
      }
      
      output_filename <- file.path(output_dir, "lstm_bin_A_output.csv")
      input_filename <- file.path(output_dir, "lstm_bin_A_input.csv")
      
    } else {
      # For categorical case, predict future treatment directly
      data_long$target <- sapply(1:nrow(data_long), function(i) {
        id <- data_long$ID[i]
        t <- data_long$time[i]
        
        # Get the treatment for the time step AFTER the window
        prediction_time <- t + window_size
        if(prediction_time >= ncol(treatment_matrix)) {
          return(5)  # Default for out of bounds
        }
        
        # Get the actual future treatment
        val <- treatment_matrix[id, prediction_time + 1]  # +1 to predict next step
        if(is.na(val) || !val %in% 0:6) {
          return(5)  # Default for invalid values
        }
        val <- ifelse(val == 0, 5, pmin(val - 1, 5))
        return(val)
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
    outcome_matrix <- as.matrix(data[target_cols])
    data_long$target <- sapply(1:nrow(data_long), function(i) {
      id <- data_long$ID[i]
      t <- data_long$time[i]
      
      # Get data index for this ID
      data_idx <- match(id, data$ID)
      if(is.na(data_idx)) {
        if(inference) return(0) else return(-1)  # During inference, default to 0 not -1
      }
      
      # Calculate proper time indices
      current_idx <- t + window_size
      if(current_idx >= ncol(outcome_matrix)) {
        if(inference) return(0) else return(-1)  # During inference, default to 0 not -1
      }
      
      # For targeting step (inference), use current value 
      val <- if(inference) {
        # During inference, handle current value and convert -1/NA to 0
        current_val <- outcome_matrix[data_idx, current_idx + 1]
        if(is.na(current_val) || current_val == -1) 0 else as.numeric(current_val)
      } else {
        # For training, preserve censoring status
        future_idx <- current_idx + 1
        if(future_idx > ncol(outcome_matrix)) return(-1)
        val <- outcome_matrix[data_idx, future_idx]
        if(is.na(val)) return(-1)
        val
      }
      
      as.numeric(val)
    })
    
    # For Y model, output is binary outcome 
    output_data <- data.frame(
      ID = data_long$ID,
      target = if(inference) {
        # During inference, ensure all values are 0/1
        ifelse(is.na(data_long$target) | data_long$target == -1, 0, data_long$target)
      } else {
        data_long$target  # Keep censoring during training
      },
      stringsAsFactors = FALSE
    )
    
    # Print debug information if needed
    if(debug) {
      cat("\nY model target distribution:")
      print(table(data_long$target))
      cat("\nCensored: ", sum(data_long$target == -1))
      cat("\nZeros: ", sum(data_long$target == 0))
      cat("\nOnes: ", sum(data_long$target == 1))
    }
    
    output_filename <- file.path(output_dir, "lstm_bin_Y_output.csv")
    input_filename <- file.path(output_dir, "lstm_bin_Y_input.csv")
  }else if(is_censoring_model) {
    # Align censoring predictions with future time steps
    censoring_matrix <- as.matrix(data[target_cols])
    # Initialize target as numeric to prevent logical type
    data_long$target <- as.numeric(NA)
    
    data_long$target <- sapply(1:nrow(data_long), function(i) {
      id <- data_long$ID[i]
      t <- data_long$time[i]
      
      # Get data index for this ID
      data_idx <- match(id, data$ID)
      if(is.na(data_idx)) {
        return(as.numeric(1))  # Default to censored
      }
      
      # Get the censoring status for the time step AFTER the window
      prediction_time <- t + window_size
      if(prediction_time >= ncol(censoring_matrix)) {
        return(as.numeric(1))  # Assume censored if beyond available data
      }
      
      # Get the actual future censoring status
      val <- censoring_matrix[data_idx, prediction_time + 1]  # +1 to predict next step
      
      # Handle censoring conversion explicitly
      if(is.na(val)) {
        return(as.numeric(1))  # Treat NA as censored
      } else if(val == -1) {
        return(as.numeric(1))  # Convert -1 to 1 (censored)
      } else {
        return(as.numeric(0))  # Not censored
      }
    })
    
    # Create output data with proper formatting 
    output_data <- data.frame(
      ID = data_long$ID,
      target = data_long$target,
      stringsAsFactors = FALSE
    )
    
    if(debug) {
      cat("\nDistribution summary:\n")
      if(is_treatment_model) {
        cat("\nTreatment distribution:")
        print(table(data_long$A, useNA="ifany"))
      } else if(is_censoring_model) {
        cat("\nCensoring target distribution:")
        print(table(output_data$target, useNA="ifany"))
        cat("\nCensored (1): ", sum(output_data$target == 1, na.rm=TRUE))
        cat("\nNot censored (0): ", sum(output_data$target == 0, na.rm=TRUE))
      } else {
        cat("\nOutcome distribution:")
        print(table(data_long$target))
        cat("\nCensored: ", sum(data_long$target == -1))
        cat("\nZeros: ", sum(data_long$target == 0))
        cat("\nOnes: ", sum(data_long$target == 1))
      }
    }
    
    output_filename <- file.path(output_dir, "lstm_bin_C_output.csv")
    input_filename <- file.path(output_dir, "lstm_bin_C_input.csv")
  }
  
  if(debug){
    print("Final output data dimensions:")
    print(dim(output_data))
    print("Names of output columns:")
    print(names(output_data))
  }
  
  # Add explicit binary/continuous type definitions
  binary_covs <- c("L2", "L3", "C", "Y", "white", "black", "latino", "other", "mdd", "bipolar", "schiz")
  continuous_covs <- c("V3", "L1")
  
  # Create input dataframe first
  input_data <- data.frame(ID = data_long$ID)
  
  for(base_col in base_covariates) {
    if(base_col %in% static_covs) {
      # Static covariate handling
      col_data <- data[[base_col]]
      id_map <- match(data_long$ID, data$ID)
      input_data[[base_col]] <- ifelse(
        is.na(id_map), 
        ifelse(base_col == "V3", mean(col_data, na.rm=TRUE), -1),
        col_data[id_map]
      )
    } else if(base_col %in% time_varying_covs) {
      # Time-varying covariate handling
      time_cols <- grep(paste0("^", base_col, "\\.[0-9]+$"), colnames(data), value=TRUE)
      if(length(time_cols) > 0) {
        time_data <- as.matrix(data[time_cols])
        time_data[is.na(time_data) | time_data == -1] <- -1  # Handle both NA and -1
        
        id_map <- match(data_long$ID, data$ID)
        sequence_matrix <- matrix(-1, nrow=nrow(data_long), ncol=window_size)
        
        valid_rows <- !is.na(id_map)
        if(any(valid_rows)) {
          for(i in which(valid_rows)) {
            t_start <- data_long$time[i] + 1
            t_end <- min(t_start + window_size - 1, ncol(time_data))
            n_valid <- t_end - t_start + 1
            
            if(n_valid > 0) {
              sequence_matrix[i, 1:n_valid] <- time_data[id_map[i], t_start:t_end]
              if(n_valid < window_size) {
                sequence_matrix[i, (n_valid+1):window_size] <- time_data[id_map[i], t_end]
              }
            }
          }
        }
        
        input_data[[base_col]] <- apply(sequence_matrix, 1, toString)
      }
    }
  }
  
  # Validate processed data
  print("Validating processed data...")
  
  # Add validation checks and debugging output
  if(debug) {
    print("Sample processed features:")
    for(col in names(input_data)) {
      if(col != "ID") {
        print(paste("Column:", col))
        print("First few values:")
        print(head(input_data[[col]]))
      }
    }
  }
  
  if(debug) {
    print("Verifying model data:")
    print(paste("Input rows:", nrow(input_data)))
    print(paste("Output rows:", nrow(output_data)))
    print(paste("Input file:", input_filename))
    print(paste("Output file:", output_filename))
  }
  
  # Write files with error handling
  tryCatch({
    print("Writing input data...")
    print(paste("Input file:", input_filename))
    write.csv(input_data, file=input_filename, row.names=FALSE)
    print("Input data written successfully")
    
    print("Writing output data...")
    print(paste("Output file:", output_filename))
    write.csv(output_data, file=output_filename, row.names=FALSE)
    print("Output data written successfully")
    
    if(debug){
      print("Directory contents after writing:")
      print(list.files(output_dir))
    }
  }, error = function(e) {
    print(paste("Error writing files:", e$message))
    print("Directory contents:")
    print(list.files(output_dir))
    stop(e)
  })
  
  # Set Python variables
  py$window_size <- as.integer(window_size)
  py$output_dir <- output_dir
  py$epochs <- as.integer(1) # 100
  py$n_hidden <- as.integer(256)
  py$hidden_activation <- 'tanh'
  py$out_activation <- out_activation
  py$lr <- 0.001
  py$dr <- 0.3
  py$nb_batches <- as.integer(256)
  py$patience <- as.integer(3)
  py$t_end <- as.integer(t_end + 1)
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