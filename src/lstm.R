lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, J, ybound, gbound, inference=FALSE, is_censoring=FALSE, debug=TRUE) {
  print("Initial data structure:")
  print(paste("Dimensions:", paste(dim(data), collapse=" x ")))
  
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
  
  # Find appropriate columns based on case (treatment or censoring)
  if(is_censoring) {
    target_cols <- grep("^C\\.[0-9]+$|^C$", colnames(data), value=TRUE)
    print(paste("Found", length(target_cols), "censoring columns"))
    print("Censoring columns:")
    print(target_cols)
  } else {
    # Original treatment column logic
    A_cols_dot <- grep("^A\\.[0-9]+$", colnames(data), value=TRUE)
    A_cols_plain <- grep("^A[0-9]+$", colnames(data), value=TRUE)
    target_cols <- if(length(A_cols_dot) > 0) A_cols_dot else A_cols_plain
    print(paste("Found", length(target_cols), "treatment columns"))
    print("Treatment columns:")
    print(target_cols)
  }
  
  # Pre-process feature columns to ensure they exist
  base_covariates <- unique(gsub("\\.[0-9]+$", "", covariates))
  print("Base covariates:")
  print(base_covariates)
  
  # Get all time-varying and static feature columns
  feature_cols <- c()
  
  # Get time-varying feature columns
  time_varying_covs <- c("L1", "L2", "L3", "C", "Y") 
  for(base_col in time_varying_covs) {
    time_cols <- grep(paste0("^", base_col, "\\.[0-9]+$"), 
                      colnames(data), value=TRUE)
    feature_cols <- c(feature_cols, time_cols)
  }
  
  
  # Predefined list of static covariates
  static_covs <- c("V3", "white", "black", "latino", "other", "mdd", "bipolar", "schiz")
  
  # Get static feature columns 
  for(base_col in static_covs) {
    # Get both base column and time point columns
    base_cols <- grep(paste0("^", base_col, "$|^", base_col, "\\.[0-9]+$"),
                      colnames(data), value=TRUE)
    feature_cols <- c(feature_cols, base_cols)
  }
  
  # Ensure unique and sorted
  feature_cols <- sort(unique(feature_cols))
  
  print("Found feature columns:")
  print(feature_cols)
  
  n_ids <- nrow(data)
  n_times <- length(target_cols)
  # Adjust for overlapping windows and time periods
  n_sequences_per_id <- n_times - window_size + 1  # Add 1 for overlapping windows
  data_long <- data.frame(
    ID = rep(1:n_ids, each=n_sequences_per_id),
    time = rep(0:(n_sequences_per_id-1), times=n_ids)
  )
  
  # Set total sequences
  n_sequences <-  n_ids * n_sequences_per_id
  target_sequences <- n_sequences
  
  # Create base input data frame
  input_data <- data.frame(ID = data_long$ID)
  
  # Handle target creation based on case
  if(is_censoring) {
    target_matrix <- as.matrix(data[target_cols])
    data_long$target <- sapply(1:nrow(data_long), function(i) {
      id <- data_long$ID[i]
      t <- data_long$time[i]
      val <- target_matrix[id, t + window_size]
      if(is.na(val)) return(0)  # Default value for NAs
      as.numeric(val > 0)  # Ensure binary values
    })
  } else {
    treatment_matrix <- as.matrix(data[target_cols])
    data_long$A <- sapply(1:nrow(data_long), function(i) {
      id <- data_long$ID[i]
      t <- data_long$time[i]
      
      # Only access target if within valid range
      if(t + window_size >= length(target_cols)) {
        return(5)  # Default to most common class if beyond range
      }
      
      val <- treatment_matrix[id, t + 1]  # Note the +1 to handle 0-based indexing
      # Ensure valid treatment values 0-6
      if(is.na(val) || !val %in% 0:6) {
        return(5)  # Most common class
      }
      return(as.numeric(val))
    })
  }
  
  # Create output_data based on case - SINGLE TARGET MAPPING HERE
  if(is_censoring) {
    output_data <- data.frame(
      ID = data_long$ID,
      target = data_long$target
    )
  } else if(loss_fn == "sparse_categorical_crossentropy") {
    # Map treatment values (0-6) to target values (0-5) only once here
    output_data <- data.frame(
      ID = data_long$ID,
      target = ifelse(data_long$A == 0, 5, pmin(data_long$A - 1, 5))
    )
  } else {
    output_data <- data.frame(
      ID = data_long$ID,
      target = data_long$A
    )
  }
  
  # Ensure target values are properly bounded
  output_data$target <- as.integer(output_data$target)
  if(!is_censoring) {
    output_data$target[output_data$target < 0 | output_data$target > 5] <- 5
  }
  
  # Validate treatment values
  print("Treatment value distribution after processing:")
  print(table(output_data$target, useNA="ifany"))
  
  # Add explicit binary/continuous type definitions
  binary_covs <- c("L2", "L3", "C", "Y", "white", "black", "latino", "other", "mdd", "bipolar", "schiz")
  continuous_covs <- c("V3", "L1")
  
  for(base_col in base_covariates) {
    if(base_col %in% static_covs) {
      # Handle static covariates more robustly
      col_data <- data[[base_col]] 
      if(base_col == "V3") {
        # V3 is continuous
        vals <- as.numeric(col_data)
        vals[is.na(vals)] <- -1  # Only replace NAs
        input_data[[base_col]] <- rep(vals, each=n_sequences_per_id)
      } else {
        # Binary covariates - maintain original values
        vals <- as.numeric(col_data)  # Keep as numeric without logical conversion
        vals[is.na(vals)] <- -1  # Only replace NAs
        input_data[[base_col]] <- rep(vals, each=n_sequences_per_id)
      }
    } else if(base_col %in% c("L1", "L2", "L3", "C", "Y")) {
      # Handle time-varying covariates
      time_cols <- grep(paste0("^", base_col, "\\.[0-9]+$"), colnames(data), value=TRUE)
      if(length(time_cols) > 0) {
        time_data <- as.matrix(data[time_cols])
        # Convert to numeric for processing but store result as character
        time_data <- matrix(as.numeric(time_data), nrow=nrow(time_data))
        time_data[is.na(time_data)] <- -1  # Only replace NAs
        
        sequence_matrix <- matrix(NA, nrow=nrow(data_long), ncol=window_size)
        
        # Process each sequence
        for(i in 1:nrow(data_long)) {
          id <- data_long$ID[i]
          t <- data_long$time[i]
          window_indices <- (t + 1):(t + window_size)
          
          if(max(window_indices) <= ncol(time_data)) {
            sequence_matrix[i,] <- time_data[id, window_indices]
          } else {
            valid_indices <- window_indices[window_indices <= ncol(time_data)]
            sequence_matrix[i, 1:length(valid_indices)] <- time_data[id, valid_indices]
            if(length(valid_indices) < window_size) {
              sequence_matrix[i, (length(valid_indices)+1):window_size] <- 
                time_data[id, tail(valid_indices, 1)]
            }
          }
        }
        
        # Store as character sequence
        input_data[[base_col]] <- apply(sequence_matrix, 1, function(x) {
          paste(as.character(x), collapse=",")
        })
      }
    }
  }
  
  # Validate processed data
  print("Validating processed data...")
  
  # Convert target columns to numeric
  if(exists("output_data")) {
    for(col in names(output_data)) {
      if(col != "ID") {
        output_data[[col]] <- as.numeric(as.character(output_data[[col]]))
      }
    }
  } else {
    stop("output_data not created - check target processing")
  }
  
  # Validate final data
  validate_data <- function(data, type="input") {
    # Check for NAs
    na_cols <- colnames(data)[apply(data, 2, anyNA)]
    if(length(na_cols) > 0) {
      warning(paste("Found NAs in", type, "data columns:", paste(na_cols, collapse=", ")))
      
      # Fix NAs only
      for(col in na_cols) {
        if(col == "ID") next
        
        # For sequences, replace only NAs  
        if(is.matrix(data[[col]])) {
          data[[col]][is.na(data[[col]])] <- -1
        } else if(is.character(data[[col]]) && grepl(",", data[[col]][1])) {
          # This is a sequence stored as string
          data[[col]] <- gsub("NA", "-1", data[[col]])
        } else {
          # For numeric columns, replace NA with appropriate default
          if(col %in% binary_covs) {
            data[[col]][is.na(data[[col]])] <- -1
          } else if(col == "target") {
            # For target, use most common class
            data[[col]][is.na(data[[col]])] <- 5
          } else {
            data[[col]][is.na(data[[col]])] <- -1
          }
        }
      }
    }
    
    # Validate types
    for(col in names(data)) {
      if(col == "ID") next
      
      if(col == "target") {
        # Ensure target is integers 0-5
        data[[col]] <- as.integer(data[[col]])
        data[[col]] <- pmin(pmax(data[[col]], 0), 5)
      } else if(col %in% c("L1", "L2", "L3", "C", "Y")) {
        # Skip sequence columns - keep as character
        next
      } else if(!grepl(",", data[[col]][1])) {
        # Convert only non-sequence columns to numeric
        data[[col]] <- as.numeric(data[[col]])
      }
    }
    
    return(data)
  }
  
  # Apply validation
  input_data <- validate_data(input_data, "input")
  output_data <- validate_data(output_data, "output")
  
  # Add after validation:
  print("Data validation summary:")
  print("Input data:")
  for(col in names(input_data)) {
    print(paste("Column:", col))
    if(is.character(input_data[[col]]) && grepl(",", input_data[[col]][1])) {
      # Sequence column
      print(paste("  Type: sequence"))
      print(paste("  Sample values:", paste(head(input_data[[col]], 3), collapse="; ")))
    } else {
      # Regular column 
      print(paste("  Type:", class(input_data[[col]])))
      print(paste("  Range:", paste(range(input_data[[col]]), collapse=" - ")))
      if(col != "ID") {
        print(paste("  Mean:", mean(input_data[[col]])))
      }
    }
  }
  
  print("\nOutput data:")
  print("Target distribution:")
  print(table(output_data$target))
  
  # Check time-varying features
  for(col in c("L1", "L2", "L3")) {
    if(col %in% names(input_data)) {
      # Sample first few rows
      sample_seqs <- head(input_data[[col]], 5)
      print(paste("Sample", col, "sequences:"))
      print(sample_seqs)
      
      # Check for all -1s
      all_neg_ones <- sapply(input_data[[col]], function(x) {
        all(as.numeric(strsplit(x, ",")[[1]]) == -1)
      })
      print(paste("Proportion of all -1 sequences in", col, ":", 
                  mean(all_neg_ones)))
    }
  }
  
  # Ensure proper data types
  input_data[] <- lapply(seq_along(input_data), function(i) {
    col <- names(input_data)[i]
    x <- input_data[[i]]
    if(col %in% c("L1", "L2", "L3", "C", "Y")) {
      # Preserve sequence format
      return(x)
    } else if(col == "ID") {
      return(x)  # Keep ID as is
    } else {
      # Convert static features to numeric
      return(as.numeric(as.character(x)))
    }
  })
  
  # Check static features
  for(col in static_covs) {
    if(col %in% names(input_data)) {
      print(paste("Summary of", col, ":"))
      print(summary(input_data[[col]]))
      print(paste("Proportion of -1 in", col, ":", 
                  mean(input_data[[col]] == -1)))
    }
  }
  
  # Check data consistency
  if(nrow(input_data) != nrow(output_data)) {
    stop("Input and output data have different numbers of rows")
  }
  
  if(!all(input_data$ID == output_data$ID)) {
    stop("ID mismatch between input and output data")
  }
  
  # Check sequence lengths
  sequence_lengths <- sapply(strsplit(input_data$L1, ","), length)
  if(!all(sequence_lengths == window_size)) {
    stop("Not all sequences have the correct window size")
  }
  
  print("Sample processed features:")
  for(col in names(input_data)) {
    if(col != "ID") {
      print(paste("Column:", col))
      print("First few values:")
      print(head(input_data[[col]]))
    }
  }
  # Print final dimensions and summaries
  print("Final data checks:")
  print(paste("Input data dimensions:", paste(dim(input_data), collapse=" x ")))
  print(paste("Output data dimensions:", paste(dim(output_data), collapse=" x ")))
  print("Sample of first few sequences:")
  print(head(input_data[c("L1", "L2", "L3")]))
  
  # Verify target values
  print("Target distribution:")
  print(table(output_data$target, useNA="ifany"))
  
  # Write data to CSV
  input_file <- paste0(output_dir, "input_data.csv")
  output_file <- paste0(output_dir, "output_data.csv")
  write.csv(input_data, input_file, row.names = FALSE)
  write.csv(output_data, output_file, row.names = FALSE)
  
  print("Final data dimensions:")
  print("Input data:")
  print(str(input_data))
  print("Output data:")
  print(str(output_data))
  
  # Extract outcome columns from the data
  if (is_censoring) {
    outcome_cols <- grep("^C\\.[0-9]+$|^C$", colnames(data), value=TRUE)
  } else {
    # For treatment or Y prediction
    if (is.character(outcome)) {
      outcome_cols <- outcome
    } else if (!is.null(colnames(outcome))) {
      outcome_cols <- colnames(outcome)
    } else {
      stop("Invalid outcome specification")
    }
  }
  
  # Set Python variables
  py$is_censoring <- is_censoring
  py$J <- as.integer(if(is_censoring) 1 else J)
  py$output_dir <- output_dir
  py$epochs <- as.integer(100)
  py$n_hidden <- as.integer(256)
  py$hidden_activation <- 'tanh'
  py$out_activation <- out_activation
  py$loss_fn <- loss_fn
  py$lr <- 0.001
  py$dr <- 0.5
  py$nb_batches <- as.integer(128)
  py$patience <- as.integer(2)
  py$t_end <- as.integer(t_end + 1)
  py$window_size <- as.integer(window_size)
  py$feature_cols <- if(length(base_covariates) > 0) base_covariates else stop("No features available")
  py$outcome_cols <- outcome_cols  # Pass outcome columns
  py$binary_features <- binary_covs
  py$continuous_features <- continuous_covs
  
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
  
  tryCatch({
    preds_file <- if(is_censoring) {
      paste0(output_dir, 'lstm_bin_C_preds.npy')
    } else {
      # Check if outcome is Y and has binary loss function
      is_Y_outcome <- any(grepl("^Y", outcome_cols))
      if(is_Y_outcome && loss_fn == "binary_crossentropy") {
        paste0(output_dir, 'lstm_bin_Y_preds.npy')
      } else {
        ifelse(loss_fn == "sparse_categorical_crossentropy",
               paste0(output_dir, 'lstm_cat_A_preds.npy'),
               paste0(output_dir, 'lstm_bin_A_preds.npy'))
      }
    }
    
    if(file.exists(preds_file)) {
      # Load and validate predictions
      preds <- np$load(preds_file)
      preds_r <- as.array(preds)
      
      # Determine prediction type
      is_Y_pred <- any(grepl("^Y", outcome_cols))
      
      if(debug) {
        cat("\nPrediction file loaded:", preds_file, "\n")
        cat("Raw dimensions:", paste(dim(preds_r), collapse=" x "), "\n")
        cat("Prediction type:", if(is_Y_pred) "Y" else if(is_censoring) "C" else "A", "\n")
      }
      
      # Calculate samples per time period
      n_total_samples <- nrow(preds_r)
      samples_per_time <- n_total_samples %/% (t_end + 1)
      
      # Function to extract time slice
      get_time_slice <- function(t) {
        start_idx <- ((t-1) * samples_per_time) + 1
        end_idx <- t * samples_per_time
        slice <- preds_r[start_idx:end_idx, , drop=FALSE]
        if(debug) {
          cat(sprintf("\nTime %d slice: %d to %d", t-1, start_idx, end_idx))
          cat("\nSlice dimensions:", paste(dim(slice), collapse=" x "), "\n")
        }
        return(slice)
      }
      
      # Process based on prediction type
      preds_list <- if(is_Y_pred) {
        # Process Y predictions
        cat("\nProcessing Y predictions...\n")
        lapply(1:(t_end + 1), function(t) {
          y_preds <- get_time_slice(t)
          if(!is.matrix(y_preds)) y_preds <- matrix(y_preds, ncol=1)
          
          # Ensure proper dimensions and bounds
          bounded_preds <- pmin(pmax(y_preds, ybound[1]), ybound[2])
          colnames(bounded_preds) <- "Y"
          
          if(debug) {
            cat(sprintf("\nTime %d Y predictions:\n", t-1))
            cat("Range:", paste(range(bounded_preds), collapse=" - "), "\n")
          }
          bounded_preds
        })
        
      } else if(is_censoring) {
        # Process censoring predictions
        cat("\nProcessing censoring predictions...\n")
        lapply(1:(t_end + 1), function(t) {
          c_preds <- get_time_slice(t)
          if(!is.matrix(c_preds)) c_preds <- matrix(c_preds, ncol=1)
          
          # Bound censoring probabilities
          bounded_preds <- pmin(pmax(c_preds, gbound[1]), gbound[2])
          colnames(bounded_preds) <- "C"
          
          if(debug) {
            cat(sprintf("\nTime %d censoring predictions:\n", t-1))
            cat("Range:", paste(range(bounded_preds), collapse=" - "), "\n")
          }
          bounded_preds
        })
      } else {
        # Process treatment predictions 
        cat("\nProcessing treatment predictions...\n")
        
        # Get predictions for each time point
        n_total_samples <- nrow(preds_r)  # 3584 samples
        samples_per_time <- n_total_samples %/% (t_end + 1)  # 3584/37 ≈ 96 samples per time
        
        # Reshape predictions to proper dimensions
        preds_list <- vector("list", t_end + 1)
        
        for(t in 1:(t_end + 1)) {
          # Get this time point's predictions
          start_idx <- ((t-1) * samples_per_time) + 1
          end_idx <- min(t * samples_per_time, n_total_samples)
          time_slice <- preds_r[start_idx:end_idx, , drop=FALSE]
          
          # Interpolate to get predictions for all patients
          new_preds <- matrix(0, nrow=n_ids, ncol=J)
          for(j in 1:J) {
            # Create sequence for interpolation
            x_old <- seq(0, 1, length.out=nrow(time_slice))
            x_new <- seq(0, 1, length.out=n_ids)
            
            # Interpolate probabilities
            new_preds[,j] <- approx(x_old, time_slice[,j], x_new)$y
          }
          
          # Normalize probabilities to ensure they sum to 1
          new_preds <- t(apply(new_preds, 1, function(row) {
            # Handle invalid values
            if(any(is.na(row)) || any(!is.finite(row))) {
              return(rep(1/J, J))
            }
            # Softmax normalization
            exp_row <- exp(row - max(row))
            exp_row / sum(exp_row)
          }))
          
          # Add column names
          colnames(new_preds) <- paste0("A", 1:J)
          preds_list[[t]] <- new_preds
          
          if(debug) {
            cat("\nPrediction summary for time", t-1, ":\n")
            cat("Dimensions:", paste(dim(new_preds), collapse=" x "), "\n")
            cat("Range:", paste(range(new_preds), collapse="-"), "\n")
            cat("Row sums:", paste(range(rowSums(new_preds)), collapse="-"), "\n")
          }
        }
        
        names(preds_list) <- paste0("t", 0:t_end)
        preds_list
      }
      
      # Final validation across all predictions
      validated_preds <- lapply(preds_list, function(mat) {
        # Handle any remaining invalid values
        mat[is.na(mat) | !is.finite(mat)] <- if(is_Y_pred) {
          mean(c(ybound[1], ybound[2]))
        } else if(is_censoring) {
          gbound[1]
        } else {
          1/J
        }
        
        # Final bounds check
        if(is_Y_pred) {
          pmin(pmax(mat, ybound[1]), ybound[2])
        } else if(is_censoring) {
          pmin(pmax(mat, gbound[1]), gbound[2])
        } else {
          # For treatment predictions, ensure proper probabilities
          t(apply(mat, 1, function(row) {
            bounded <- pmax(row, gbound[1])
            bounded / sum(bounded)
          }))
        }
      })
      
      if(debug) {
        cat("\nFinal validation summary:\n")
        cat("Number of time points:", length(validated_preds), "\n")
        cat("Dimensions at each time:", paste(dim(validated_preds[[1]]), collapse=" x "), "\n")
        ranges <- range(do.call(rbind, validated_preds))
        cat("Overall range:", paste(ranges, collapse=" - "), "\n")
      }
      
      return(validated_preds)
      
    } else {
      warning(paste("Predictions file not found:", preds_file))
      return(NULL)
    }
  }, error = function(e) {
    warning(paste("Error processing predictions:", e$message))
    print(paste("Predictions file:", preds_file))
    if(exists("preds")) {
      print("Raw predictions info:")
      print(paste("Class:", class(preds)))
      print(paste("Shape:", paste(dim(preds), collapse=" x ")))
      print(paste("Contains NaN:", any(is.nan(as.array(preds)))))
      print(paste("Contains Inf:", any(is.infinite(as.array(preds)))))
    }
    return(NULL)
  })
}