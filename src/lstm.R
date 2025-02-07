lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, J, ybound, gbound, inference=FALSE, is_censoring=FALSE, debug=TRUE) {
  # At start of lstm.R
  print("Input parameters:")
  print(paste("Loss function:", loss_fn))
  print(paste("Is censoring:", is_censoring))
  print(paste("Outcome:", paste(outcome, collapse=",")))
  print(paste("Length of outcome:", length(outcome)))
  print(paste("J:", J))
  
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
        print(paste("Outcome column prefixes:", paste(outcome_prefix, collapse=",")))
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
  
  # Find appropriate columns based on model type
  if(is_censoring_model) {
    target_cols <- grep("^C\\.[0-9]+$|^C$", colnames(data), value=TRUE)
    print(paste("Found", length(target_cols), "censoring columns"))
    print("Censoring columns:")
    print(target_cols)
    outcome_cols <- target_cols
  } else if(is_Y_model) {
    target_cols <- grep("^Y\\.[0-9]+$|^Y$", colnames(data), value=TRUE)
    print(paste("Found", length(target_cols), "outcome columns"))
    print("Outcome columns:")
    print(target_cols)
    outcome_cols <- target_cols
  } else {
    # Treatment columns
    A_cols_dot <- grep("^A\\.[0-9]+$", colnames(data), value=TRUE)
    A_cols_plain <- grep("^A[0-9]+$", colnames(data), value=TRUE)
    target_cols <- if(length(A_cols_dot) > 0) A_cols_dot else A_cols_plain
    print(paste("Found", length(target_cols), "treatment columns"))
    print("Treatment columns:")
    print(target_cols)
    outcome_cols <- target_cols
  }
  
  # Pre-process feature columns
  base_covariates <- unique(gsub("\\.[0-9]+$", "", covariates))
  print("Base covariates:")
  print(base_covariates)
  
  # Get all time-varying and static feature columns
  feature_cols <- c()
  
  # Define feature sets by model type
  base_features <- c("L1", "L2", "L3")
  a_model_features <- base_features
  c_model_features <- c(base_features)
  y_model_features <- c(c_model_features, "Y")
  
  # Select appropriate time-varying features based on model type
  time_varying_covs <- if(is_treatment_model) {
    a_model_features
  } else if(is_censoring_model) {
    c_model_features
  } else {
    y_model_features
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
  
  print("Found feature columns:")
  print(feature_cols)
  
  n_ids <- length(unique(data$ID))
  n_times <- length(target_cols)
  # Adjust for overlapping windows and time periods
  n_sequences_per_id <- n_times - window_size + 1  # Add 1 for overlapping windows
  data_long <- data.frame(
    ID = sort(rep(unique(data$ID), each=n_sequences_per_id)),
    time = rep(0:(n_sequences_per_id-1), times=n_ids)
  )
  
  # Set total sequences
  n_sequences <-  n_ids * n_sequences_per_id
  target_sequences <- n_sequences
  
  # Create base input data frame
  input_data <- data.frame(ID = data_long$ID)
  
  # Handle target creation based on case  
  if(is_censoring_model) {
    # Get censoring data from full matrix
    target_matrix <- matrix(-1, nrow=nrow(data), ncol=length(target_cols))
    for(i in 1:length(target_cols)) {
      target_matrix[,i] <- as.numeric(data[[target_cols[i]]] == -1)
    }
    
    # Map to long format
    data_long$target <- sapply(1:nrow(data_long), function(i) {
      id <- data_long$ID[i]
      t <- data_long$time[i]
      if(t + window_size > ncol(target_matrix)) return(1)
      val <- target_matrix[id, t + window_size]
      if(is.na(val)) return(1)  # Treat NA as censored
      return(val)  # Already 0/1 encoded  # Keeps original encoding (1=event)
    })
  } else if(is_treatment_model) {
    treatment_matrix <- as.matrix(data[target_cols])
    rownames(treatment_matrix) <- data$ID  # Set row names for proper indexing
    
    data_long$A <- sapply(1:nrow(data_long), function(i) {
      id <- data_long$ID[i]
      t <- data_long$time[i]
      
      # Check bounds
      if(t + window_size >= length(target_cols)) {
        return(5)  # Default for out of bounds
      }
      
      # Get ID index by matching against row names
      id_idx <- which(rownames(treatment_matrix) == id)
      if(length(id_idx) == 0) return(5)  # Default if ID not found
      
      # Get value using proper indexing
      val <- treatment_matrix[id_idx[1], t + 1]
      if(is.na(val) || !val %in% 0:6) {
        return(5)  # Default for invalid values
      }
      return(as.numeric(val))
    })
  } else {
    # Y model case
    outcome_matrix <- as.matrix(data[target_cols])
    data_long$target <- sapply(1:nrow(data_long), function(i) {
      id <- data_long$ID[i]
      t <- data_long$time[i]
      val <- outcome_matrix[id, t + window_size]
      if(is.na(val)) return(0)
      as.numeric(val)
    })
  }
  
  # Create output_data based on case
  if(is_treatment_model) {
    if(loss_fn == "binary_crossentropy") {
      # For binary treatment case, create one-hot encoded output
      output_data <- data.frame(
        ID = data_long$ID,
        stringsAsFactors = FALSE
      )
      
      # Convert treatment values to 0-5 categories
      categorical_target <- ifelse(data_long$A == 0, 5, pmin(data_long$A - 1, 5))
      
      # Create one-hot columns A0 through A5
      for(j in 0:5) {
        col_name <- paste0("A", j)
        output_data[[col_name]] <- as.integer(categorical_target == j)
      }
      
      # Set filenames for binary case
      output_filename <- file.path(output_dir, "lstm_bin_A_output.csv")
      input_filename <- file.path(output_dir, "lstm_bin_A_input.csv")
      
    } else {
      # For categorical case
      output_data <- data.frame(
        ID = data_long$ID,
        target = ifelse(data_long$A == 0, 5, pmin(data_long$A - 1, 5)),
        stringsAsFactors = FALSE
      )
      
      # Set filenames for categorical case
      output_filename <- file.path(output_dir, "lstm_cat_A_output.csv")
      input_filename <- file.path(output_dir, "lstm_cat_A_input.csv")
    }
  } else if(is_Y_model) {
    # Y model handling (binary)
    output_data <- data.frame(
      ID = data_long$ID,
      target = data_long$target,
      stringsAsFactors = FALSE
    )
    output_filename <- file.path(output_dir, "lstm_bin_Y_output.csv")
    input_filename <- file.path(output_dir, "lstm_bin_Y_input.csv")
  } else if(is_censoring_model) {
    # Censoring model handling
    output_data <- data.frame(
      ID = data_long$ID,
      target = data_long$target,
      stringsAsFactors = FALSE
    )
    
    print("Censoring model summary:")
    print(paste("Total samples:", nrow(output_data)))
    print(paste("Censored (target=1):", sum(output_data$target == 1)))
    print(paste("Uncensored (target=0):", sum(output_data$target == 0)))
    
    # Set C model filenames
    output_filename <- file.path(output_dir, "lstm_bin_C_output.csv")
    input_filename <- file.path(output_dir, "lstm_bin_C_input.csv")
  }
  
  print("Final output data dimensions:")
  print(dim(output_data))
  print("Names of output columns:")
  print(names(output_data))
  
  # Add explicit binary/continuous type definitions
  binary_covs <- c("L2", "L3", "C", "Y", "white", "black", "latino", "other", "mdd", "bipolar", "schiz")
  continuous_covs <- c("V3", "L1")
  
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
  
  # Before writing files, add:
  if(is_censoring_model) {
    print("Verifying censoring model data:")
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
    
    print("Directory contents after writing:")
    print(list.files(output_dir))
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
  py$lr <- 0.005
  py$dr <- 0.1
  py$nb_batches <- as.integer(64)
  py$patience <- as.integer(2)
  py$t_end <- as.integer(t_end + 1)
  py$feature_cols <- if(length(base_covariates) > 0) base_covariates else stop("No features available")
  py$outcome_cols <- outcome_cols
  
  # Synchronize model type and settings
  if(is_censoring_model) {
    print("Setting Censoring (C) Model params...")
    is_censoring <- TRUE  # Set R variable
    py$is_censoring <- TRUE  # Set Python variable
    py$J <- as.integer(1)
    py$loss_fn <- "binary_crossentropy"
  } else if(is_Y_model) {
    print("Training Outcome (Y) Model...")
    is_censoring <- FALSE  # Set R variable
    py$is_censoring <- FALSE  # Set Python variable
    py$J <- as.integer(1) 
    py$loss_fn <- "binary_crossentropy"
  } else {
    print("Training Treatment (A) Model...")
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
    # Determine prediction file path
    preds_file <- if(is_censoring) {
      file.path(output_dir, 'lstm_bin_C_preds.npy')
    } else {
      # Check if outcome is Y and has binary loss function
      is_Y_outcome <- any(grepl("^Y", outcome_cols))
      if(is_Y_outcome && loss_fn == "binary_crossentropy") {
        file.path(output_dir, 'lstm_bin_Y_preds.npy')
      } else {
        file.path(output_dir, 
                  if(loss_fn == "sparse_categorical_crossentropy") 'lstm_cat_A_preds.npy' 
                  else 'lstm_bin_A_preds.npy')
      }
    }
    
    if(!file.exists(preds_file)) {
      warning(paste("Predictions file not found:", preds_file))
      return(NULL)
    }
    
    # Load prediction array
    preds_r <- as.array(np$load(preds_file))
    n_total_samples <- nrow(preds_r)
    samples_per_time <- n_total_samples %/% (t_end + 1)
    prediction_type <- if(is_Y_outcome) "Y" else if(is_censoring) "C" else "A"
    
    if(debug) {
      cat("\nLoaded predictions from:", preds_file, "\n")
      cat("Array shape:", paste(dim(preds_r), collapse=" x "), "\n")
      cat("Samples per time period:", samples_per_time, "\n")
      cat("Prediction type:", prediction_type, "\n")
    }
    
    # Function to extract a valid time slice
    get_time_slice <- function(t) {
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
      if(nrow(slice) == 0 || ncol(slice) == 0) {
        if(debug) cat("Empty slice extracted\n")
        return(NULL)
      }
      
      if(debug) {
        cat(sprintf("\nTime %d slice [%d:%d]:\n", t-1, start_idx, end_idx))
        cat("Shape:", paste(dim(slice), collapse=" x "), "\n")
        cat("Range:", paste(range(slice), collapse=" - "), "\n")
      }
      
      slice
    }
    
    # Function to process predictions for any type
    process_predictions <- function(slice, type="A") {
      # Handle invalid slice
      if(is.null(slice)) {
        if(debug) cat("Creating default predictions for NULL slice\n")
        return(matrix(
          if(type == "A") 1/J else 0,
          nrow=n_ids,
          ncol=if(type == "A") J else 1
        ))
      }
      
      # Ensure matrix format and proper dimensions
      if(!is.matrix(slice)) {
        slice <- matrix(slice, ncol=if(type == "A") J else 1)
      }
      
      # Interpolate if needed
      if(nrow(slice) != n_ids) {
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
                         pmin(pmax(slice, ybound[1]), ybound[2])
                       },
                       "C" = {
                         pmin(pmax(slice, gbound[1]), gbound[2])
                       },
                       "A" = {
                         # For treatment predictions, ensure proper probabilities
                         t(apply(slice, 1, function(row) {
                           if(any(is.na(row)) || any(!is.finite(row))) return(rep(1/J, J))
                           bounded <- pmax(row, gbound[1])
                           bounded / sum(bounded)
                         }))
                       }
      )
      
      # Add column names
      colnames(result) <- switch(type,
                                 "Y" = "Y",
                                 "C" = "C",
                                 "A" = paste0("A", 1:J)
      )
      
      result
    }
    
    # Process all time periods
    validated_preds <- lapply(1:(t_end + 1), function(t) {
      slice <- get_time_slice(t)
      processed <- process_predictions(slice, type=prediction_type)
      
      if(debug) {
        cat(sprintf("\nProcessed predictions for time %d:\n", t-1))
        cat("Shape:", paste(dim(processed), collapse=" x "), "\n")
        cat("Range:", paste(range(processed), collapse=" - "), "\n")
        if(prediction_type == "A") {
          cat("Row sums:", paste(range(rowSums(processed)), collapse=" - "), "\n")
        }
      }
      
      processed
    })
    
    # Add time period names
    names(validated_preds) <- paste0("t", 0:t_end)
    
    if(debug) {
      cat("\nFinal validation summary:\n")
      cat("Time periods processed:", length(validated_preds), "\n")
      cat("Predictions per period:", nrow(validated_preds[[1]]), "\n")
      ranges <- range(do.call(rbind, validated_preds))
      cat("Overall range:", paste(ranges, collapse=" - "), "\n")
    }
    
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