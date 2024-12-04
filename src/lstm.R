lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, inference=FALSE, J=7, is_censoring=FALSE) {
  print("Initial data structure:")
  print(paste("Dimensions:", paste(dim(data), collapse=" x ")))
  
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
  
  if(length(target_cols) > 0) {
    # Pre-process feature columns to ensure they exist
    base_covariates <- unique(gsub("\\.[0-9]+$", "", covariates))
    print("Base covariates:")
    print(base_covariates)
    
    # Get all actual available covariate columns
    feature_cols <- c()
    for(base_col in base_covariates) {
      pattern <- paste0("^", base_col, "\\.[0-9]+$")
      matching_cols <- grep(pattern, colnames(data), value=TRUE)
      feature_cols <- c(feature_cols, matching_cols)
    }
    
    print("Found feature columns:")
    print(feature_cols)
    
    # Create base data frame with proper ID and time handling
    n_ids <- nrow(data)
    n_times <- length(target_cols)
    
    # Create proper ID sequence and time sequence
    data_long <- data.frame(
      ID = rep(1:n_ids, each=n_times),
      time = rep(0:(n_times-1), times=n_ids)
    )
    
    # Add target column based on case
    if(is_censoring) {
      data_long$target <- as.vector(t(as.matrix(data[target_cols])))
    } else {
      data_long$A <- as.vector(t(as.matrix(data[target_cols])))
    }
    
    # Add features with proper recycling
    for(base_col in base_covariates) {
      for(t in 0:(n_times-1)) {
        col_name <- paste0(base_col, ".", t)
        if(col_name %in% colnames(data)) {
          target_rows <- data_long$time == t
          data_long[target_rows, base_col] <- data[[col_name]]
        }
      }
    }
    
    if(is_censoring) {
      print("Censoring distribution before processing:")
      print(table(data_long$target, useNA="ifany"))
      
      # Handle binary censoring case
      data_long <- data_long[order(data_long$ID, data_long$time), ]
      
      # Ensure complete ID-time grid
      expected_rows <- n_ids * n_times
      if(nrow(data_long) != expected_rows) {
        id_time_grid <- expand.grid(
          ID = 1:n_ids,
          time = 0:(n_times-1)
        )
        data_long <- merge(id_time_grid, data_long, by=c("ID", "time"), all.x=TRUE)
      }
      
      # Fill NAs with forward fill within each ID
      data_long <- do.call(rbind, 
                           lapply(split(data_long, data_long$ID), function(df) {
                             df <- df[order(df$time), ]
                             df$target <- zoo::na.locf(df$target, na.rm=FALSE)
                             return(df)
                           })
      )
      
      # Fill remaining NAs with 0
      data_long$target[is.na(data_long$target)] <- 0
      
      # Ensure all columns exist
      all_cols <- unique(c(names(data_long), base_covariates))
      for(col in setdiff(all_cols, names(data_long))) {
        data_long[[col]] <- -1
      }
      
      # Create input and output data
      input_data <- data.frame(ID = data_long$ID)
      for(col in base_covariates) {
        if(col %in% colnames(data_long)) {
          input_data[[col]] <- data_long[[col]]
        } else {
          warning(paste("Covariate", col, "not found in data"))
          input_data[[col]] <- -1
        }
      }
      
      output_data <- data.frame(
        ID = data_long$ID,
        target = data_long$target
      )
      
    } else if(loss_fn == "sparse_categorical_crossentropy") {
      print("Treatment distribution before processing:")
      print(table(data_long$A, useNA="ifany"))
      
      # Handle categorical treatment case
      data_long <- data_long[order(data_long$ID, data_long$time), ]
      data_long <- do.call(rbind, 
                           lapply(split(data_long, data_long$ID), function(df) {
                             df <- df[order(df$time), ]
                             df$A <- zoo::na.locf(df$A, na.rm=FALSE)
                             return(df)
                           })
      )
      
      if(any(is.na(data_long$A))) {
        data_long$A[is.na(data_long$A)] <- -1
      }
      
      treatment_map <- numeric(7)
      treatment_map[1] <- 6
      treatment_map[2:7] <- 1:6
      names(treatment_map) <- 0:6
      
      if(!all(data_long$A %in% 0:6)) {
        stop("Invalid treatment values found. Expected values 0-6")
      }
      
      data_long$target <- treatment_map[as.numeric(as.character(data_long$A)) + 1]
      
      output_data <- data.frame(
        ID = data_long$ID,
        target = data_long$target - 1
      )
      
      input_data <- data.frame(ID = data_long$ID)
      for(col in base_covariates) {
        if(col %in% colnames(data_long)) {
          input_data[[col]] <- data_long[[col]]
        } else {
          warning(paste("Covariate", col, "not found in data"))
          input_data[[col]] <- -1
        }
      }
      
    } else {
      # Binary treatment case
      input_data <- data.frame(ID = data_long$ID)
      for(col in base_covariates) {
        if(col %in% colnames(data_long)) {
          input_data[[col]] <- data_long[[col]]
        } else {
          warning(paste("Covariate", col, "not found in data"))
          input_data[[col]] <- -1
        }
      }
      
      output_data <- data_long[c("ID", "A")]
      names(output_data)[names(output_data) == "A"] <- "target"
    }
    
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
    
    # Fill NAs in features with -1
    input_data[is.na(input_data)] <- -1
    
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
    
  } else {
    stop("No target columns found in data")
  }
  
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
  py$epochs <- as.integer(1)
  py$n_hidden <- as.integer(256)
  py$hidden_activation <- 'tanh'
  py$out_activation <- out_activation
  py$loss_fn <- loss_fn
  py$lr <- 0.001
  py$dr <- 0.5
  py$nb_batches <- as.integer(256)
  py$patience <- as.integer(1)
  py$t_end <- as.integer(t_end + 1)
  py$window_size <- as.integer(window_size)
  py$feature_cols <- if(length(base_covariates) > 0) base_covariates else stop("No features available")
  py$outcome_cols <- outcome_cols  # Pass outcome columns
  
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
      # Load predictions
      preds <- np$load(preds_file)
      preds_r <- as.array(preds)
      
      # Get prediction dimensions and samples per time step
      pred_dims <- dim(preds_r)
      n_total_samples <- pred_dims[1]
      
      # Ensure proper division for time steps
      samples_per_time <- n_total_samples %/% (t_end + 1)
      
      # Pre-calculate time indices
      time_indices <- lapply(1:(t_end + 1), function(t) {
        start_idx <- ((t-1) * samples_per_time) + 1
        end_idx <- t * samples_per_time
        start_idx:end_idx
      })
      
      # Initialize prediction list for each time step
      preds_list <- vector("list", t_end + 1)
      
      if(is_censoring) {
        # Handle binary censoring case
        if(length(pred_dims) == 1) {
          preds_r <- matrix(preds_r, ncol=1)
        }
        
        # Process predictions for each time step
        preds_list <- lapply(time_indices, function(idx) {
          # Extract and reshape slice, then add column name
          mat <- matrix(preds_r[idx], ncol=1)
          colnames(mat) <- "C"
          mat
        })
        
      } else if(loss_fn == "binary_crossentropy") {
        print("Processing multi-class predictions")
        print(paste("Number of classes:", J))
        
        # Ensure predictions are properly shaped
        if(length(pred_dims) == 1) {
          preds_r <- matrix(preds_r, ncol=J)
        } else if(length(pred_dims) == 3) {
          preds_r <- matrix(preds_r, ncol=J)
        }
        
        # Process predictions for each time step
        preds_list <- lapply(time_indices, function(idx) {
          # Extract time slice and reshape
          mat <- matrix(preds_r[idx, ], ncol=J)
          colnames(mat) <- paste0("A", 0:(J-1))
          mat
        })
        
        # Print shapes for verification
        for(t in 1:(t_end + 1)) {
          print(paste0("Time ", t-1, " prediction shape: ", 
                       paste(dim(preds_list[[t]]), collapse=" x ")))
        }
        
      } else {
        # Handle categorical treatment case
        if(length(pred_dims) == 1) {
          preds_r <- matrix(preds_r, ncol=J)
        }
        
        # Process predictions for each time step
        preds_list <- lapply(time_indices, function(idx) {
          # Extract time slice and reshape
          mat <- matrix(preds_r[idx, ], ncol=J)
          colnames(mat) <- paste0("class", 1:J)
          mat
        })
      }
      
      # Add names to the time steps
      names(preds_list) <- paste0("t", 0:t.end)
      
      # Clean up predictions
      preds_list <- lapply(preds_list, function(mat) {
        mat[is.nan(mat) | is.infinite(mat)] <- NA
        mat
      })
      
      return(preds_list)
      
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