lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, inference=FALSE, J=7) {
  print("Initial data structure:")
  print(paste("Dimensions:", paste(dim(data), collapse=" x ")))
  
  # Find treatment columns with both patterns
  A_cols_dot <- grep("^A\\.[0-9]+$", colnames(data), value=TRUE)
  A_cols_plain <- grep("^A[0-9]+$", colnames(data), value=TRUE)
  A_cols <- if(length(A_cols_dot) > 0) A_cols_dot else A_cols_plain
  
  print(paste("Found", length(A_cols), "treatment columns"))
  print("Treatment columns:")
  print(A_cols)
  
  if(length(A_cols) > 0) {
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
    
    # Create base data frame with ID, time, and treatments
    data_long <- data.frame(
      ID = rep(1:nrow(data), length(A_cols)),
      time = rep(0:(length(A_cols)-1), each=nrow(data)),
      A = unlist(data[A_cols])
    )
    
    # Add features ensuring correct time matching 
    for(base_col in base_covariates) {
      for(t in 0:(length(A_cols)-1)) {
        col_name <- paste0(base_col, ".", t)
        if(col_name %in% colnames(data)) {
          target_rows <- data_long$time == t
          data_long[target_rows, base_col] <- data[[col_name]]
        }
      }
    }
    
    print("Treatment distribution before processing:")
    print(table(data_long$A, useNA="ifany"))
    
    if(loss_fn == "sparse_categorical_crossentropy") {
      # Fill NAs with forward fill within each ID
      data_long <- data_long[order(data_long$ID, data_long$time), ]
      data_long <- do.call(rbind, 
                           lapply(split(data_long, data_long$ID), function(df) {
                             df$A <- zoo::na.locf(df$A, na.rm=FALSE)
                             return(df)
                           })
      )
      
      # Fill any remaining NAs with most common treatment
      if(any(is.na(data_long$A))) {
        most_common <- as.numeric(names(sort(table(data_long$A[!is.na(data_long$A)]), decreasing=TRUE)[1]))
        data_long$A[is.na(data_long$A)] <- most_common
      }
      
      # Create a proper mapping for all possible treatment values (0-6)
      treatment_map <- numeric(7) # Create a vector of length 7 for treatments 0-6
      treatment_map[1] <- 6      # Map treatment 0 to class 6
      treatment_map[2:7] <- 1:6  # Map treatments 1-6 to classes 1-6
      names(treatment_map) <- 0:6
      
      # Ensure treatment values are valid before mapping
      if(!all(data_long$A %in% 0:6)) {
        stop("Invalid treatment values found. Expected values 0-6")
      }
      
      # Map treatments to class indices (1-based)
      data_long$target <- treatment_map[as.numeric(as.character(data_long$A)) + 1]
      
      # Create final output with 0-based targets for TensorFlow
      output_data <- data.frame(
        ID = data_long$ID,
        target = data_long$target - 1  # Convert to 0-based for TensorFlow
      )
      
      # Create input data with base covariates
      # First ensure all base covariates exist
      input_data <- data.frame(ID = data_long$ID)
      for(col in base_covariates) {
        if(col %in% colnames(data_long)) {
          input_data[[col]] <- data_long[[col]]
        } else {
          warning(paste("Covariate", col, "not found in data"))
          input_data[[col]] <- -1  # Default value for missing covariates
        }
      }
      
      print("Target distribution (0-based):")
      print(table(output_data$target, useNA="ifany"))
      
      print("Final target range check:")
      print(paste("Min:", min(output_data$target, na.rm=TRUE)))
      print(paste("Max:", max(output_data$target, na.rm=TRUE)))
    } else {
      # Add handling for binary crossentropy case
      # Create input data with base covariates
      input_data <- data.frame(ID = data_long$ID)
      for(col in base_covariates) {
        if(col %in% colnames(data_long)) {
          input_data[[col]] <- data_long[[col]]
        } else {
          warning(paste("Covariate", col, "not found in data"))
          input_data[[col]] <- -1  # Default value for missing covariates
        }
      }
      
      # Create output data for binary case
      output_data <- data_long[c("ID", "A")]
    }
    
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
    
    # Rest of the code
    print("Final data dimensions:")
    print("Input data:")
    print(str(input_data))
    print("Output data:")
    print(str(output_data))
    
    # Set Python variables and continue...
  } else {
    stop("No treatment columns found in data")
  }
  
  # Set Python variables
  py$J <- as.integer(J)
  py$output_dir <- output_dir
  py$epochs <- as.integer(10)
  py$n_hidden <- as.integer(256)
  py$hidden_activation <- 'tanh'
  py$out_activation <- out_activation
  py$loss_fn <- loss_fn
  py$lr <- 0.001
  py$dr <- 0.5
  py$nb_batches <- as.integer(128)
  py$patience <- as.integer(2)
  py$t_end <- as.integer(t.end + 1)
  py$window_size <- as.integer(window_size)
  py$feature_cols <- if(length(base_covariates) > 0) base_covariates else stop("No features available")
  py$outcome_cols <- if(is.character(outcome)) {
    if(length(outcome) > 0) outcome else stop("Empty outcome")
  } else if(!is.null(colnames(outcome))) {
    colnames(outcome)
  } else {
    stop("Invalid outcome specification")
  }
  
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
    preds_file <- ifelse(loss_fn == "sparse_categorical_crossentropy",
                         paste0(output_dir, 'lstm_cat_preds.npy'),
                         paste0(output_dir, 'lstm_bin_preds.npy'))
    
    if(file.exists(preds_file)) {
      # Load predictions
      preds <- np$load(preds_file)
      
      # Convert to R array
      preds_r <- as.array(preds)
      
      if(loss_fn == "binary_crossentropy") {
        print("Processing multi-class predictions")
        print(paste("Number of classes:", J))
        print(paste("Predictions shape:", paste(dim(preds_r), collapse=" x ")))
        
        # Initialize list for time steps
        preds_list <- vector("list", t_end + 1)
        
        # Get dimensions
        n_samples <- dim(preds_r)[1]
        
        # For each time step, create a copy of the predictions matrix
        for(t in 1:(t_end + 1)) {
          # Create a copy of the predictions
          time_matrix <- preds_r
          
          # Ensure correct dimensions and names
          dim(time_matrix) <- c(n_samples, J)
          colnames(time_matrix) <- paste0("A", 0:(J-1))
          
          # Store in list
          preds_list[[t]] <- time_matrix
          
          # Print dimensions for verification
          print(paste0("Time ", t-1, " prediction shape: ", 
                       paste(dim(time_matrix), collapse=" x ")))
        }
        
        # Add names to list elements
        names(preds_list) <- paste0("t", 0:t_end)
        
        # Filter out potential NaN/Inf values
        preds_list <- lapply(preds_list, function(mat) {
          mat[is.nan(mat) | is.infinite(mat)] <- NA
          return(mat)
        })
        
        return(preds_list)
        
      } else {
        # For categorical case
        pred_matrix <- preds_r
        dim(pred_matrix) <- c(dim(preds_r)[1], J)
        colnames(pred_matrix) <- paste0("class", 1:J)
        return(pred_matrix)
      }
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