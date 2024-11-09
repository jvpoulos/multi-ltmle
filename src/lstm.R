lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, inference=FALSE, J=7) {
  # Print initial data info for debugging
  print("Initial data structure:")
  print(paste("Dimensions:", paste(dim(data), collapse=" x ")))
  print("Available columns:")
  print(colnames(data))
  
  # Ensure ID column exists
  if(!"ID" %in% colnames(data)) {
    data$ID <- 1:nrow(data)
  }
  
  # Clean up column names - remove periods
  colnames(data) <- gsub("\\.", "", colnames(data))
  
  # Find treatment columns
  A_cols <- grep("^A[0-9]+", colnames(data), value=TRUE)
  if(length(A_cols) == 0) {
    # Try with period pattern
    A_cols_temp <- grep("^A\\.[0-9]+", names(data), value=TRUE)
    A_cols <- gsub("\\.", "", A_cols_temp)
  }
  
  print("Treatment columns found:")
  print(A_cols)
  
  # Verify outcome specification
  if(is.character(outcome)) {
    outcome_cols <- gsub("\\.", "", outcome)
  } else if(is.data.frame(outcome)) {
    outcome_cols <- gsub("\\.", "", colnames(outcome))
  } else {
    outcome_cols <- A_cols
  }
  
  # Verify outcome columns exist
  missing_cols <- setdiff(outcome_cols, colnames(data))
  if(length(missing_cols) > 0) {
    print(paste("Missing columns:", paste(missing_cols, collapse=", ")))
    outcome_cols <- intersect(outcome_cols, colnames(data))
    if(length(outcome_cols) == 0) {
      stop("No valid outcome columns found")
    }
  }
  
  print("Using outcome columns:")
  print(outcome_cols)
  
  # Get feature columns
  all_cols <- colnames(data)
  feature_cols <- setdiff(all_cols, c("ID", outcome_cols))
  
  print("Feature columns:")
  print(feature_cols)
  
  # Verify data structure
  print("Data structure check:")
  print(paste("Number of features:", length(feature_cols)))
  print(paste("Number of outcomes:", length(outcome_cols)))
  
  # Prepare input and output data
  input_data <- data[, c("ID", feature_cols), drop=FALSE]
  output_data <- data[, c("ID", outcome_cols), drop=FALSE]
  
  print("Prepared data dimensions:")
  print(paste("Input data:", paste(dim(input_data), collapse="x")))
  print(paste("Output data:", paste(dim(output_data), collapse="x")))
  
  # Write data to CSV
  input_file <- paste0(output_dir, "input_data.csv")
  output_file <- paste0(output_dir, "output_data.csv")
  write.csv(input_data, input_file, row.names = FALSE)
  write.csv(output_data, output_file, row.names = FALSE)
  
  # Set Python variables
  py$J <- as.integer(J)
  py$output_dir <- output_dir
  py$epochs <- as.integer(100)
  py$n_hidden <- as.integer(256)
  py$hidden_activation <- 'tanh'
  py$out_activation <- out_activation
  py$loss_fn <- loss_fn
  py$lr <- 0.001
  py$dr <- 0.5
  py$nb_batches <- as.integer(256)
  py$patience <- as.integer(2)
  py$t_end <- as.integer(t_end + 1)
  py$window_size <- as.integer(window_size)
  py$feature_cols <- feature_cols
  py$outcome_cols <- outcome_cols
  
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
  
  # Load predictions
  tryCatch({
    preds_file <- ifelse(loss_fn == "sparse_categorical_crossentropy",
                         paste0(output_dir, 'lstm_cat_preds.npy'),
                         paste0(output_dir, 'lstm_bin_preds.npy'))
    
    if(file.exists(preds_file)) {
      preds <- np$load(preds_file)
      preds_r <- as.array(preds)
      
      if(is.null(dim(preds_r)) || length(dim(preds_r)) < 2) {
        warning("Incorrect dimensions for prediction matrix")
        return(NULL)
      }
      
      if(loss_fn == "binary_crossentropy") {
        # Process binary predictions
        preds_list <- vector("list", t_end + 1)
        for(t in 1:(t_end + 1)) {
          preds_list[[t]] <- matrix(preds_r[,min(t, ncol(preds_r))], ncol=1)
          colnames(preds_list[[t]]) <- "prob"
        }
        return(preds_list)
      } else {
        # Return categorical predictions as is
        return(preds_r)
      }
    } else {
      warning(paste("Predictions file not found:", preds_file))
      return(NULL)
    }
  }, error = function(e) {
    warning(paste("Error loading predictions:", e$message))
    return(NULL)
  })
}