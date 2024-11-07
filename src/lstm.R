lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, inference=FALSE, J=7) {
  # Ensure ID column exists and is the first column
  if(!"ID" %in% colnames(data)) {
    data$ID <- 1:nrow(data)
  }
  data <- data[, c("ID", setdiff(colnames(data), "ID"))]
  
  # Fix column names to match expected format
  fixed_colnames <- gsub("\\.", "", colnames(data))
  colnames(data) <- fixed_colnames
  
  # Update outcome and covariates to match fixed column names
  if(is.character(outcome)) {
    outcome_cols <- gsub("\\.", "", outcome)
  } else if(is.data.frame(outcome)) {
    outcome_cols <- gsub("\\.", "", colnames(outcome))
  } else {
    stop("outcome must be either column names or a data frame")
  }
  
  covariates <- gsub("\\.", "", covariates)
  
  # Verify columns exist
  missing_cols <- setdiff(c(outcome_cols, covariates), colnames(data))
  if(length(missing_cols) > 0) {
    stop(paste("Missing columns:", paste(missing_cols, collapse=", ")))
  }
  
  # Separate input and output data
  input_data <- data[, c("ID", covariates)]
  output_data <- data[, c("ID", outcome_cols)]
  
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
  py$outcome_cols <- outcome_cols
  py$covariate_cols <- covariates
  
  # Import numpy
  np <- reticulate::import("numpy")
  
  # Print debug info
  print(paste("Data dimensions:", paste(dim(data), collapse="x")))
  print(paste("Input columns:", paste(covariates, collapse=", ")))
  print(paste("Output columns:", paste(outcome_cols, collapse=", ")))
  
  # Run Python script
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
      
      if(is.null(dim(preds_r)) || length(dim(preds_r)) != 2) {
        warning("Incorrect dimensions for prediction matrix")
        return(NULL)
      }
      
      if(loss_fn == "binary_crossentropy") {
        # Process binary predictions
        preds_list <- vector("list", t_end + 1)
        for(t in 1:(t_end + 1)) {
          preds_list[[t]] <- matrix(preds_r[,1], ncol=1)
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