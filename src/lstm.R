lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, inference=FALSE, J=7){
  print(paste("Data dimensions:", paste(dim(data), collapse="x")))
  print(paste("Outcome:", paste(outcome, collapse=", ")))
  print(paste("Covariates:", paste(covariates, collapse=", ")))
  
  # Ensure ID column exists
  if(!"ID" %in% colnames(data)) {
    data$ID <- 1:nrow(data)
  }
  
  # Separate input and output data
  input_data <- data[, c("ID", covariates)]
  output_data <- data[, c("ID", outcome)]
  
  # Write data to CSV
  input_file <- paste0(output_dir, "input_data.csv")
  output_file <- paste0(output_dir, "output_data.csv")
  write.csv(input_data, input_file, row.names = FALSE)
  write.csv(output_data, output_file, row.names = FALSE)
  
  # Print summary
  print("Input data summary (sample):")
  print(summary(input_data))
  print("Output data summary (sample):")
  print(summary(output_data))
  
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
  py$nb_batches <- as.integer(64)
  py$patience <- as.integer(2)
  py$t_end <- as.integer(t_end + 1)
  py$window_size <- as.integer(window_size)
  py$outcome <- outcome
  py$covariates <- covariates
  
  # Run Python script
  if (inference) {
    tryCatch({
      source_python("src/test_lstm.py")
    }, error = function(e) {
      print("Error in test_lstm.py:")
      print(e$message)
      print(paste("Error details:", reticulate::py_last_error()))
    })
  } else {
    tryCatch({
      source_python("src/train_lstm.py")
    }, error = function(e) {
      print("Error in train_lstm.py:")
      print(e$message)
      print(paste("Error details:", reticulate::py_last_error()))
    })
  }
  
  # Read and return predictions
  preds_file <- ifelse(loss_fn == "sparse_categorical_crossentropy", 
                       paste0(output_dir, 'lstm_cat_preds.npy'),
                       paste0(output_dir, 'lstm_bin_preds.npy'))
  
  tryCatch({
    if (file.exists(preds_file)) {
      preds <- np$load(preds_file)
      preds_r <- as.array(preds)
      return(preds_r)
    } else {
      warning(paste("Predictions file not found:", preds_file))
      return(NULL)
    }
  }, error = function(e) {
    warning(paste("Error loading predictions:", e$message))
    return(NULL)
  })
}