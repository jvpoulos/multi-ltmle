###################################
# LSTM for Simulations #
###################################

lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, inference=FALSE, J=7){
  # lstm() function calls train_lstm.py, which inputs the data and outputs predictions. The prediction task is to predict 10,000 outcomes,
  # all are factor variables with 7 classes (0,1,2,3,4,5,6), where 0 represents missing values.
  #
  # data: data.frame in T x N format
  # output_dir: character string output directory
  # out_activation: Activation function to use for the output step
  # loss_fn: Loss function corresponding to output type
  
  # Converting the data to a floating point matrix
  input_data <- data.matrix(data[covariates]) # T x N
  output_data <- data.matrix(data[outcome]) # T x N
  
  print(paste0("input_data dimensions: ", dim(input_data)))
  print(paste0("output_data dimensions: ", dim(output_data)))


  output_data_long <- data.frame(
  id = rep(1:ncol(output_data), each = nrow(output_data)),
  t = rep(1:nrow(output_data), times = ncol(output_data)),
  output = c(t(output_data))
  )

  input_data_long <- data.frame(
  id = rep(1:ncol(input_data), each = nrow(input_data)),
  t = rep(1:nrow(input_data), times = ncol(input_data)),
  feature = rep(colnames(input_data), each = nrow(input_data)),
  value = c(t(input_data))
  )

  write.csv(input_data_long, paste0(output_dir, "input_cat_data.csv"), row.names = FALSE)
  write.csv(output_data_long, paste0(output_dir, "output_cat_data.csv"), row.names = FALSE)
  
  py <- import_main()
  py$J <- as.integer(J)
  py$output_dir <- output_dir
  py$epochs <- 200
  py$n_hidden <- 32
  py$hidden_activation <- 'tanh'
  py$out_activation <- out_activation
  py$loss_fn <- loss_fn
  py$lr <- 0.001
  py$dr <- 0.5
  py$nb_batches <- 16
  py$patience <- 10
  py$t_end <- as.integer((t_end + 1))
  py$window_size <- as.integer(window_size)
  
  np <- import('numpy')
  
  # Checks
  if (py_available()) {
    print("Python environment is set up correctly.")
  } else {
    print("Python environment is not set up correctly.")
  }
  
  if (py_module_available("numpy")) {
    print("NumPy is installed.")
  } else {
    print("NumPy is not installed.")
  }
  
  print(paste0("J: ", py$J))
  print(paste0("output_dir: ", py$output_dir))
  
  if (!(out_activation %in% c("sigmoid", "softmax"))) {
    stop(paste0("Invalid out_activation: ", out_activation))
  }

  if (inference) {
    source_python("src/test_lstm.py")
    print("Reading predictions")
    if (loss_fn == "sparse_categorical_crossentropy") {
      preds <- np$load(paste0(output_dir, 'lstm_new_cat_preds.npy'))
    } else {
      preds <- np$load(paste0(output_dir, 'lstm_new_bin_preds.npy'))
    }
  } else {
    source_python("src/train_lstm.py")
    print("Reading predictions")
    if (loss_fn == "sparse_categorical_crossentropy") {
      preds <- np$load(paste0(output_dir, 'lstm_cat_preds.npy'))
    } else {
      preds <- np$load(paste0(output_dir, 'lstm_bin_preds.npy'))
    }
  }
  
  print("Converting to list")
  preds_r_array <- as.array(preds)
  
  # Get the dimensions of the predictions array
  preds_dim <- dim(preds_r_array)

  # Check if preds_r_array has at least 3 dimensions
  if (length(preds_dim) < 3) {
    stop("Predictions array does not have the expected number of dimensions.")
  }

  # Initialize an empty list to store the predictions
  lstm_preds <- vector("list", length = t_end + 1)

  for (t in 1:(t_end + 1)) {
    # Fill the list with prediction matrices for each time step
    lstm_preds[[t]] <- preds_r_array[, , t]
  }
  
  print(paste0("Dimensions of preds_r_array: ", paste(preds_dim, collapse = " x ")))
  print(paste0("Length of lstm_preds: ", length(lstm_preds)))
  
  print("Returning predictions")
  return(lstm_preds)
}