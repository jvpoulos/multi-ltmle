###################################
# LSTM for Simulations            #
###################################

lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, inference=FALSE){
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
  
  write.csv(input_data,paste0(output_dir,"input_data.csv"),row.names = FALSE)
  write.csv(output_data,paste0(output_dir,"output_data.csv"),row.names = FALSE)

  py <- import_main()
  py$output_dir <- output_dir
  py$gpu <- 3
  py$epochs <- 500
  py$n_hidden <- 32
  py$hidden_activation <- 'tanh'
  py$out_activation <- out_activation
  py$loss_fn <- loss_fn

  py$lr <- 0.001
  py$dr <- 0.5
  py$nb_batches <- 6
  py$patience <- 15
  py$t_end <- (t_end+1)
  py$window_size <- window_size
  
  np <- import('numpy')
  
  if(inference){
    source_python("src/test_lstm.py")
    
    print("Reading predictions")
    preds <- np$load(paste0(output_dir, 'lstm_new_preds.npy'))
  }else{
    source_python("src/train_lstm.py")
    
    print("Reading predictions")
    preds <- np$load(paste0(output_dir, 'lstm_preds.npy'))
  }
  
  print("Converting to list")
  preds_r_array <- as.array(preds)
  
  # Initialize an empty list to store the predictions
  lstm_preds <- vector("list", length = nrow(preds_r_array))
  
  if(out_activation == "sigmoid"){
    # Fill the list with prediction vectors for binary classification
    for (i in 1:nrow(preds_r_array)) {
      lstm_preds[[i]] <- preds_r_array[i,]
    }
  } else if(out_activation == "softmax"){
    # Fill the list with matrices for multiclass classification
    for (i in 1:dim(preds_r_array)[1]) {
      lstm_preds[[i]] <- preds_r_array[i,,]
    }
  }
  
  print("Returning predictions")
  return(lstm_preds)
}