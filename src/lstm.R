###################################
# LSTM for Simulations            #
###################################

lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir){
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
  py$n_hidden <- 128
  py$hidden_activation <- 'tanh'
  py$out_activation <- out_activation
  py$loss_fn <- loss_fn

  py$lr <- 0.001
  py$dr <- 0.5
  py$nb_batches <- 8
  py$patience <- 25
  py$t_end <- (t_end+1)
  py$window_size <- window_size
  
  source_python("src/train_lstm_sim.py")
  
  print("Reading predictions")
  lstm.pred <- as.matrix(as.data.frame(read_csv(paste0(output_dir, "lstm_preds.csv"))), col_names = colnames(output_data))
  
  print("Renaming predictions")
  lstm.pred <- rbind(output_data[1:window_size,], lstm.pred)
  
  lstm.pred <-lstm.pred[,match(colnames(output_data), colnames(lstm.pred))] # same order
  
  return(lstm.pred)
}