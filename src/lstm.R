###################################
# LSTM for Simulations            #
###################################

lstm <- function(data, outcome, covariates, t_end, out_activation, loss_fn, output_dir){
  # data: data.frame in T x N format
  # output_dir: character string output directory
  # out_activation: Activation function to use for the output step
  # loss_fn: Loss function corresponding to output type
  
  # Converting the data to a floating point matrix
  input_data <- data.matrix(data[covariates]) # T x N
  output_data <- data.matrix(data[outcome]) # T x N
  
  write.csv(data,paste0("input_data.csv"),row.names = FALSE)
  write.csv(data,paste0("output_data.csv"),row.names = FALSE)

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
  py$penalty <- 0.01
  py$nb_batches <- 32
  py$patience <- 25
  py$t_end <- t_end
  py$n_pre <- ceiling(t.end/2)
  
  source_python("src/train_lstm_sim.py")
  
  lstm.pred <- as.matrix(read_csv(paste0(output_dir, "lstm_preds.csv"), col_names = FALSE))
  colnames(lstm.pred) <- colnames(output_data)
  
  lstm.pred <-lstm.pred[,match(colnames(output_data), colnames(lstm.pred))] # same order
  
  return(t(lstm.pred))  # N x T
}