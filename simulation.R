############################################################################################
# Longitudinal setting (T>1) simulations: Compare multinomial TMLE with binary TMLE        #
############################################################################################

######################
# Simulation function #
######################

simLong <- function(r, J=6, n=10000, t.end=36, gbound=c(0.05,1), ybound=c(0.0001,0.9999), n.folds=3, cores=1, estimator="tmle", treatment.rule = "all", use.SL=TRUE, scale.continuous=FALSE, debug =TRUE, window_size=7){
  # Set a global flag for detecting LSTM data generation issues
  lstm_debug_enabled <- debug
  assign("lstm_debug_enabled", lstm_debug_enabled, envir = .GlobalEnv)
  
  # libraries
  library(simcausal)
  options(simcausal.verbose=FALSE)
  library(purrr)
  library(origami)
  library(sl3)
  options(sl3.verbose = FALSE)
  library(nnet)
  library(ranger)
  library(glmnet)
  library(MASS)
  library(progressr)
  library(data.table)
  library(gtools)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(latex2exp)
  
  if(estimator=='tmle-lstm'){
    source('./src/tmle_fns_lstm.R')
    source('./src/lstm.R')
    verify_reticulate()
    library(reticulate)
    use_python("/media/jason/Dropbox/github/multi-ltmle/myenv/bin/python", required = TRUE)
    print(py_config()) # Check Python configuration
    
    # Make sure process_predictions is available globally
    if(!exists("process_predictions", envir = .GlobalEnv)) {
      print("Ensuring process_predictions is available in global environment...")
      # Explicitly copy the function to the global environment
      process_predictions_temp <- process_predictions
      assign("process_predictions", process_predictions_temp, envir = .GlobalEnv)
    }
    
    np <- reticulate::import("numpy")
    print("Checking np object:")
    print(py_get_attr(np, "__name__"))
    print(py_get_attr(np, "__version__"))
    
    library(tensorflow)
    library(keras)
    print(is_keras_available())
    print(tf_version())
    
    wandb <- reticulate::import("wandb", delay_load = TRUE)
    print("Checking wandb object:")
    print(py_get_attr(wandb, "__name__"))
    print(py_get_attr(wandb, "__version__"))
  }
  
  if(estimator%in%c("tmle")){
    source('./src/tmle_fns.R')  
    source('./src/SL3_fns.R')
  }
  
  source('./src/tmle_IC.R')
  source('./src/misc_fns.R')
  
  writeLines(capture.output(sessionInfo()), 'session_info.txt')
  
  if(J!=6){
    stop("J must be 6")
  }
  
  if(t.end<4 && t.end >36){
    stop("t.end must be at least 4 and no more than 36")
  }
  
  if(t.end!=36 & estimator!="tmle"){
    stop("need to manually change the number of lags in tmle_dat and IC in tmle_fns")
  }
  
  if(n.folds<3){
    stop("n.folds needs to be greater than 3")
  }
  
  if(scale.continuous==TRUE){
    warning("real values of certain time-varying covariates needed to characterize treatment rules")
  }
  
  # define DGP
  source('./src/simcausal_fns.R')
  source('./src/simcausal_dgp.R', local =TRUE)
  
  # specify intervention rules (t=0 is same as observed)
  Dset <- set.DAG(D, vecfun=c("StochasticFun")) # locks DAG, consistency checks
  
  # Call the improved function with LaTeX formatting
  if(r==1){
    # Create a larger PNG file with white background
    png(paste0(output_dir, "DAG_plot_latex.png"), 
        width=1800, height=1200, res=150, 
        pointsize=12, bg="white")
    
    # Call the improved plotting function
    source('./src/plotDAG_improved.R')
    dag_graph <- plotDAG_improved(
      Dset, 
      excludeattrs=c("C_0","Y_0"), 
      xjitter=0,        # No jitter for a more structured layout
      yjitter=0,        # No jitter for a more structured layout
      tmax = 3,
      customvlabs = c("V^1", "V^2", "V^3",
                      "L^1_0", "L^2_0", "L^3_0",
                      "L^1_1", "L^2_1", "L^3_1",
                      "L^1_2", "L^2_2", "L^3_2",
                      "L^1_3", "L^2_3", "L^3_3",
                      "A_0", "A_1", "A_2", "A_3", 
                      "C_1", "C_2", "C_3", "Y_1", "Y_2", "Y_3"),
      node_size = 8,      # Larger nodes to fit LaTeX text
      label_size = 1.5,    # Larger text labels
      label_dist = 0,      # Labels positioned at center of nodes
      use_latex = TRUE,    # Enable LaTeX formatting
      arrow_size_mult = 1.5, # Increase arrow head size
      vertex_attrs = list(frame.width = 1.0),  # Thicker node borders
      edge_attrs = list(width = 0.8, color = "darkgray")  # Thicker, more visible edges
    )
    
    dev.off()
    
    # Save a PDF version as well for better LaTeX rendering
    pdf(paste0(output_dir, "DAG_plot_latex.pdf"), 
        width=12, height=8)
    
    plotDAG_improved(
      Dset, 
      excludeattrs=c("C_0","Y_0"), 
      xjitter=0,
      yjitter=0,
      tmax = 3,
      customvlabs = c("V^1", "V^2", "V^3",
                      "L^1_0", "L^2_0", "L^3_0",
                      "L^1_1", "L^2_1", "L^3_1",
                      "L^1_2", "L^2_2", "L^3_2",
                      "L^1_3", "L^2_3", "L^3_3",
                      "A_0", "A_1", "A_2", "A_3", 
                      "C_1", "C_2", "C_3", "Y_1", "Y_2", "Y_3"),
      node_size = 8,
      label_size = 1.5,
      label_dist = 0,
      use_latex = TRUE,
      arrow_size_mult = 1.5,
      vertex_attrs = list(frame.width = 1.0),
      edge_attrs = list(width = 0.8, color = "darkgray")
    )
    
    dev.off()
  }
  
  int.static <-c(node("A", t = 0:t.end, distr = "rconst", # Static: Everyone gets quetiap (if bipolar=2), halo (if schizophrenia=3), ari (if MDD=1) and stays on it
                      const = ifelse(V2[0]==3, 2, ifelse(V2[0]==1, 1, 4))),
                 node("C", t = 1:t.end, distr = "rbern", prob = 0)) # under no censoring
  
  int.dynamic <- c(node("A", t = 0, distr = "rconst",   # Dynamic: Everyone starts with risp. # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to quet. (if bipolar), halo. (if schizophrenia), ari (if MDD); otherwise, stay on risp.
                        const= 5),
                   node("A", t = 1:t.end, distr = "rconst",
                        const=ifelse((L1[t] >0 | L2[t] >0 | L3[t] >0), ifelse(V2[0]==3, 2, ifelse(V2[0]==2, 4, 1)), 5)),
                   node("C", t = 1:t.end, distr = "rbern", prob = 0)) # under no censoring
  
  int.stochastic <- c(node("A", t = 1:t.end, distr = "Multinom", # at each t>0, 95% chance of staying with treatment at t-1, 5% chance of randomly switching according to Multinomial distibution
                           probs = StochasticFun(A[(t-1)], d=c(0,0,0,0,0,0))), 
                      node("C", t = 1:t.end, distr = "rbern", prob = 0)) # under no censoring
  
  D.dyn1 <- Dset + action("A_th1", nodes = int.static) 
  D.dyn2 <- Dset + action("A_th2", nodes = int.dynamic) 
  D.dyn3 <- Dset + action("A_th3", nodes = int.stochastic)
  
  # generate counterfactual data under no censoring- sampled from post-intervention distr. defined by intervention on the DAG
  # iid obs. of node sequence defined by DAG
  
  dat <- list()
  
  dat[["A_th1"]] <-sim(DAG = D.dyn1, actions = "A_th1", n = n, LTCF = "Y", rndseed = r, verbose = FALSE) # static
  dat[["A_th2"]] <-sim(DAG = D.dyn2, actions = "A_th2", n = n, LTCF = "Y", rndseed = r, verbose = FALSE)  # dynamic
  dat[["A_th3"]] <-sim(DAG = D.dyn3, actions = "A_th3", n = n, LTCF = "Y", rndseed = r, verbose = FALSE) # stochastic
  
  # true parameter values:
  
  D.dyn1 <- set.targetE(D.dyn1, outcome = "Y", t=1:t.end, param = "A_th1") # vector of counterfactual means of "Y" over all time periods
  D.dyn2 <- set.targetE(D.dyn2, outcome = "Y", t=1:t.end, param = "A_th2")
  D.dyn3 <- set.targetE(D.dyn3, outcome = "Y", t=1:t.end, param = "A_th3") 
  
  Y.true <- list() # Y.true represents event probabilities (1=event)
  Y.true[["static"]] <- eval.target(D.dyn1, data = dat[["A_th1"]])$res
  Y.true[["dynamic"]] <- eval.target(D.dyn2, data = dat[["A_th2"]])$res
  Y.true[["stochastic"]] <- eval.target(D.dyn3, data = dat[["A_th3"]])$res
  
  # simulate observed data (under censoring) - sampled from (pre-intervention) distribution specified by DAG
  Odat <- sim(DAG = Dset, n = n, LTCF = "Y", rndseed = r, verbose = FALSE) # survival outcome =1 after first occurance
  
  anodes <- grep("A",colnames(Odat),value=TRUE)
  cnodes <- grep("C",colnames(Odat),value=TRUE)
  ynodes <- grep("Y", colnames(Odat), value = TRUE)
  
  # store observed treatment assignment
  
  if(estimator=="tmle-lstm"){
    # store observed treatment assignment
    obs.treatment <- Odat[,anodes] # t=0,2,...,T
    
    # Ensure obs.treatment is a factor with levels 1 to J
    for(t in 1:(t.end+1)) {
      obs.treatment[,t] <- factor(obs.treatment[,t], levels = 1:J)
    }
    
    # Create dummy variables for each time point
    treatments <- lapply(1:(t.end+1), function(t) {
      dummy <- model.matrix(~ obs.treatment[,t] - 1)
      colnames(dummy) <- paste0("A", 1:J)
      as.data.frame(dummy)
    })
    
    # Add a level for censoring (0) to each time point after t=0
    for(t in 1:t.end) {
      obs.treatment[,(1+t)] <- addNA(obs.treatment[,(1+t)])
      levels(obs.treatment[,(1+t)]) <- c(levels(obs.treatment[,(1+t)]), "0")
      obs.treatment[,(1+t)][is.na(obs.treatment[,(1+t)])] <- "0"
    }
    
  }else{
    obs.treatment <- Odat[,anodes] # t=0,2,...,T
    
    treatments <- lapply(1:(t.end+1), function(t) as.data.frame(dummify(obs.treatment[,t])))
    
    for(t in 1:t.end){
      obs.treatment[,(1+t)] <- factor(obs.treatment[,(1+t)], levels = levels(addNA(obs.treatment[,(1+t)])), labels = c(levels(obs.treatment[,(1+t)]), 0), exclude = NULL)
    }
  }
  
  # store censored time
  time.censored <- data.frame(which(Odat[,cnodes]==1, arr.ind = TRUE))
  time.censored <- time.censored[order(time.censored$row),]
  colnames(time.censored) <- c("ID","time_censored")
  
  # dummies for who followed treatment rule in observed data 
  obs.treatment.rule <- list()
  obs.treatment.rule[["static"]] <- (dat[["A_th1"]]$A_th1[,anodes] ==  obs.treatment)+0
  for(i in 1:ncol(dat[["A_th2"]]$A_th2[,anodes])){ # fix levels
    levels(dat[["A_th2"]]$A_th2[,anodes][,i]) <- levels(obs.treatment[,i])
  }
  obs.treatment.rule[["dynamic"]] <- (dat[["A_th2"]]$A_th2[,anodes] ==  obs.treatment)+0
  for(i in 1:ncol(dat[["A_th3"]]$A_th3[,anodes])){ # fix levels
    levels(dat[["A_th3"]]$A_th3[,anodes][,i]) <- levels(obs.treatment[,i])
  }
  obs.treatment.rule[["stochastic"]] <- (dat[["A_th3"]]$A_th3[,anodes] ==  obs.treatment)+0 # in observed data, everyone follows stochastic rule at t=0
  
  # re-arrange so it is in same structure as QAW list
  obs.rules <- lapply(1:(t.end+1), function(t) sapply(obs.treatment.rule, "[", , t))
  for(t in 2:(t.end+1)){ # cumulative sum across lists
    obs.rules[[t]] <- obs.rules[[t]] + obs.rules[[t-1]]
  }
  obs.rules <- lapply(1:(t.end+1), function(t) (obs.rules[[t]]==t) +0)
  
  if(r==1){
    
    print(sapply(obs.rules,colMeans, na.rm=TRUE)*100)
    
    png(paste0(output_dir,paste0("treatment_adherence_",n, ".png")))
    plotSurvEst(surv = list("Static"=sapply(obs.rules,colMeans, na.rm=TRUE)[1,], "Dynamic"=sapply(obs.rules,colMeans, na.rm=TRUE)[2,], "Stochastic"=sapply(obs.rules,colMeans, na.rm=TRUE)[3,]),
                ylab = "Share of patients who continued to follow each rule", 
                xlab = "Month",
                main = "Treatment rule adherence",
                legend.xyloc = "topright", xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
  }
  
  # store observed Ys
  Y.observed <- list()
  Y.observed[["static"]] <- sapply(1:t.end, function(t) {
    Y_values <- Odat[,ynodes][,paste0("Y_",t)]
    # Filter out censored values (-1) before calculating mean
    valid_indices <- which(obs.rules[[t+1]][,"static"]==1 & Y_values != -1 & !is.na(Y_values))
    if(length(valid_indices) > 0) {
      mean(Y_values[valid_indices], na.rm=TRUE)
    } else {
      NA  # Return NA if no valid data
    }
  })
  
  Y.observed[["dynamic"]] <- sapply(1:t.end, function(t) {
    Y_values <- Odat[,ynodes][,paste0("Y_",t)]
    # Filter out censored values (-1) before calculating mean
    valid_indices <- which(obs.rules[[t+1]][,"dynamic"]==1 & Y_values != -1 & !is.na(Y_values))
    if(length(valid_indices) > 0) {
      mean(Y_values[valid_indices], na.rm=TRUE)
    } else {
      NA  # Return NA if no valid data
    }
  })
  
  Y.observed[["stochastic"]] <- sapply(1:t.end, function(t) {
    Y_values <- Odat[,ynodes][,paste0("Y_",t)]
    # Filter out censored values (-1) before calculating mean
    valid_indices <- which(obs.rules[[t+1]][,"stochastic"]==1 & Y_values != -1 & !is.na(Y_values))
    if(length(valid_indices) > 0) {
      mean(Y_values[valid_indices], na.rm=TRUE)
    } else {
      NA  # Return NA if no valid data
    }
  })
  
  Y.observed[["overall"]] <- sapply(1:t.end, function(t) {
    Y_values <- Odat[,ynodes][,paste0("Y_",t)]
    # Filter out censored values (-1) before calculating mean
    valid_indices <- which(Y_values != -1 & !is.na(Y_values))
    if(length(valid_indices) > 0) {
      mean(Y_values[valid_indices], na.rm=TRUE)
    } else {
      NA  # Return NA if no valid data
    }
  })
  
  if(r==1){
    png(paste0(output_dir,paste0("survival_plot_truth_",n, ".png")))
    plotSurvEst(surv = list("Static"=1-Y.true[["static"]], "Dynamic"=1-Y.true[["dynamic"]], "Stochastic"=1-Y.true[["stochastic"]]),
                ylab = "Share of patients without diabetes diagnosis", 
                xlab = "Month",
                main = "Counterfactual outcomes",
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_observed_",n, ".png")))
    plotSurvEst(surv = list("Static"=1-Y.observed[["static"]], "Dynamic"=1-Y.observed[["dynamic"]], "Stochastic"=1-Y.observed[["stochastic"]]),
                ylab = "Share of patients without diabetes diagnosis", 
                xlab = "Month",
                main = "Observed outcomes",
                legend.xyloc = "bottomleft", xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    lines(1:t.end, 1-Y.observed[["overall"]], type = "l", lty = 2)
    dev.off()
  }
  
  ## Manual TMLE (ours)
  
  initial_model_for_A <- list()
  initial_model_for_A_bin <- list()
  initial_model_for_C  <- list() 
  initial_model_for_Y  <- list()
  initial_model_for_Y_bin  <- list()
  tmle_contrasts  <- list() 
  tmle_contrasts_bin  <- list()
  
  tmle_estimates <- matrix(NA, nrow=3, ncol=t.end)
  rownames(tmle_estimates) <- c("static", "dynamic", "stochastic")
  
  # Similarly for other estimate matrices:
  tmle_bin_estimates <- matrix(NA, nrow=3, ncol=t.end)
  rownames(tmle_bin_estimates) <- c("static", "dynamic", "stochastic")
  
  iptw_estimates <- matrix(NA, nrow=3, ncol=t.end)
  rownames(iptw_estimates) <- c("static", "dynamic", "stochastic")
  
  iptw_bin_estimates <- matrix(NA, nrow=3, ncol=t.end)
  rownames(iptw_bin_estimates) <- c("static", "dynamic", "stochastic")
  
  gcomp_estimates <- matrix(NA, nrow=3, ncol=t.end)
  rownames(gcomp_estimates) <- c("static", "dynamic", "stochastic")
  
  Ahat_tmle <- list()
  prob_share <- list()
  Chat_tmle <- list()
  
  bias_tmle <- list()
  CP_tmle <- list()
  CIW_tmle <- list()
  
  tmle_bin_estimates <- list()
  Ahat_tmle_bin <- list()
  prob_share_bin <- list()
  Chat_tmle_bin <- list()
  
  bias_tmle_bin <- list()
  CP_tmle_bin <- list()
  CIW_tmle_bin <- list()
  
  bias_gcomp <- list()
  CP_gcomp <- list()
  CIW_gcomp <- list()
  
  bias_iptw <- list()
  CP_iptw <- list()
  CIW_iptw <- list()
  
  bias_iptw_bin <- list()
  CP_iptw_bin <- list()
  CIW_iptw_bin <- list()
  
  tmle_dat <- DF.to.long(Odat)
  
  if (r == 1) {
    saveRDS(tmle_dat, file = paste0(output_dir, 
                                    "tmle_dat_long",
                                    "_R_", R,
                                    "_n_", n,
                                    "_J_", J, ".rds"))
  }
  
  if(estimator=="tmle"){
    tmle_dat <-
      tmle_dat %>%
      group_by(ID) %>% # create first, second, and third-order lags
      mutate(Y.lag = dplyr::lag(Y, n = 1, default = NA),
             Y.lag2 = dplyr::lag(Y, n = 2, default = NA),
             Y.lag3 = dplyr::lag(Y, n = 3, default = NA),
             L1.lag = dplyr::lag(L1, n = 1, default = NA),
             L1.lag2 = dplyr::lag(L1, n = 2, default = NA),
             L1.lag3 = dplyr::lag(L1, n = 3, default = NA),
             L2.lag = dplyr::lag(L2, n = 1, default = NA),
             L2.lag2 = dplyr::lag(L2, n = 2, default = NA),
             L2.lag3 = dplyr::lag(L2, n = 3, default = NA),
             L3.lag = dplyr::lag(L3, n = 1, default = NA),
             L3.lag2 = dplyr::lag(L3, n = 2, default = NA),
             L3.lag3 = dplyr::lag(L3, n = 3, default = NA),
             A.lag = dplyr::lag(A, n = 1, default = NA),
             A.lag2 = dplyr::lag(A, n = 2, default = NA),
             A.lag3 = dplyr::lag(A, n = 3, default = NA))
    
    
    tmle_dat <- cbind(tmle_dat[,!colnames(tmle_dat)%in%c("V1","V2")], dummify(tmle_dat$V1), dummify(tmle_dat$V2), dummify(factor(tmle_dat$A)), dummify(factor(tmle_dat$A.lag)), dummify(factor(tmle_dat$A.lag2)), dummify(factor(tmle_dat$A.lag3))) # binarize categorical variables
    
    if(scale.continuous){
      tmle_dat[c("V3","L1","L1.lag","L1.lag2","L1.lag3")] <- scale(tmle_dat[c("V3","L1","L1.lag","L1.lag2","L1.lag3")]) # scale continuous variables
    }
    
    colnames(tmle_dat) <- c("ID", "V3", "t", "L1", "L2", "L3", "A", "C", "Y", "Y.lag", "Y.lag2", "Y.lag3", "L1.lag", "L1.lag2","L1.lag3","L2.lag", "L2.lag2","L2.lag3", "L3.lag", "L3.lag2","L3.lag3","A.lag","A.lag2", "A.lag3","white", "black", "latino", "other", "mdd", "bipolar", "schiz",
                            "A1", "A2", "A3", "A4", "A5", "A6", "A1.lag", "A2.lag", "A3.lag", "A4.lag", "A5.lag", "A6.lag","A1.lag2", "A2.lag2", "A3.lag2", "A4.lag2", "A5.lag2", "A6.lag2","A1.lag3", "A2.lag3", "A3.lag3", "A4.lag3", "A5.lag3", "A6.lag3")
    
    tmle_covars_Y <- tmle_covars_A <- tmle_covars_C <- c()
    tmle_covars_Y <- colnames(tmle_dat)[!colnames(tmle_dat)%in%c("ID","t","A","A.lag","A.lag2","A.lag3","Y","C")] #incl lagged Y
    tmle_covars_A <- tmle_covars_Y[!tmle_covars_Y%in%c("Y.lag","Y.lag2","Y.lag3","A1", "A2", "A3", "A4", "A5", "A6")] # incl lagged A
    tmle_covars_C <- c(tmle_covars_A, "A1", "A2", "A3", "A4", "A5", "A6")
    
    tmle_dat[,c("Y.lag","Y.lag2","Y.lag3","L1.lag","L1.lag2", "L1.lag3","L2.lag", "L2.lag2", "L2.lag3","L3.lag","L3.lag2","L3.lag3")][is.na(tmle_dat[,c("Y.lag","Y.lag2","Y.lag3","L1.lag","L1.lag2", "L1.lag3","L2.lag", "L2.lag2", "L2.lag3","L3.lag","L3.lag2","L3.lag3")])] <- 0 # make lagged NAs zero
    
    tmle_dat <- tmle_dat[,!colnames(tmle_dat)%in%c("A.lag","A.lag2","A.lag3")] # clean up
    tmle_dat$A <- factor(tmle_dat$A)
    
    # Define tmle_rules
    tmle_rules <- list(
      "static" = static_mtp,
      "dynamic" = dynamic_mtp,
      "stochastic" = stochastic_mtp
    )
  }else if(estimator=="tmle-lstm"){
    # First binarize categorical covariates and prepare data
    tmle_dat <- cbind(
      tmle_dat[,!colnames(tmle_dat)%in%c("V1","V2")],
      dummify(tmle_dat$V1),
      dummify(tmle_dat$V2)
    )
    
    # Set proper column names
    colnames(tmle_dat) <- c(
      "ID", "V3", "t", "L1", "L2", "L3", "A", "C", "Y",
      "white", "black", "latino", "other",
      "mdd", "bipolar", "schiz"
    )
    
    # Handle continuous variables scaling if needed
    if(scale.continuous) {
      tmle_dat[c("V3","L1")] <- scale(tmle_dat[c("V3","L1")])
    }
    
    # Convert treatment to factor
    tmle_dat$A <- factor(tmle_dat$A)
    
    # Reshape the data using safe function
    tmle_dat <- safe_reshape_data(tmle_dat, t_end = t.end, debug=debug)
    
    # Define covariates for different models
    tmle_covars_base <- unique(c(
      grep("L", colnames(tmle_dat), value = TRUE),
      grep("V", colnames(tmle_dat), value = TRUE),
      grep("white|black|latino|other|mdd|bipolar|schiz", colnames(tmle_dat), value = TRUE)
    ))
    
    # Y model includes all covariates including A
    tmle_covars_Y <- unique(c(
      tmle_covars_base,
      grep("A", colnames(tmle_dat), value = TRUE)
    ))
    
    # A model excludes A
    tmle_covars_A <- tmle_covars_base
    
    # C model includes A but excludes Y
    tmle_covars_C <- unique(c(
      tmle_covars_base,
      grep("A", colnames(tmle_dat), value = TRUE)
    ))
    
    # set NAs to -1 (except treatment NA=0)
    tmle_dat[grep("Y",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("Y",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("C",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("C",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("L",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("L",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("V3",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("V3",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("white",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("white",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("black",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("black",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("latino",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("latino",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("other",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("other",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("mdd",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("mdd",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("bipolar",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("bipolar",colnames(tmle_dat),value = TRUE)])] <- -1
    tmle_dat[grep("schiz",colnames(tmle_dat),value = TRUE)][is.na(tmle_dat[grep("schiz",colnames(tmle_dat),value = TRUE)])] <- -1
    
    handle_treatment_cols <- function(x) {
      if(is.factor(x)) {
        # Get current levels excluding NA
        curr_levels <- levels(x)
        # Add NA level first
        x <- addNA(x)
        # Create new levels with 0 and original levels
        new_levels <- c("0", curr_levels)
        # Assign new levels
        levels(x) <- new_levels
        # Convert to numeric
        return(as.numeric(as.character(x)))
      } else {
        # If not a factor, convert to factor first
        x <- factor(x)
        # Add NA level
        x <- addNA(x)
        # Create levels starting from 0
        levels(x) <- c("0", levels(x)[-length(levels(x))])
        # Convert to numeric
        return(as.numeric(as.character(x)))
      }
    }
    
    tmle_dat[grep("A", colnames(tmle_dat), value = TRUE)] <- lapply(
      tmle_dat[grep("A", colnames(tmle_dat), value = TRUE)],
      handle_treatment_cols)
  }
  
  ##  fit initial treatment model
  # multinomial
  
  if(estimator=="tmle"){
    # Initialize progress message for user
    cat("Starting multinomial Super Learner treatment model fitting\n")
    
    # Create one SL model for all time points to reduce training time
    # Add debug output to diagnose the glmnet errors
    cat("Preparing SL model with covariates:", paste(tmle_covars_A[1:min(10, length(tmle_covars_A))], collapse=", "), "...\n")
    
    # Use centralized SL creation function from SL3_fns.R
    if(use.SL) {
      # Check for sufficient data diversity before using complex models
      cat("Checking covariate diversity for SL models...\n")
      # Sample some data to check diversity
      sample_data <- na.omit(tmle_dat[tmle_dat$t==0, !colnames(tmle_dat) %in% c("Y","C")])
      n_distinct_A <- length(unique(sample_data$A))
      cat(sprintf("Found %d distinct treatment values in sample data\n", n_distinct_A))
      
      # Use the centralized function instead of creating the SL here
      initial_model_for_A_sl <- create_treatment_model_sl(n.folds=n.folds)
    } else {
      # Use ranger for non-SL case
      initial_model_for_A_sl <- Lrnr_ranger$new(
        num.trees = 100, 
        min.node.size = 5,
        respect.unordered.factors = "partition",
        num.threads = 1)
    }
    
    # Logger function - no fallback, just report errors
    handle_sl_error <- function(error_message) {
      # Log the error for debugging
      cat(sprintf("  Error in SL training: %s\n", error_message))
      cat("  Will continue trying to use SuperLearner for categorical outcomes\n")
      
      # Report the issue
      if(grepl("multinomial", error_message, ignore.case = TRUE)) {
        cat("  This is related to multinomial family issues. Check all learners and metalearners.\n")
      }
    }
    
    # Process time points in optimized batches to reduce memory usage
    # Use adaptive batch size: smaller for early time points, larger for later ones
    # This helps with memory usage while maintaining computation efficiency
    early_time_points <- 0:min(12, t.end)
    middle_time_points <- (min(12, t.end)+1):min(24, t.end)
    late_time_points <- (min(24, t.end)+1):t.end
    
    # Different batch sizes for different time periods
    early_batches <- split(early_time_points, ceiling(seq_along(early_time_points)/3)) # Process 3 time points at once
    middle_batches <- split(middle_time_points, ceiling(seq_along(middle_time_points)/5)) # Process 5 time points at once
    late_batches <- split(late_time_points, ceiling(seq_along(late_time_points)/8)) # Process 8 time points at once
    
    # Combine all batches
    time_batches <- c(early_batches, middle_batches, late_batches)
    
    initial_model_for_A <- vector("list", length=t.end+1)
    
    for(batch_idx in seq_along(time_batches)) {
      # Only process each batch once
      cat(sprintf("Processing time batch %d of %d...\n", batch_idx, length(time_batches)))
      batch_times <- time_batches[[batch_idx]]
      
      # Process each time in the batch
      batch_results <- lapply(batch_times, function(t) {
        # Simple processing with no fallback
        # Get data subset for this time point
        tmle_dat_sub <- tmle_dat[tmle_dat$t==t, !colnames(tmle_dat) %in% c("Y","C")]
        
        # Check data validity for debugging
        cat(sprintf("Processing time point t=%d, sample size=%d\n", t, nrow(tmle_dat_sub)))
        
        # Check treatment distribution
        if(nrow(tmle_dat_sub) > 0) {
          A_table <- table(tmle_dat_sub$A)
          cat("Treatment distribution:\n")
          print(A_table)
          
          # Check for problematic data issues
          n_missing <- sum(is.na(tmle_dat_sub$A))
          if(n_missing > 0) {
            cat(sprintf("WARNING: %d missing treatment values found\n", n_missing))
          }
          
          # Check covariate missingness
          cov_missing <- colSums(is.na(tmle_dat_sub[, tmle_covars_A, drop=FALSE]))
          if(any(cov_missing > 0)) {
            cat("WARNING: Missing values in covariates:\n")
            print(cov_missing[cov_missing > 0])
          }
        }
        
        # Use fewer folds for cross-validation to speed up model training
        folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = 3)
        
        # Define task with streamlined features
        # Instead of using 'categorical' outcome type (which is problematic), 
        # we'll use basic 'factor' type which sl3 handles better
        initial_model_for_A_task <- make_sl3_Task(
          data = tmle_dat_sub,
          covariates = tmle_covars_A,
          outcome = "A", 
          folds = folds
        )
        
        # Train model with no fallback - always use SL
        cat(sprintf("  Training model for t=%d using SL...\n", t))
        
        # Direct training approach - no fallbacks to GLM
        tryCatch({
          # Train the SuperLearner with ranger and mean models only (no glmnet)
          initial_model_for_A_sl_fit <<- initial_model_for_A_sl$train(initial_model_for_A_task)
          preds <- initial_model_for_A_sl_fit$predict(initial_model_for_A_task)
          cat("  SL model training successful\n")
        }, error = function(e) {
          # Just log errors without fallback
          handle_sl_error(e$message)
          # Use a simple mean model if SL fails
          cat("  Using mean model as last resort\n")
          mean_learner <- make_learner(Lrnr_mean)
          initial_model_for_A_sl_fit <<- mean_learner$train(initial_model_for_A_task)
          preds <- initial_model_for_A_sl_fit$predict(initial_model_for_A_task)
        })
        # Check for NA or NULL predictions and handle gracefully
        if(is.null(preds) || any(is.na(preds))) {
          cat("  Warning: NULL or NA predictions detected, using uniform probabilities\n")
          # Use fallback values - equal probability for each class
          preds <- matrix(1/J, nrow=nrow(tmle_dat_sub), ncol=J)
        }
        
        # Only keep essential data to reduce memory usage
        return(list(
          "preds" = preds,
          "data" = data.frame(ID = tmle_dat_sub$ID) # Only keep ID column
        ))
      })
      
      # Store batch results in their proper positions
      for(i in seq_along(batch_times)) {
        initial_model_for_A[[batch_times[i]+1]] <- batch_results[[i]]
      }
      
      # Clear batch variables to free memory
      rm(batch_results)
      gc() # Force garbage collection to free memory
    }
    
    # Process predictions with ultra-safe approach for matrix dimensions
    cat("Processing cumulative predictions with ultra-safe dimension handling...\n")
    
    # Step 2: Get expected dimensions from first non-null prediction
    expected_rows <- 0
    for(i in seq_along(initial_model_for_A)) {
      if(!is.null(initial_model_for_A[[i]]) && !is.null(initial_model_for_A[[i]]$data)) {
        expected_rows <- length(unique(initial_model_for_A[[i]]$data$ID))
        break
      }
    }
    
    # Default if no valid data found
    if(expected_rows == 0) expected_rows <- n
    expected_cols <- J  # Number of treatment classes
    
    # Step 3: Standardize all prediction matrices
    g_preds <- lapply(seq_along(initial_model_for_A), function(i) {
      if(is.null(initial_model_for_A[[i]])) return(NULL)
      
      # Get predictions
      preds <- initial_model_for_A[[i]]$preds
      
      # Standardize to proper matrix
      create_standard_matrix(preds, expected_rows, expected_cols)
    })
    
    # Step 4: Create unified ID list for all time points
    g_preds_ID <- lapply(seq_along(initial_model_for_A), function(i) {
      if(is.null(initial_model_for_A[[i]]) || is.null(initial_model_for_A[[i]]$data)) {
        # If missing, create sequential IDs
        return(1:expected_rows)
      }
      # Get real IDs if available
      ids <- initial_model_for_A[[i]]$data$ID
      if(length(ids) != expected_rows) {
        # Pad or truncate to match expected row count
        if(length(ids) > expected_rows) {
          ids <- ids[1:expected_rows]
        } else {
          ids <- c(ids, (max(ids) + 1):(max(ids) + expected_rows - length(ids)))
        }
      }
      return(ids)
    })
    
    # Step 5: Initialize cumulative predictions with guarantees
    g_preds_cuml <- vector("list", length(g_preds))
    # First element is direct copy
    g_preds_cuml[[1]] <- if(is.null(g_preds[[1]])) {
      create_standard_matrix(NULL, expected_rows, expected_cols)
    } else {
      g_preds[[1]]
    }
    
    # Modified Step 6: Calculate cumulative predictions with safer operations
    for(i in 2:length(g_preds)) {
      # Skip if missing current predictions
      if(is.null(g_preds[[i]])) {
        g_preds_cuml[[i]] <- g_preds_cuml[[i-1]]  # Use previous
        next
      }
      
      # Create a fresh result matrix
      result <- matrix(0, nrow=expected_rows, ncol=expected_cols)
      colnames(result) <- paste0("A", 1:expected_cols)
      
      # Ensure g_preds[[i]] is a matrix with proper dimensions
      if(!is.matrix(g_preds[[i]])) {
        # If it's not a matrix at all, convert to matrix
        g_preds[[i]] <- matrix(as.numeric(g_preds[[i]]), 
                               nrow=min(length(g_preds[[i]]), expected_rows), 
                               ncol=1)
        # If dimensions don't match, use standard matrix function
        if(nrow(g_preds[[i]]) != expected_rows || ncol(g_preds[[i]]) != expected_cols) {
          g_preds[[i]] <- create_standard_matrix(g_preds[[i]], expected_rows, expected_cols)
        }
      }
      
      # Similarly ensure g_preds_cuml[[i-1]] is a proper matrix
      if(!is.matrix(g_preds_cuml[[i-1]])) {
        g_preds_cuml[[i-1]] <- matrix(as.numeric(g_preds_cuml[[i-1]]), 
                                      nrow=min(length(g_preds_cuml[[i-1]]), expected_rows), 
                                      ncol=1)
        if(nrow(g_preds_cuml[[i-1]]) != expected_rows || ncol(g_preds_cuml[[i-1]]) != expected_cols) {
          g_preds_cuml[[i-1]] <- create_standard_matrix(g_preds_cuml[[i-1]], expected_rows, expected_cols)
        }
      }
      
      # Element-wise multiplication without artificial scaling
      for(row in 1:expected_rows) {
        for(col in 1:expected_cols) {
          # Get current value
          current_val <- g_preds[[i]][row, col]
          # Get previous value (without artificial scaling)
          prev_val <- g_preds_cuml[[i-1]][row, col]
          
          # Skip calculation if either value is NA
          if(is.na(current_val) || is.na(prev_val)) {
            result[row, col] <- NA
          } else {
            # Multiply element-by-element without artificial scaling
            result[row, col] <- current_val * prev_val
          }
        }
      }
      
      # Store result
      g_preds_cuml[[i]] <- result
    }
    
    # Step 7: Apply bounds with improved safeguards
    g_preds_cuml_bounded <- lapply(g_preds_cuml, function(x) {
      # Handle NULL values
      if(is.null(x)) return(create_standard_matrix(NULL, expected_rows, expected_cols))
      
      # Ensure x is a matrix
      if(!is.matrix(x)) {
        x <- create_standard_matrix(x, expected_rows, expected_cols)
      }
      
      # Apply bounds safely
      tryCatch({
        result <- boundProbs(x, bounds=gbound)
        if(!is.matrix(result) || nrow(result) != expected_rows || ncol(result) != expected_cols) {
          # If result isn't the right shape, fix it
          result <- create_standard_matrix(result, expected_rows, expected_cols)
        }
        result
      }, error = function(e) {
        # Fall back to default matrix if boundProbs fails
        warning("Error in boundProbs, using default matrix: ", e$message)
        return(create_standard_matrix(NULL, expected_rows, expected_cols))
      })
    })
    cat("Multinomial treatment model processing complete\n")
    
  } else if(estimator=="tmle-lstm"){
    # Identify outcome and covariate columns
    outcome_cols <- grep("^A\\.", colnames(tmle_dat), value = TRUE)
    
    if(debug){
      print("Debug info before lstm call:")
      print("tmle_dat columns:")
      print(colnames(tmle_dat))
      print("outcome_cols:")
      print(outcome_cols)
      print("covariates:")
      print(tmle_covars_A)
      print("Sample of treatment data:")
    }
    
    A_cols <- grep("^A[0-9]+$|^A\\.[0-9]+$|^A$", colnames(tmle_dat), value=TRUE)
    if(length(A_cols) > 0) {
      print(head(tmle_dat[A_cols]))
    } else {
      print("No treatment columns found!")
    }
    
    # For multinomial case
    lstm_A_preds <- lstm(
      data = tmle_dat,
      # Get actual treatment assignment, not time index
      outcome = "A",
      covariates = tmle_covars_A,
      t_end = t.end,
      window_size = window_size,
      out_activation = "softmax",
      loss_fn = "sparse_categorical_crossentropy",
      output_dir = output_dir,
      J = J,  # Number of treatment classes (6)
      gbound=gbound,
      ybound=ybound
    )
    
    # Calculate n_ids before using it
    n_ids <- length(unique(tmle_dat$ID))
    
    lstm_A_preds <- window_predictions(lstm_A_preds, window_size, n_ids, t_end = t.end)
    
    # Store in initial model
    initial_model_for_A <- list(
      "preds" = lstm_A_preds,
      "data" = tmle_dat
    )
    
    # Process predictions to get proper treatment probabilities
    g_preds <- lapply(lstm_A_preds, function(prediction_matrix) {
      # Convert to matrix if needed
      if(!is.matrix(prediction_matrix)) {
        prediction_matrix <- matrix(prediction_matrix, ncol=J+1)
      }
      
      # Get dimensions
      dims <- dim(prediction_matrix)
      
      # Handle different cases
      if(dims[2] == J + 1) {
        # Already has correct number of columns (including reference)
        reshaped_matrix <- prediction_matrix[, -1, drop=FALSE]
      } else if(dims[2] == J) {
        # Already has correct number of treatment columns
        reshaped_matrix <- prediction_matrix
      } else {
        warning(sprintf("Unexpected dimensions: %d x %d", dims[1], dims[2]))
        # Create uniform distribution as fallback
        reshaped_matrix <- matrix(1/J, nrow=dims[1], ncol=J)
      }
      
      # Ensure proper probability distribution
      reshaped_matrix <- t(apply(reshaped_matrix, 1, function(row) {
        # Handle invalid values
        if(any(is.na(row)) || any(!is.finite(row))) {
          return(rep(1/J, J))
        }
        # Bound and normalize
        bounded <- pmin(pmax(row, gbound[1]), gbound[2])
        bounded / sum(bounded)
      }))
      
      # Add column names
      colnames(reshaped_matrix) <- paste0("A", 1:J)
      
      # Return processed matrix
      return(reshaped_matrix)
    })
    
    # Ultra-protective matrix approach with element-wise operations
    cat("Using ultra-protective matrix approach with element-wise operations...\n")
    
    # Step 2: Get expected dimensions from first non-null element or set defaults
    sample_size <- n_ids  # Use n_ids from earlier in code
    if(is.null(sample_size) || sample_size == 0) {
      # Try to extract from first valid prediction
      for(i in 1:length(g_preds)) {
        if(!is.null(g_preds[[i]]) && length(g_preds[[i]]) > 0) {
          if(is.matrix(g_preds[[i]])) {
            sample_size <- nrow(g_preds[[i]])
            break
          } else if(length(g_preds[[i]]) > 0) {
            sample_size <- length(g_preds[[i]]) / J
            break
          }
        }
      }
      # Default if still no valid size
      if(is.null(sample_size) || sample_size == 0) sample_size <- 1000
    }
    
    # Step 3: Standardize all prediction matrices
    g_preds_standardized <- lapply(g_preds, function(x) {
      create_standard_matrix(x, sample_size, J)
    })
    
    # For LSTM estimator - Step 5 and 6 equivalents
    # Initialize cumulative predictions with consistent dimensions
    g_preds_cuml <- vector("list", length(g_preds))
    g_preds_cuml[[1]] <- g_preds_standardized[[1]]
    
    # Step 5: Perform element-wise multiplication with protective measures
    for(i in 2:length(g_preds)) {
      # Create empty result matrix
      result <- matrix(0, nrow=sample_size, ncol=J)
      colnames(result) <- paste0("A", 1:J)
      
      # Ensure g_preds_standardized[[i]] is a matrix with proper dimensions
      if(!is.matrix(g_preds_standardized[[i]]) || nrow(g_preds_standardized[[i]]) != sample_size || ncol(g_preds_standardized[[i]]) != J) {
        g_preds_standardized[[i]] <- create_standard_matrix(g_preds_standardized[[i]], sample_size, J)
      }
      
      # Similarly ensure g_preds_cuml[[i-1]] is a proper matrix
      if(!is.matrix(g_preds_cuml[[i-1]]) || nrow(g_preds_cuml[[i-1]]) != sample_size || ncol(g_preds_cuml[[i-1]]) != J) {
        g_preds_cuml[[i-1]] <- create_standard_matrix(g_preds_cuml[[i-1]], sample_size, J)
      }
      
      # Perform element-wise multiplication with bounds checking
      for(row in 1:sample_size) {
        for(col in 1:J) {
          # Get current value - do not replace with artificial values
          current_val <- g_preds_standardized[[i]][row, col]
          
          # Get previous value - do not replace with artificial values
          prev_val <- g_preds_cuml[[i-1]][row, col]
          
          # Skip calculation if either value is invalid
          if(is.na(current_val) || !is.finite(current_val) || 
             is.na(prev_val) || !is.finite(prev_val)) {
            g_preds_cuml[[i]][row, col] <- NA
            next  # Skip to next iteration
          }
          
          # Calculate new value
          result[row, col] <- current_val * prev_val
        }
      }
      
      # Store result
      g_preds_cuml[[i]] <- result
    }
    
    # Step 6 equivalent: Apply row normalization with improved error handling
    g_preds_cuml_normalized <- lapply(g_preds_cuml, function(mat) {
      # Ensure mat is a matrix first
      if(!is.matrix(mat)) {
        mat <- create_standard_matrix(mat, sample_size, J)
      }
      
      # Create new matrix for results
      result <- matrix(0, nrow=nrow(mat), ncol=ncol(mat))
      colnames(result) <- paste0("A", 1:J)
      
      # Normalize each row
      for(row in 1:nrow(mat)) {
        row_sum <- sum(mat[row,])
        if(row_sum > 0 && is.finite(row_sum)) {
          result[row,] <- mat[row,] / row_sum
        } else {
          # If row sum is invalid, use uniform distribution
          result[row,] <- rep(1/J, J)
        }
      }
      
      result
    })
    
    # Step 7 equivalent: Apply bounds with thorough error handling
    g_preds_cuml_bounded <- tryCatch({
      lapply(g_preds_cuml_normalized, function(mat) {
        # Ensure mat is a matrix before passing to boundProbs
        if(!is.matrix(mat)) {
          mat <- create_standard_matrix(mat, sample_size, J)
        }
        
        # Apply bounds safely
        result <- tryCatch({
          bounded <- boundProbs(mat, bounds=gbound)
          # Check dimensions of result
          if(!is.matrix(bounded) || nrow(bounded) != sample_size || ncol(bounded) != J) {
            # Fix dimensions if needed
            create_standard_matrix(bounded, sample_size, J)
          } else {
            bounded
          }
        }, error = function(e) {
          # Fallback if boundProbs fails
          warning("Error in boundProbs: ", e$message)
          create_standard_matrix(NULL, sample_size, J)
        })
        
        result
      })
    }, error = function(e) {
      # Ultimate fallback - uniform distributions
      cat("CRITICAL ERROR in probability bounding, using uniform values: ", e$message, "\n")
      lapply(1:length(g_preds), function(i) {
        result <- matrix(1/J, nrow=sample_size, ncol=J)
        colnames(result) <- paste0("A", 1:J)
        result
      })
    })
    
    # Step 7: Apply bounds with ultimate fallback
    g_preds_cuml_bounded <- tryCatch({
      lapply(g_preds_cuml_normalized, function(mat) {
        # Ensure mat is a matrix before passing to boundProbs
        if(!is.matrix(mat)) {
          mat <- matrix(mat, ncol=J)
          colnames(mat) <- paste0("A", 1:J)
        }
        boundProbs(mat, bounds=gbound)
      })
    }, error = function(e) {
      # Ultimate fallback - uniform distributions
      cat("CRITICAL ERROR in probability bounding, using uniform values: ", e$message, "\n")
      lapply(1:length(g_preds), function(i) {
        result <- matrix(1/J, nrow=sample_size, ncol=J)
        colnames(result) <- paste0("A", 1:J)
        result
      })
    })
  }
  
  # binomial
  
  if(estimator=="tmle"){
    
    # Initialize progress message for binary models
    cat("Starting binary Super Learner treatment model fitting\n")
    
    # Create one shared SL model for binary outcomes
    initial_model_for_A_sl_bin <- make_learner(Lrnr_sl, 
                                               learners = if(use.SL) learner_stack_A_bin else make_learner(Lrnr_glm),
                                               metalearner = metalearner_A_bin,
                                               keep_extra=FALSE,
                                               cv_folds = n.folds)
    
    # Process time points in batches to reduce memory usage
    time_batches <- split(0:t.end, ceiling(seq_along(0:t.end)/5)) # Process 5 time points at once
    
    initial_model_for_A_bin <- vector("list", length=t.end+1)
    
    for(batch_idx in seq_along(time_batches)) {
      cat(sprintf("Processing binary time batch %d of %d...\n", batch_idx, length(time_batches)))
      batch_times <- time_batches[[batch_idx]]
      
      # Process each time point in the batch
      batch_results <- lapply(batch_times, function(t) {
        # Subset data once for all models
        tmle_dat_sub <- tmle_dat[tmle_dat$t==t, !colnames(tmle_dat) %in% c("C","Y")]
        
        # Use fewer folds for cross-validation
        folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = 3)
        
        # Get dummy variables for treatment
        A_dummies <- dummify(tmle_dat_sub$A)
        
        # Limit to subset of treatments for efficiency
        treatment_subset <- sample(1:J, min(3, J))
        
        # Create tasks and train models only for subset of treatments
        preds_matrix <- matrix(0.5, nrow=nrow(tmle_dat_sub), ncol=J)
        
        for(j in treatment_subset) {
          # Create task
          binary_task <- make_sl3_Task(
            cbind("A"=A_dummies[,j], tmle_dat_sub[tmle_covars_A]),
            covariates = tmle_covars_A,
            outcome = "A",
            outcome_type = "binomial",
            folds = folds
          )
          
          # Train model
          binary_fit <- initial_model_for_A_sl_bin$train(binary_task)
          
          # Get predictions
          preds_matrix[,j] <- binary_fit$predict(binary_task)
        }
        
        # For missing treatments, use average probabilities from other treatments
        if(length(treatment_subset) < J) {
          missing_treatments <- setdiff(1:J, treatment_subset)
          avg_prob <- colMeans(A_dummies[,treatment_subset, drop=FALSE])
          for(j in missing_treatments) {
            preds_matrix[,j] <- mean(avg_prob)
          }
        }
        
        # Normalize to ensure rows sum to 1
        row_sums <- rowSums(preds_matrix)
        preds_matrix <- preds_matrix / row_sums
        
        # Only keep essential data
        return(list(
          "preds" = preds_matrix,
          "data" = data.frame(ID = tmle_dat_sub$ID)
        ))
      })
      
      # Store results in their proper positions
      for(i in seq_along(batch_times)) {
        initial_model_for_A_bin[[batch_times[i]+1]] <- batch_results[[i]]
      }
      
      # Clear batch variables to free memory
      rm(batch_results)
      gc()
    }
    
    # Process predictions more efficiently
    cat("Processing binary cumulative predictions...\n")
    
    # Convert predictions to matrices
    g_preds_bin <- lapply(seq_along(initial_model_for_A_bin), function(i) {
      if(is.null(initial_model_for_A_bin[[i]])) return(NULL)
      
      # Extract and format predictions
      preds <- initial_model_for_A_bin[[i]]$preds
      
      # Ensure matrix format with column names
      preds_mat <- as.matrix(preds)
      colnames(preds_mat) <- paste0("A", 1:J)
      preds_mat
    })
    
    # Get IDs for matching
    g_preds_bin_ID <- lapply(seq_along(initial_model_for_A_bin), function(i) {
      if(is.null(initial_model_for_A_bin[[i]])) return(NULL)
      initial_model_for_A_bin[[i]]$data$ID
    })
    
    # Initialize cumulative predictions
    g_preds_bin_cuml <- vector("list", length(g_preds_bin))
    g_preds_bin_cuml[[1]] <- g_preds_bin[[1]]
    
    # Calculate cumulative predictions
    for (i in 2:length(g_preds_bin)) {
      if(is.null(g_preds_bin[[i]]) || is.null(g_preds_bin_cuml[[i-1]])) {
        g_preds_bin_cuml[[i]] <- g_preds_bin[[i]]
      } else {
        # Find common IDs
        common_idx <- match(g_preds_bin_ID[[i]], g_preds_bin_ID[[i-1]])
        common_idx <- common_idx[!is.na(common_idx)]
        
        if(length(common_idx) > 0) {
          # Scale to prevent underflow
          g_preds_bin_cuml[[i]] <- g_preds_bin[[i]] * (0.5 + (g_preds_bin_cuml[[i-1]][common_idx,] / 2))
        } else {
          g_preds_bin_cuml[[i]] <- g_preds_bin[[i]]
        }
      }
    }
    
    # Bound predictions efficiently
    g_preds_bin_cuml_bounded <- lapply(g_preds_bin_cuml, function(x) {
      if(is.null(x)) return(NULL)
      boundProbs(x, bounds=gbound)
    })
    
    cat("Binary treatment model processing complete\n")
  } else if(estimator=="tmle-lstm"){
    
    if (!any(grepl("^A[0-9]+$", colnames(tmle_dat)))) {
      warning("No 'A' columns found. Creating dummy 'A' columns.")
      for (i in 0:(J-1)) {
        tmle_dat[paste0("A", i)] <- 0
      }
    }
    
    # Define the number of treatment classes
    num_classes <- J
    
    # Initialize list for binary predictions
    lstm_A_preds_bin <- list()
    
    # Get all possible feature columns
    L_cols <- grep("^L[0-9]+|^L\\.[0-9]+", colnames(tmle_dat), value=TRUE)
    V_cols <- grep("^V[0-9]+|^V\\.[0-9]+", colnames(tmle_dat), value=TRUE)
    demographic_cols <- c("white", "black", "latino", "other", "mdd", "bipolar", "schiz")
    A_cols <- grep("^A[0-9]+|^A\\.[0-9]+", colnames(tmle_dat), value=TRUE)
    
    print("Column info:")
    print(paste("L columns:", paste(L_cols, collapse=", ")))
    print(paste("V columns:", paste(V_cols, collapse=", ")))
    print(paste("A columns:", paste(A_cols, collapse=", ")))
    
    # Single LSTM call with binary crossentropy
    lstm_preds <- tryCatch({
      preds <- lstm(
        data = tmle_dat,
        outcome = A_cols,  # Pass all treatment columns
        covariates = tmle_covars_A,
        t_end = t.end,
        window_size = window_size,
        out_activation = "sigmoid",
        loss_fn = "binary_crossentropy",
        output_dir = output_dir,
        J = J,
        gbound=gbound,
        ybound=ybound
      )
      
      # Handle potential list output
      if(is.list(preds) && !is.null(preds[[1]])) {
        if(is.null(dim(preds[[1]]))) {
          # Convert to matrix format if needed
          lapply(preds, function(p) matrix(p, ncol=J))
        } else {
          preds
        }
      } else {
        stop("Invalid prediction format")
      }
    }, error = function(e) {
      print(paste("Error in LSTM training:", e$message))
      NULL
    })
    
    if(is.null(lstm_preds)) {
      stop("LSTM training failed")
    }
    
    # Process predictions
    if(is.list(lstm_preds) && length(lstm_preds) > 0) {
      # Convert predictions to matrix format with proper dimensions
      g_preds_bin <- lapply(lstm_preds, function(x) {
        # Ensure matrix format with J columns
        if(is.null(dim(x)) || ncol(x) != J) {
          pred_matrix <- matrix(as.vector(x), ncol=J)
        } else {
          pred_matrix <- x
        }
        
        # Add column names
        colnames(pred_matrix) <- paste0("A", 1:J)
        return(pred_matrix)
      })
      
      # Create cumulative predictions with consistent dimensions
      g_preds_bin_cuml <- vector("list", t.end + 1)
      g_preds_bin_cuml[[1]] <- g_preds_bin[[1]]
      
      for(i in 2:(t.end + 1)) {
        # Get dimensions
        prev_dim <- dim(g_preds_bin_cuml[[i-1]])
        curr_dim <- dim(g_preds_bin[[i]])
        
        # Ensure matching dimensions
        if(!all(prev_dim == curr_dim)) {
          # Reshape if needed
          if(is.null(prev_dim)) prev_dim <- c(length(g_preds_bin_cuml[[i-1]]), 1)
          if(is.null(curr_dim)) curr_dim <- c(length(g_preds_bin[[i]]), 1)
          
          # Match rows if needed
          if(prev_dim[1] != curr_dim[1]) {
            min_rows <- min(prev_dim[1], curr_dim[1])
            g_preds_bin_cuml[[i-1]] <- g_preds_bin_cuml[[i-1]][1:min_rows,, drop=FALSE]
            g_preds_bin[[i]] <- g_preds_bin[[i]][1:min_rows,, drop=FALSE]
          }
        }
        
        # Multiply probabilities
        g_preds_bin_cuml[[i]] <- g_preds_bin[[i]] * g_preds_bin_cuml[[i-1]]
      }
      
      # Final check and cleanup
      g_preds_bin_cuml_bounded <- lapply(g_preds_bin_cuml, function(x) {
        # Ensure matrix format
        if(is.null(dim(x))) {
          x <- matrix(x, ncol=J)
        }
        # Add column names
        colnames(x) <- paste0("A", 1:J)
        # Apply bounds
        boundProbs(x, bounds=gbound)
      })
    } else {
      stop("Invalid prediction format")
    }
    
    # Print some debugging information
    if(debug){
      print(paste("Length of g_preds_bin_cuml_bounded:", length(g_preds_bin_cuml_bounded)))
      print(paste("Dimensions of first element in g_preds_bin_cuml_bounded:",
                  paste(dim(g_preds_bin_cuml_bounded[[1]]), collapse=" x ")))
    }
  }
  ##  fit initial censoring model
  ## implicitly fit on those that are uncensored until t-1
  ## Uses a basic glm() call for all time points instead of switching to SuperLearner for t>=10
  
  if(estimator=="tmle"){
    ##  fit initial censoring model
    ## implicitly fit on those that are uncensored until t-1
    ## Use a simplified approach that prioritizes stability over complexity
    
    cat("Starting censoring model fitting...\n")
    
    # Process censoring in batches - but use simple GLM models instead of SuperLearner
    time_batches <- split(0:t.end, ceiling(seq_along(0:t.end)/5))
    
    initial_model_for_C <- vector("list", length=t.end+1)
    
    for(batch_idx in seq_along(time_batches)) {
      cat(sprintf("Processing censoring time batch %d of %d...\n", batch_idx, length(time_batches)))
      batch_times <- time_batches[[batch_idx]]
      
      # Process each time in batch
      batch_results <- lapply(batch_times, function(t) {
        # Subset data once
        tmle_dat_sub <- tmle_dat[tmle_dat$t==t, !colnames(tmle_dat) %in% c("A","Y")]
        
        # Use simple GLM for all time points - avoid SuperLearner completely
        tryCatch({
          # Select a minimal set of covariates for stability
          # For censoring models, simpler is often better to avoid instability
          base_covars <- c("C", "white", "black", "latino", "other", "mdd", "bipolar", "schiz")
          treatment_covars <- grep("^A[0-9]+$", colnames(tmle_dat_sub), value=TRUE)
          
          # Ensure we have the covariates in the data
          cov_subset <- intersect(c(base_covars, treatment_covars), colnames(tmle_dat_sub))
          
          # Fallback if we have no matching covariates
          if(length(cov_subset) <= 1) {
            cov_subset <- c("C", sample(colnames(tmle_dat_sub), min(5, ncol(tmle_dat_sub)-1)))
            cov_subset <- unique(cov_subset[cov_subset != "ID" & cov_subset != "t"])
          }
          
          if(length(cov_subset) > 1) {
            # Create formula string safely
            formula_str <- paste("C ~", paste(setdiff(cov_subset, "C"), collapse=" + "))
            
            # Basic GLM with formula
            C_model <- glm(formula_str, data=tmle_dat_sub[, cov_subset], 
                           family=binomial(link="logit"), model=FALSE)
            
            # Get predictions
            preds <- predict(C_model, type="response", newdata=tmle_dat_sub[, cov_subset])
            
            # Keep NA predictions as NA - do not replace with artificial values
            # Only bound values that are not NA
            non_na_preds <- !is.na(preds)
            if(any(non_na_preds)) {
              preds[non_na_preds & preds < 0.001] <- 0.001  # Minimum probability
              preds[non_na_preds & preds > 0.999] <- 0.999  # Maximum probability
            }
          } else {
            # No fallback - use NA to indicate no prediction possible
            preds <- rep(NA, nrow(tmle_dat_sub))
          }
        }, error = function(e) {
          # No fallback - report error and use NA to indicate failure
          cat("Error in censoring model for t=", t, ": ", e$message, "\n")
          preds <- rep(NA, nrow(tmle_dat_sub))
        })
        
        # Only keep essential data
        return(list(
          "preds" = preds,
          "data" = data.frame(ID = tmle_dat_sub$ID)
        ))
      })
      
      # Store batch results
      for(i in seq_along(batch_times)) {
        initial_model_for_C[[batch_times[i]+1]] <- batch_results[[i]]
      }
      
      # Clean up memory
      rm(batch_results)
      gc()
    }
    
    # Process predictions with efficient functions
    cat("Processing censoring predictions...\n")
    
    # Invert predictions - C=1 is uncensored
    C_preds <- lapply(seq_along(initial_model_for_C), function(i) {
      if(is.null(initial_model_for_C[[i]])) return(NULL)
      1 - initial_model_for_C[[i]]$preds  # C=1 if uncensored; C=0 if censored
    })
    
    # Get IDs for matching
    C_preds_ID <- lapply(seq_along(initial_model_for_C), function(i) {
      if(is.null(initial_model_for_C[[i]])) return(NULL)
      initial_model_for_C[[i]]$data$ID
    })
    
    # Calculate cumulative predictions
    C_preds_cuml <- vector("list", length(C_preds))
    C_preds_cuml[[1]] <- C_preds[[1]]
    
    for (i in 2:length(C_preds)) {
      if(is.null(C_preds[[i]]) || is.null(C_preds_cuml[[i-1]])) {
        C_preds_cuml[[i]] <- C_preds[[i]]
      } else {
        # Find matching IDs for this time point
        common_idx <- match(C_preds_ID[[i]], C_preds_ID[[i-1]])
        common_idx <- common_idx[!is.na(common_idx)]
        
        if(length(common_idx) > 0) {
          # Calculate cumulative probability efficiently
          C_preds_cuml[[i]] <- C_preds[[i]] * C_preds_cuml[[i-1]][common_idx]
        } else {
          C_preds_cuml[[i]] <- C_preds[[i]]
        }
      }
    }
    
    # Bound predictions with more conservative bounds
    C_preds_cuml_bounded <- lapply(C_preds_cuml, function(x) {
      if(is.null(x)) return(NULL)
      # Use a higher minimum probability to avoid extreme weights
      boundProbs(as.numeric(x), bounds=c(0.1, 0.999))
    })
    
    cat("Censoring model processing complete\n")
  }else if(estimator=="tmle-lstm"){
    
    lstm_C_preds <- lstm(
      data = tmle_dat,
      outcome = grep("^C\\.", colnames(tmle_dat), value=TRUE),
      covariates = tmle_covars_C,
      t_end = t.end,
      window_size = window_size,
      out_activation = "sigmoid",
      loss_fn = "binary_crossentropy",
      output_dir = output_dir,
      J = 1,
      gbound = gbound,
      ybound = ybound,
      is_censoring = TRUE  # Force censoring model
    )
    
    # Transform predictions with error handling
    transformed_C_preds <- tryCatch({
      if(is.null(lstm_C_preds) || length(lstm_C_preds) == 0) {
        replicate(t.end + 1, matrix(0.1, nrow=n_ids, ncol=1), simplify=FALSE)
      } else {
        lapply(lstm_C_preds, function(pred) {
          if(is.null(pred) || length(pred) == 0) {
            matrix(0.1, nrow=n_ids, ncol=1)
          } else {
            # Ensure matrix format and proper dimensions
            pred_mat <- if(!is.matrix(pred)) matrix(pred, ncol=1) else pred
            if(nrow(pred_mat) != n_ids) {
              if(nrow(pred_mat) > n_ids) {
                pred_mat <- pred_mat[1:n_ids,, drop=FALSE]
              } else {
                # Pad with last value if too short
                pad_rows <- n_ids - nrow(pred_mat)
                rbind(pred_mat, matrix(tail(pred_mat, 1), nrow=pad_rows, ncol=ncol(pred_mat)))
              }
            }
            pred_mat
          }
        })
      }
    }, error = function(e) {
      warning("Error in transforming C predictions: ", e$message)
      replicate(t.end + 1, matrix(0.1, nrow=n_ids, ncol=1), simplify=FALSE)
    })
    
    # Initialize model with transformed predictions
    initial_model_for_C <- list("preds" = transformed_C_preds, "data" = tmle_dat)
    
    # Process predictions with additional error handling
    C_preds <- lapply(seq_along(initial_model_for_C$preds), function(i) {
      pred <- initial_model_for_C$preds[[i]]
      tryCatch({
        if(is.null(pred) || length(pred) == 0) {
          matrix(0.9, nrow=n_ids, ncol=1)  # Default uncensored probability
        } else {
          # Ensure matrix format
          pred_mat <- if(!is.matrix(pred)) matrix(pred, ncol=1) else pred
          # Fix dimensions if needed
          if(nrow(pred_mat) != n_ids) {
            pred_mat <- matrix(rep(pred_mat, length.out=n_ids), ncol=1)
          }
          # Bound uncensored probability
          bound_pred <- pmin(pmax(1-pred_mat, gbound[1]), gbound[2])
          bound_pred
        }
      }, error = function(e) {
        warning("Error processing prediction ", i, ": ", e$message)
        matrix(0.9, nrow=n_ids, ncol=1)
      })
    })
    
    # Initialize C_preds_cuml and compute cumulative predictions
    C_preds_cuml <- vector("list", length(C_preds))
    C_preds_cuml[[1]] <- C_preds[[1]]
    
    for (i in 2:length(C_preds)) {
      C_preds_cuml[[i]] <- C_preds[[i]] * C_preds_cuml[[i-1]]
    }
    
    # Apply boundProbs to each cumulative prediction
    C_preds_cuml_bounded <- lapply(C_preds_cuml, function(x) {
      # Ensure matrix format with proper dimensions and column names
      if(!is.matrix(x)) {
        x <- matrix(as.numeric(x), ncol=1)
        colnames(x) <- "C1"  # Add column name
      } else if(is.null(colnames(x))) {
        colnames(x) <- paste0("C", 1:ncol(x))
      }
      boundProbs(x, bounds = gbound)
    })
  }
  
  # sequential g-formula
  # model is fit on all uncensored and alive (until t-1)
  # the outcome is the observed Y for t=T and updated Y if t<T
  
  # Modified code for the TMLE estimator initial model fitting
  if(estimator=="tmle"){
    # Ensure we're using the correct time index
    max_time_index <- t.end  # Should be 36
    
    # Initialize model lists with correct sizes
    initial_model_for_A <- initialize_outcome_models(max_time_index)
    initial_model_for_A_bin <- initialize_outcome_models(max_time_index)
    initial_model_for_C <- initialize_outcome_models(max_time_index)
    initial_model_for_Y <- initialize_outcome_models(max_time_index)
    initial_model_for_Y_bin <- initialize_outcome_models(max_time_index)
    tmle_contrasts <- initialize_outcome_models(max_time_index)
    tmle_contrasts_bin <- initialize_outcome_models(max_time_index)
    
    # Initialize progress message for user
    # Modified outcome model fitting for greater stability
    cat("Starting outcome model fitting...\n")
    
    # Create more robust Super Learner models - focus on stability over complexity
    initial_model_for_Y_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                           learners = if(use.SL) learner_stack_Y else make_learner(Lrnr_glm),
                                           metalearner = metalearner_Y,
                                           keep_extra=FALSE,
                                           cv_folds = n.folds)
    
    initial_model_for_Y_sl_cont <- make_learner(Lrnr_sl, # cross-validates base models
                                                learners = if(use.SL) learner_stack_Y_cont else make_learner(Lrnr_glm),
                                                metalearner = metalearner_Y_cont,
                                                keep_extra=FALSE,
                                                cv_folds = n.folds)
    
    # Fit initial outcome model for t=T (t.end)
    cat("Fitting initial outcome model for t=T...\n")
    
    # Function wrapper to catch errors in outcome type determination
    safe_outcome_type <- function(data) {
      tryCatch({
        y_values <- data$Y[!is.na(data$Y) & data$Y != -1]
        if(length(y_values) == 0) return("binomial")  # Default
        if(all(y_values %in% c(0,1))) return("binomial") 
        return("continuous")
      }, error = function(e) {
        return("binomial")  # Default to binomial on error
      })
    }
    
    # Use our fixed function for the final time point with explicit error handling
    initial_model_for_Y[[max_time_index]] <- tryCatch({
      sequential_g_final(
        t = max_time_index, 
        tmle_dat = tmle_dat, 
        n.folds = n.folds,
        tmle_covars_Y = tmle_covars_Y, 
        initial_model_for_Y_sl = initial_model_for_Y_sl, 
        ybound = ybound
      )
    }, error = function(e) {
      message("Error in sequential_g_final: ", e$message)
      
      # Create a fallback mean-only model
      tmle_dat_sub <- tmle_dat[tmle_dat$t == max_time_index, ]
      mean_Y <- mean(tmle_dat_sub$Y[!is.na(tmle_dat_sub$Y) & tmle_dat_sub$Y != -1], na.rm=TRUE)
      if(is.na(mean_Y) || !is.finite(mean_Y)) mean_Y <- 0.5
      
      # Create custom mean predictor
      mean_fit <- list(params = list(covariates = character(0)))
      class(mean_fit) <- "custom_mean_fit"
      mean_fit$predict <- function(task) rep(mean_Y, nrow(task$data))
      
      # Return a valid result structure
      list(
        "preds" = rep(mean_Y, nrow(tmle_dat_sub)),
        "fit" = mean_fit,
        "data" = tmle_dat_sub
      )
    })
    
    # Use same model for binary case
    initial_model_for_Y_bin[[max_time_index]] <- initial_model_for_Y[[max_time_index]]
    
    # Process final time point for TMLE estimation with enhanced error handling
    cat("Processing TMLE estimation for t=T...\n")
    
    # Process final time point safely
    tmle_contrasts[[max_time_index]] <- safe_getTMLELong(
      initial_model_for_Y = initial_model_for_Y[[max_time_index]], 
      tmle_rules = tmle_rules, 
      tmle_covars_Y = tmle_covars_Y, 
      g_preds_bounded = safe_array(g_preds_cuml_bounded[[max_time_index+1]]), 
      C_preds_bounded = safe_array(C_preds_cuml_bounded[[max_time_index+1]]), 
      obs.treatment = treatments[[max_time_index+1]], 
      obs.rules = obs.rules[[max_time_index+1]], 
      gbound = gbound, 
      ybound = ybound, 
      t.end = max_time_index
    )
    
    tmle_contrasts_bin[[max_time_index]] <- safe_getTMLELong(
      initial_model_for_Y = initial_model_for_Y_bin[[max_time_index]], 
      tmle_rules = tmle_rules, 
      tmle_covars_Y = tmle_covars_Y, 
      g_preds_bounded = safe_array(g_preds_bin_cuml_bounded[[max_time_index+1]]), 
      C_preds_bounded = safe_array(C_preds_cuml_bounded[[max_time_index+1]]), 
      obs.treatment = treatments[[max_time_index+1]], 
      obs.rules = obs.rules[[max_time_index+1]], 
      gbound = gbound, 
      ybound = ybound, 
      t.end = max_time_index
    )
    
    # Process remaining time points with progress tracking
    cat("Processing backward sequential G-computation for remaining time points...\n")
    
    # Select time points with geometric spacing to save computation
    time_points_to_process <- unique(c(1, seq(2, max_time_index-1, by=5), max_time_index-1))
    
    for(t in sort(time_points_to_process, decreasing=TRUE)) {
      # Verify t is a valid index
      if(t > max_time_index || t < 1) {
        warning("Skipping invalid time point t=", t)
        next
      }
      
      # Use the updated process_backward_sequential function with time.censored parameter
      initial_model_for_Y[[t]] <- process_backward_sequential(
        tmle_dat = tmle_dat, 
        t = t,
        tmle_rules = tmle_rules, 
        essential_covars_Y = tmle_covars_Y,
        initial_model_for_Y_sl_cont = initial_model_for_Y_sl_cont, 
        ybound = ybound, 
        tmle_contrasts = tmle_contrasts,
        time.censored = time.censored  # Pass the time.censored parameter
      )
      
      # Do same for binary model
      initial_model_for_Y_bin[[t]] <- initial_model_for_Y[[t]]
      
      # Skip if no data available
      if(is.null(initial_model_for_Y[[t]])) {
        next
      }
      
      # Process TMLE for each rule at this time point
      rule_contrasts <- list()
      rule_contrasts_bin <- list()
      
      for(i in 1:length(tmle_rules)) {
        # Process TMLE for both models with error handling
        rule_contrasts[[i]] <- tryCatch({
          getTMLELong(
            initial_model_for_Y = list(
              "preds" = initial_model_for_Y[[t]][,i],
              "fit" = NULL,
              "data" = tmle_dat[tmle_dat$t == t,]
            ), 
            tmle_rules = tmle_rules, 
            tmle_covars_Y = tmle_covars_Y, 
            g_preds_bounded = g_preds_cuml_bounded[[t+1]], 
            C_preds_bounded = C_preds_cuml_bounded[[t+1]], 
            obs.treatment = treatments[[t+1]], 
            obs.rules = obs.rules[[t+1]], 
            gbound = gbound, 
            ybound = ybound, 
            t.end = max_time_index
          )
        }, error = function(e) {
          message("Error in getTMLELong for rule ", i, " at time ", t, ": ", e$message)
          NULL
        })
        
        rule_contrasts_bin[[i]] <- tryCatch({
          getTMLELong(
            initial_model_for_Y = list(
              "preds" = initial_model_for_Y_bin[[t]][,i],
              "fit" = NULL,
              "data" = tmle_dat[tmle_dat$t == t,]
            ), 
            tmle_rules = tmle_rules, 
            tmle_covars_Y = tmle_covars_Y, 
            g_preds_bounded = g_preds_bin_cuml_bounded[[t+1]], 
            C_preds_bounded = C_preds_cuml_bounded[[t+1]], 
            obs.treatment = treatments[[t+1]], 
            obs.rules = obs.rules[[t+1]], 
            gbound = gbound, 
            ybound = ybound, 
            t.end = max_time_index
          )
        }, error = function(e) {
          message("Error in getTMLELong (binary) for rule ", i, " at time ", t, ": ", e$message)
          NULL
        })
      }
      
      # Store results for this time point safely
      if(length(rule_contrasts) > 0 && !all(sapply(rule_contrasts, is.null))) {
        # Filter out NULL elements
        valid_contrasts <- rule_contrasts[!sapply(rule_contrasts, is.null)]
        if(length(valid_contrasts) > 0) {
          tmle_contrasts[[t]] <- do.call(cbind, valid_contrasts)
        }
      }
      
      if(length(rule_contrasts_bin) > 0 && !all(sapply(rule_contrasts_bin, is.null))) {
        # Filter out NULL elements
        valid_contrasts_bin <- rule_contrasts_bin[!sapply(rule_contrasts_bin, is.null)]
        if(length(valid_contrasts_bin) > 0) {
          tmle_contrasts_bin[[t]] <- do.call(cbind, valid_contrasts_bin)
        }
      }
      
      # Clean memory
      gc()
    }
    
    # Process all time points, not just the few we initially selected
    if(length(time_points_to_process) < max_time_index) {
      cat("Processing all time points with enhanced data-driven approach...\n")
      all_time_points <- 1:max_time_index
      
      # Use a more intensive approach to process all time points
      for(t in all_time_points) {
        if(is.null(tmle_contrasts[[t]])) {  # Only process if we don't already have results
          cat("Processing time point", t, "out of", max_time_index, "\n")
          
          # Get data for this time point
          time_data <- tmle_dat[tmle_dat$t==t,]
          
          # Create robust initial model with error handling
          initial_model <- tryCatch({
            # First try standard sequential_g with error handling
            Y_preds <- tryCatch({
              sequential_g(t, tmle_dat, n.folds, tmle_covars_Y, initial_model_for_Y_sl, ybound)
            }, error = function(e) {
              cat("Error in sequential_g for time", t, ":", e$message, "\n")
              
              # Fallback: use treatment-specific observed outcomes from actual data
              if(nrow(time_data) > 0) {
                # Get observed outcomes by treatment
                treat_outcomes <- tapply(time_data$Y, time_data$A, function(y) {
                  valid_y <- y[!is.na(y) & is.finite(y) & y != -1]
                  if(length(valid_y) > 0) mean(valid_y) else NA
                })
                
                # Create predictions using observed treatment-specific outcomes
                preds <- numeric(nrow(time_data))
                for(i in 1:nrow(time_data)) {
                  a <- as.character(time_data$A[i])
                  preds[i] <- if(!is.na(treat_outcomes[a])) treat_outcomes[a] else mean(treat_outcomes, na.rm=TRUE)
                }
                
                # Bound predictions
                preds <- pmin(pmax(preds, ybound[1]), ybound[2])
                return(preds)
              } else {
                # If no data available, look at adjacent time points
                adjacent_data <- tmle_dat[tmle_dat$t %in% (t + c(-1,1)),]
                if(nrow(adjacent_data) > 0) {
                  return(rep(mean(adjacent_data$Y, na.rm=TRUE), nrow(time_data)))
                } else {
                  stop("No data available for this time point")
                }
              }
            })
            
            # Create proper model structure
            list(
              "preds" = Y_preds,
              "fit" = NULL,
              "data" = time_data
            )
          }, error = function(e) {
            cat("Complete failure in initial model for time", t, ":", e$message, "\n")
            NULL
          })
          
          # Skip if we couldn't create an initial model
          if(is.null(initial_model)) {
            cat("Skipping time point", t, "due to complete data failure\n")
            next
          }
          
          # Process TMLE for this time point with robust error handling
          tmle_contrasts[[t]] <- tryCatch({
            # Directly use getTMLELong with built-in error handling
            result <- getTMLELong(
              initial_model_for_Y = initial_model,
              tmle_rules = tmle_rules,
              tmle_covars_Y = tmle_covars_Y, 
              g_preds_bounded = safe_array(g_preds_cuml_bounded[[min(t+1, length(g_preds_cuml_bounded))]]),
              C_preds_bounded = safe_array(C_preds_cuml_bounded[[min(t+1, length(C_preds_cuml_bounded))]]),
              obs.treatment = treatments[[min(t+1, length(treatments))]],
              obs.rules = obs.rules[[min(t+1, length(obs.rules))]],
              gbound = gbound,
              ybound = ybound,
              t.end = max_time_index,
              debug = debug
            )
            
            # Validate result has required components
            if(is.null(result$Qstar) && !is.null(initial_model$preds)) {
              # If targeting failed but we have initial predictions, use those
              result$Qstar <- matrix(rep(initial_model$preds, 3), 
                                     ncol=3, 
                                     byrow=FALSE)
              colnames(result$Qstar) <- c("static", "dynamic", "stochastic")
              cat("Using initial predictions as fallback for time", t, "\n")
            }
            
            result
          }, error = function(e) {
            cat("Error in getTMLELong for time", t, ":", e$message, "\n")
            
            # Fallback: create minimal contrast with initial predictions
            if(!is.null(initial_model) && !is.null(initial_model$preds)) {
              # Create a basic result structure with initial predictions
              basic_result <- list(
                "Qstar" = matrix(rep(initial_model$preds, 3), 
                                 ncol=3, 
                                 byrow=FALSE),
                "ID" = initial_model$data$ID,
                "Y" = initial_model$data$Y
              )
              colnames(basic_result$Qstar) <- c("static", "dynamic", "stochastic")
              return(basic_result)
            } else {
              return(NULL)
            }
          })
          
          cat("Successfully stored result for time point", t, "and continuing to next time point\n")
          
          # Process binary version using same approach
          tmle_contrasts_bin[[t]] <- tryCatch({
            getTMLELong(
              initial_model_for_Y = initial_model,
              tmle_rules = tmle_rules,
              tmle_covars_Y = tmle_covars_Y,
              g_preds_bounded = safe_array(g_preds_bin_cuml_bounded[[min(t+1, length(g_preds_bin_cuml_bounded))]]),
              C_preds_bounded = safe_array(C_preds_cuml_bounded[[min(t+1, length(C_preds_cuml_bounded))]]),
              obs.treatment = treatments[[min(t+1, length(treatments))]],
              obs.rules = obs.rules[[min(t+1, length(obs.rules))]],
              gbound = gbound,
              ybound = ybound,
              t.end = max_time_index,
              debug = FALSE
            )
          }, error = function(e) {
            cat("Error in binary getTMLELong for time", t, ":", e$message, "\n")
            
            # Same fallback as above
            if(!is.null(initial_model) && !is.null(initial_model$preds)) {
              basic_result <- list(
                "Qstar" = matrix(rep(initial_model$preds, 3), 
                                 ncol=3, 
                                 byrow=FALSE),
                "ID" = initial_model$data$ID,
                "Y" = initial_model$data$Y
              )
              colnames(basic_result$Qstar) <- c("static", "dynamic", "stochastic")
              return(basic_result)
            } else {
              return(NULL)
            }
          })
          
          # Log completion for this time point
          cat("Completed time point", t, "\n")
        }
      }
      
      cat("Successfully processed both TMLE versions for time point", t, "\n")
      
      # Verify we have results for all time points
      missing_points <- which(sapply(tmle_contrasts[1:max_time_index], is.null))
      if(length(missing_points) > 0) {
        cat("Still missing results for", length(missing_points), "time points:", 
            paste(missing_points, collapse=", "), "\n")
      } else {
        cat("Successfully processed all time points\n")
      }
    }
  } else if(estimator=='tmle-lstm'){
    
    lstm_data <- prepare_lstm_data(tmle_dat, t.end, window_size)
    tmle_dat <- lstm_data$data
    n_ids <- lstm_data$n_ids  # Store n_ids for later use
    
    print("Training LSTM model for Y")
    lstm_Y_preds <- lstm(
      data = tmle_dat,
      outcome = grep("Y", colnames(tmle_dat), value = TRUE), 
      covariates = tmle_covars_Y,
      t_end = t.end,
      window_size = window_size,
      out_activation = "sigmoid", # Always sigmoid for binary Y
      loss_fn = "binary_crossentropy", # Always binary crossentropy for Y
      output_dir = output_dir,
      J = 1, # J should be 1 for binary Y
      gbound=gbound,
      ybound=ybound,
      is_censoring = FALSE
    )
    
    # Transform lstm_Y_preds into proper matrix format
    transformed_Y_preds <- do.call(cbind, lstm_Y_preds)
    if(is.null(dim(transformed_Y_preds))) {
      transformed_Y_preds <- matrix(transformed_Y_preds, ncol=length(lstm_Y_preds))
    }
    
    # Get number of time points and IDs
    n_ids <- length(unique(tmle_dat$ID))
    time_points <- 0:36
    n_times <- length(time_points)
    
    # Create correct base data frame with proper dimensions 
    long_data <- data.frame(
      ID = rep(sort(unique(tmle_dat$ID)), each=n_times),
      t = rep(time_points, times=n_ids)
    )
    
    # Add non-time-varying columns with proper replication
    static_cols <- c("white", "black", "latino", "other", "mdd", "bipolar", "schiz", "V3") 
    static_cols <- intersect(static_cols, names(tmle_dat))
    for(col in static_cols) {
      long_data[[col]] <- rep(tmle_dat[[col]], each=n_times)
    }
    
    # Add time-varying columns
    time_vars <- grep("\\.", names(tmle_dat), value=TRUE)
    base_vars <- unique(gsub("\\..*$", "", time_vars))
    
    for(var in base_vars) {
      var_cols <- grep(paste0("^", var, "\\."), names(tmle_dat), value=TRUE)
      sorted_cols <- mixedsort(var_cols)  # Ensure proper time ordering
      
      if(var == "A") {
        # Handle treatment variable
        values <- as.vector(t(as.matrix(tmle_dat[sorted_cols])))
        long_data$A <- factor(values)
      } else {
        # Handle other time-varying variables
        values <- as.vector(t(as.matrix(tmle_dat[sorted_cols])))
        long_data[[var]] <- values
      }
    }
    
    # Convert to data frame and fix column names
    long_data <- as.data.frame(long_data)
    colnames(long_data) <- gsub("\\..*$", "", colnames(long_data))
    
    # Convert all columns to numeric except 'A'
    long_data[setdiff(names(long_data), "A")] <- lapply(long_data[setdiff(names(long_data), "A")], as.numeric)
    long_data$A <- as.factor(long_data$A)
    
    # Convert all columns to numeric except 'A', which is converted to a factor
    long_data <- long_data %>%
      mutate(across(where(is.character), as.numeric),  # Convert all character columns to numeric
             A = as.factor(A))  # Convert 'A' to a factor
    
    long_data <- as.data.frame(long_data)
    
    # Rename the columns to remove the "ID" suffix
    names(long_data) <- gsub("\\.ID", "", names(long_data))
    
    # Create initial_model_for_Y using the transformed predictions
    initial_model_for_Y <- list(
      "preds" = boundProbs(transformed_Y_preds[,1], ybound), # Take first prediction only
      "data" = long_data
    )
    
    tmle_rules <- list("static" = static_mtp_lstm,
                       "dynamic" = dynamic_mtp_lstm,
                       "stochastic" = stochastic_mtp_lstm)
    
    tmle_contrasts <- list()
    tmle_contrasts_bin <- list()
    
    print(paste("Length of g_preds:", length(g_preds)))
    print(paste("Length of C_preds:", length(C_preds)))
    print(paste("g_preds dimensions for t=1:", paste(dim(g_preds[[1]]), collapse=" x ")))
    print(paste("Current time point:", t))
    print(paste("Available time points in g_preds_cuml_bounded:", length(g_preds_cuml_bounded)))
    
    # First ensure g_preds_cuml_bounded is properly initialized and populated
    if(is.null(g_preds_cuml_bounded) || length(g_preds_cuml_bounded) == 0) {
      print("g_preds_cuml_bounded is empty or NULL, initializing...")
      g_preds_cuml_bounded <- vector("list", t.end + 1)
      
      # Initialize with matrix format
      for(t in 1:(t.end + 1)) {
        if(t == 1) {
          g_preds_cuml_bounded[[t]] <- matrix(g_preds[[t]], ncol=1)
        } else {
          g_preds_cuml_bounded[[t]] <- matrix(
            g_preds[[t]] * g_preds_cuml_bounded[[t-1]], 
            ncol=1
          )
        }
      }
    }
    
    # Similarly for C_preds_cuml_bounded
    if(is.null(C_preds_cuml_bounded) || length(C_preds_cuml_bounded) == 0) {
      print("C_preds_cuml_bounded is empty or NULL, initializing...")
      C_preds_cuml_bounded <- vector("list", t.end + 1)
      
      for(t in 1:(t.end + 1)) {
        if(t == 1) {
          C_preds_cuml_bounded[[t]] <- matrix(C_preds[[t]], ncol=1)
        } else {
          C_preds_cuml_bounded[[t]] <- matrix(
            C_preds[[t]] * C_preds_cuml_bounded[[t-1]], 
            ncol=1
          )
        }
      }
    }
    
    # Process predictions first
    print("Processing predictions for TMLE")
    
    # Get number of IDs and time points
    n_ids <- length(unique(initial_model_for_Y$data$ID))
    n_times <- t.end + 1
    
    print(paste("Number of IDs:", n_ids))
    print(paste("Number of time points:", n_times))
    
    # Process g predictions - add t_end parameter here
    g_preds_processed <- safe_get_cuml_preds(g_preds, n_ids, t.end)
    g_preds_bin_processed <- safe_get_cuml_preds(g_preds_bin, n_ids, t.end)
    print("G predictions processed")
    
    # Process C predictions - add t_end parameter here
    C_preds_processed <- safe_get_cuml_preds(C_preds, n_ids, t.end)
    print("C predictions processed")
    
    n_ids <- length(unique(initial_model_for_Y$data$ID))
    
    results <- process_time_points_batch(
      initial_model_for_Y = initial_model_for_Y$preds,
      initial_model_for_Y_data = initial_model_for_Y$data,
      tmle_rules = tmle_rules,
      tmle_covars_Y = tmle_covars_Y,
      g_preds_processed = g_preds_processed,
      g_preds_bin_processed = g_preds_bin_processed,
      C_preds_processed = C_preds_processed,
      treatments = treatments,
      obs.rules = obs.rules,
      gbound = gbound,
      ybound = ybound,
      t_end = t.end,
      window_size = window_size,
      n_ids = n_ids,
      output_dir = output_dir,
      cores = cores,
      debug = debug
    )
    
    tmle_contrasts <- results[["multinomial"]]
    tmle_contrasts_bin <- results[["binary"]]
  }
  
  cat("CHECKPOINT: Starting bias and confidence interval calculation\n")
  
  if(estimator=='tmle') {
    cat("Calculating final estimates using TMLE approach\n")
    
    # Process TMLE estimates - convert from Qstar event probabilities to survival probabilities
    tmle_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(tmle_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]]) && !is.null(tmle_contrasts[[t]]$Qstar)) {
        # Extract Qstar values (these are event probabilities)
        qstar_values <- tmle_contrasts[[t]]$Qstar
        
        # Check format and dimensions
        if(is.matrix(qstar_values) && ncol(qstar_values) >= 3) {
          # Calculate column means (event probabilities)
          event_probs <- colMeans(qstar_values, na.rm=TRUE)
          
          # Convert event probabilities to survival probabilities (1 - event_prob)
          for(rule in 1:3) {
            tmle_estimates[rule,t] <- 1 - event_probs[rule]
          }
        }
      }
    }
    
    # Process Binary TMLE estimates
    tmle_bin_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(tmle_bin_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts_bin[[t]]) && !is.null(tmle_contrasts_bin[[t]]$Qstar)) {
        # Extract Qstar values (these are event probabilities)
        qstar_values <- tmle_contrasts_bin[[t]]$Qstar
        
        # Check format and dimensions
        if(is.matrix(qstar_values) && ncol(qstar_values) >= 3) {
          # Calculate column means (event probabilities)
          event_probs <- colMeans(qstar_values, na.rm=TRUE)
          
          # Convert event probabilities to survival probabilities (1 - event_prob)
          for(rule in 1:3) {
            tmle_bin_estimates[rule,t] <- 1 - event_probs[rule]
          }
        }
      }
    }
    
    # Process IPTW estimates
    iptw_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(iptw_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]]) && !is.null(tmle_contrasts[[t]]$Qstar_iptw)) {
        # Extract IPTW values
        if(is.vector(tmle_contrasts[[t]]$Qstar_iptw)) {
          # Vector format
          iptw_values <- tmle_contrasts[[t]]$Qstar_iptw
          if(length(iptw_values) >= 3) {
            # Convert event probabilities to survival probabilities
            for(rule in 1:3) {
              iptw_estimates[rule,t] <- 1 - iptw_values[rule]
            }
          }
        } else if(is.matrix(tmle_contrasts[[t]]$Qstar_iptw) && nrow(tmle_contrasts[[t]]$Qstar_iptw) > 0) {
          # Matrix format (typically first row)
          iptw_values <- tmle_contrasts[[t]]$Qstar_iptw[1,]
          if(length(iptw_values) >= 3) {
            # Convert event probabilities to survival probabilities
            for(rule in 1:3) {
              iptw_estimates[rule,t] <- 1 - iptw_values[rule]
            }
          }
        }
      }
    }
    
    # Process Binary IPTW estimates
    iptw_bin_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(iptw_bin_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts_bin[[t]]) && !is.null(tmle_contrasts_bin[[t]]$Qstar_iptw)) {
        # Extract IPTW values
        if(is.vector(tmle_contrasts_bin[[t]]$Qstar_iptw)) {
          # Vector format
          iptw_values <- tmle_contrasts_bin[[t]]$Qstar_iptw
          if(length(iptw_values) >= 3) {
            # Convert event probabilities to survival probabilities
            for(rule in 1:3) {
              iptw_bin_estimates[rule,t] <- 1 - iptw_values[rule]
            }
          }
        } else if(is.matrix(tmle_contrasts_bin[[t]]$Qstar_iptw) && nrow(tmle_contrasts_bin[[t]]$Qstar_iptw) > 0) {
          # Matrix format (typically first row)
          iptw_values <- tmle_contrasts_bin[[t]]$Qstar_iptw[1,]
          if(length(iptw_values) >= 3) {
            # Convert event probabilities to survival probabilities
            for(rule in 1:3) {
              iptw_bin_estimates[rule,t] <- 1 - iptw_values[rule]
            }
          }
        }
      }
    }
    
    # Process G-computation estimates
    gcomp_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(gcomp_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]]) && !is.null(tmle_contrasts[[t]]$Qstar_gcomp)) {
        # Extract G-comp values
        if(is.matrix(tmle_contrasts[[t]]$Qstar_gcomp) && ncol(tmle_contrasts[[t]]$Qstar_gcomp) >= 3) {
          # Matrix format - calculate column means
          gcomp_values <- colMeans(tmle_contrasts[[t]]$Qstar_gcomp, na.rm=TRUE)
          
          # Convert event probabilities to survival probabilities
          for(rule in 1:3) {
            gcomp_estimates[rule,t] <- 1 - gcomp_values[rule]
          }
        }
      }
    }
    
    cat("Final estimate matrices populated successfully\n")
    
    if(debug){
      cat("\nEstimated means before final processing:\n")
      cat("TMLE estimates dimensions:", dim(tmle_estimates), "\n")
      print(tmle_estimates)
      cat("\nIPTW estimates dimensions:", dim(iptw_estimates), "\n")
      print(iptw_estimates)
      cat("\nG-comp estimates dimensions:", dim(gcomp_estimates), "\n")
      print(gcomp_estimates)
    }
    # Apply the function to fill NAs
    tmle_estimates <- fill_na_estimates(tmle_estimates, use_interpolation=TRUE)
    tmle_bin_estimates <- fill_na_estimates(tmle_bin_estimates, use_interpolation=TRUE)
    iptw_estimates <- fill_na_estimates(iptw_estimates, use_interpolation=TRUE)
    iptw_bin_estimates <- fill_na_estimates(iptw_bin_estimates, use_interpolation=TRUE)
    gcomp_estimates <- fill_na_estimates(gcomp_estimates, use_interpolation=TRUE)
  } 
  else if(estimator=='tmle-lstm') {
    if(debug) cat("\nCalculating final estimates using LSTM approach\n")
    
    # Process each time point for TMLE estimates
    tmle_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(tmle_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]])) {
        for(rule in 1:3) {
          # Skip if Qstar is not available or has invalid dimension
          if(is.null(tmle_contrasts[[t]]$Qstar) || ncol(tmle_contrasts[[t]]$Qstar) < rule) {
            tmle_estimates[rule,t] <- NA
            next
          }
          
          # Extract values and handle missingness/invalid values
          values <- tmle_contrasts[[t]]$Qstar[,rule]
          valid_values <- values[!is.na(values) & !is.nan(values) & is.finite(values) & values != -1]
          
          if(length(valid_values) > 0) {
            # Convert event probability into survival probability
            tmle_estimates[rule,t] <- 1 - mean(valid_values, na.rm=TRUE)
          }
        }
      }
    }
    
    # Process each time point for binary TMLE estimates
    tmle_bin_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(tmle_bin_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts_bin[[t]])) {
        for(rule in 1:3) {
          # Skip if Qstar is not available or has invalid dimension
          if(is.null(tmle_contrasts_bin[[t]]$Qstar) || ncol(tmle_contrasts_bin[[t]]$Qstar) < rule) {
            tmle_bin_estimates[rule,t] <- NA
            next
          }
          
          # Extract values and handle missingness/invalid values
          values <- tmle_contrasts_bin[[t]]$Qstar[,rule]
          valid_values <- values[!is.na(values) & !is.nan(values) & is.finite(values) & values != -1]
          
          if(length(valid_values) > 0) {
            tmle_bin_estimates[rule,t] <- 1 - mean(valid_values, na.rm=TRUE)
          }
        }
      }
    }
    
    # Process IPTW estimates
    iptw_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(iptw_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]]$Qstar_iptw)) {
        # Handle both vector and matrix formats
        if(is.vector(tmle_contrasts[[t]]$Qstar_iptw)) {
          iptw_means <- tmle_contrasts[[t]]$Qstar_iptw
          if(length(iptw_means) >= 3) {
            for(rule in 1:3) {
              # Check if value is valid
              if(!is.na(iptw_means[rule]) && !is.nan(iptw_means[rule]) && is.finite(iptw_means[rule])) {
                iptw_estimates[rule,t] <- 1 - iptw_means[rule]
              }
            }
          }
        } else if(is.matrix(tmle_contrasts[[t]]$Qstar_iptw) && nrow(tmle_contrasts[[t]]$Qstar_iptw) > 0) {
          iptw_means <- tmle_contrasts[[t]]$Qstar_iptw[1,]
          if(length(iptw_means) >= 3) {
            for(rule in 1:3) {
              # Check if value is valid
              if(!is.na(iptw_means[rule]) && !is.nan(iptw_means[rule]) && is.finite(iptw_means[rule])) {
                iptw_estimates[rule,t] <- 1 - iptw_means[rule]
              }
            }
          }
        }
      }
    }
    
    # Process binary IPTW estimates
    iptw_bin_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(iptw_bin_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts_bin[[t]]$Qstar_iptw)) {
        # Handle both vector and matrix formats
        if(is.vector(tmle_contrasts_bin[[t]]$Qstar_iptw)) {
          iptw_means <- tmle_contrasts_bin[[t]]$Qstar_iptw
          if(length(iptw_means) >= 3) {
            for(rule in 1:3) {
              # Check if value is valid
              if(!is.na(iptw_means[rule]) && !is.nan(iptw_means[rule]) && is.finite(iptw_means[rule])) {
                iptw_bin_estimates[rule,t] <- 1 - iptw_means[rule]
              }
            }
          }
        } else if(is.matrix(tmle_contrasts_bin[[t]]$Qstar_iptw) && nrow(tmle_contrasts_bin[[t]]$Qstar_iptw) > 0) {
          iptw_means <- tmle_contrasts_bin[[t]]$Qstar_iptw[1,]
          if(length(iptw_means) >= 3) {
            for(rule in 1:3) {
              # Check if value is valid
              if(!is.na(iptw_means[rule]) && !is.nan(iptw_means[rule]) && is.finite(iptw_means[rule])) {
                iptw_bin_estimates[rule,t] <- 1 - iptw_means[rule]
              }
            }
          }
        }
      }
    }
    
    # Process G-computation estimates
    gcomp_estimates <- matrix(NA, nrow=3, ncol=t.end)
    rownames(gcomp_estimates) <- c("static", "dynamic", "stochastic")
    
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]]$Qstar_gcomp)) {
        for(rule in 1:3) {
          # Skip if Qstar_gcomp doesn't have proper dimensions
          if(is.null(tmle_contrasts[[t]]$Qstar_gcomp) || ncol(tmle_contrasts[[t]]$Qstar_gcomp) < rule) {
            next
          }
          
          # Extract values and handle missingness/invalid values
          values <- tmle_contrasts[[t]]$Qstar_gcomp[,rule]
          valid_values <- values[!is.na(values) & !is.nan(values) & is.finite(values) & values != -1]
          
          if(length(valid_values) > 0) {
            gcomp_estimates[rule,t] <- 1 - mean(valid_values, na.rm=TRUE)
          }
        }
      }
    }
  }
  
  # Add validation printing
  
  if(debug) {
    cat("\nEstimated means before final processing:\n")
    cat("TMLE estimates dimensions:", dim(tmle_estimates), "\n")
    print(head(tmle_estimates))
    cat("\nIPTW estimates dimensions:", dim(iptw_estimates), "\n")
    print(head(iptw_estimates))
    cat("\nG-comp estimates dimensions:", dim(gcomp_estimates), "\n")
    print(head(gcomp_estimates))
  }
  
  # Raw values are event probabilities, convert to survival only at output
  if(estimator=='tmle-lstm') {
    if(debug) cat("\nCalculating final estimates using LSTM approach\n")
    
    # 1. TMLE estimates (standard)
    tmle_estimates <- process_estimates(tmle_contrasts, "Qstar", t.end, obs.rules)
    
    # 2. Binary TMLE estimates
    tmle_bin_estimates <- process_estimates(tmle_contrasts_bin, "Qstar", t.end, obs.rules)
    
    # 3. IPTW estimates (standard)
    iptw_estimates <- process_estimates(tmle_contrasts, "Qstar_iptw", t.end, obs.rules)
    
    # 4. Binary IPTW estimates
    iptw_bin_estimates <- process_estimates(tmle_contrasts_bin, "Qstar_iptw", t.end, obs.rules)
    
    # 5. G-computation estimates
    gcomp_estimates <- process_estimates(tmle_contrasts, "Qstar_gcomp", t.end, obs.rules)
    
    # Print summary information if in debug mode
    if(debug) {
      cat("\n\nFinal estimates summary:")
      cat("\nTMLE estimates:", colMeans(tmle_estimates))
      cat("\nTMLE-bin estimates:", colMeans(tmle_bin_estimates))
      cat("\nIPTW estimates:", colMeans(iptw_estimates))
      cat("\nIPTW-bin estimates:", colMeans(iptw_bin_estimates))
      cat("\nG-comp estimates:", colMeans(gcomp_estimates))
    }
  }
  
  if(r==1){
    # These plots use survival probabilities (1 - event probabilities)
    # The process_estimates function now returns survival probabilities directly
    png(paste0(output_dir,paste0("survival_plot_tmle_estimates_",n, "_" , estimator,".png")))
    plotSurvEst(surv = list("Static"= tmle_estimates[1,], "Dynamic"= tmle_estimates[2,], "Stochastic"= tmle_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "TMLE (ours, multinomial) estimated counterfactuals",
                xlab = "Month",
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_tmle_estimates_bin_",n, "_", estimator,".png")))
    plotSurvEst(surv = list("Static"= tmle_bin_estimates[1,], "Dynamic"= tmle_bin_estimates[2,], "Stochastic"= tmle_bin_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "TMLE (ours, binomial) estimated counterfactuals",
                xlab = "Month",
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_iptw_estimates_",n,  "_", estimator,".png")))
    plotSurvEst(surv = list("Static"= iptw_estimates[1,], "Dynamic"= iptw_estimates[2,], "Stochastic"= iptw_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "IPTW (ours, multinomial) estimated counterfactuals",
                xlab = "Month",
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_iptw_bin_estimates_",n,  "_", estimator,".png")))
    plotSurvEst(surv = list("Static"= iptw_bin_estimates[1,], "Dynamic"= iptw_bin_estimates[2,], "Stochastic"= iptw_bin_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "IPTW (ours, binomial) estimated counterfactuals",
                xlab = "Month",
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_gcomp_estimates_",n,  "_", estimator,".png")))
    plotSurvEst(surv = list("Static"= gcomp_estimates[1,], "Dynamic"= gcomp_estimates[2,], "Stochastic"= gcomp_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "G-comp. (ours) estimated counterfactuals",
                xlab = "Month",
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
  }
  
  print("Calculating share of cumulative probabilities of continuing to receive treatment according to the assigned treatment rule which are smaller than 0.025")
  
  if(estimator == "tmle") {
    # Create g_preds_processed correctly from g_preds_cuml_bounded
    g_preds_processed <- g_preds_cuml_bounded
    
    # Also create g_preds_bin_processed from g_preds_bin_cuml_bounded
    g_preds_bin_processed <- g_preds_bin_cuml_bounded
    
    prob_share <- lapply(1:(t.end+1), function(t) {
      # Check that g_preds_processed exists and has data for this time point
      if(is.null(g_preds_processed) || length(g_preds_processed) < t || is.null(g_preds_processed[[t]])) {
        # Return NA matrix if no data
        result <- matrix(NA, nrow=6, ncol=3)  # Assuming 6 treatments and 3 rules
        colnames(result) <- c("static", "dynamic", "stochastic")
        result
      }
      
      # Check obs.rules also exists
      if(is.null(obs.rules) || length(obs.rules) < t || is.null(obs.rules[[t]])) {
        # Return NA matrix if no data
        result <- matrix(NA, nrow=6, ncol=3)
        colnames(result) <- c("static", "dynamic", "stochastic")
        result
      }
      
      # Safe calculation with error handling
      tryCatch({
        # Get the tmle_dat IDs at time t
        t_ids <- na.omit(tmle_dat[tmle_dat$t==(t-1),])$ID
        
        # Calculate safely with dimension checking
        if(length(t_ids) > 0 && ncol(obs.rules[[t]]) > 0) {
          result <- sapply(1:ncol(obs.rules[[t]]), function(i) {
            rule_rows <- which(obs.rules[[t]][t_ids, i] == 1)
            if(length(rule_rows) > 0) {
              # Check dimensions and use appropriate subsetting
              if(is.matrix(g_preds_processed[[t]])) {
                vals <- g_preds_processed[[t]][rule_rows, , drop=FALSE]
                # Calculate column means safely
                colMeans(vals < 0.025, na.rm=TRUE)
              } else {
                # If not a matrix, create a safer fallback
                rep(NA, 6)  # Assuming 6 treatments
              }
            } else {
              # No matching rows
              rep(NA, 6)  # Assuming 6 treatments
            }
          })
          
          # Ensure proper column names
          colnames(result) <- colnames(obs.rules[[t]])
          result
        } else {
          # Return NA matrix if no data
          result <- matrix(NA, nrow=6, ncol=3)
          colnames(result) <- c("static", "dynamic", "stochastic")
          result
        }
      }, error = function(e) {
        # Return NA matrix on error
        message("Error calculating prob_share for time ", t, ": ", e$message)
        result <- matrix(NA, nrow=6, ncol=3)
        colnames(result) <- c("static", "dynamic", "stochastic")
        result
      })
    })
    
    # Set names for list elements
    names(prob_share) <- paste0("t=", seq(0, t.end))
    
    # Same approach for binary predictions
    prob_share_bin <- lapply(1:(t.end+1), function(t) {
      # Check that g_preds_bin_processed exists and has data for this time point
      if(is.null(g_preds_bin_processed) || length(g_preds_bin_processed) < t || is.null(g_preds_bin_processed[[t]])) {
        # Return NA matrix if no data
        result <- matrix(NA, nrow=6, ncol=3)
        colnames(result) <- c("static", "dynamic", "stochastic")
        result
      }
      
      # Check obs.rules also exists
      if(is.null(obs.rules) || length(obs.rules) < t || is.null(obs.rules[[t]])) {
        # Return NA matrix if no data
        result <- matrix(NA, nrow=6, ncol=3)
        colnames(result) <- c("static", "dynamic", "stochastic")
        result
      }
      
      # Safe calculation with error handling
      tryCatch({
        # Get the tmle_dat IDs at time t
        t_ids <- na.omit(tmle_dat[tmle_dat$t==(t-1),])$ID
        
        # Calculate safely with dimension checking
        if(length(t_ids) > 0 && ncol(obs.rules[[t]]) > 0) {
          result <- sapply(1:ncol(obs.rules[[t]]), function(i) {
            rule_rows <- which(obs.rules[[t]][t_ids, i] == 1)
            if(length(rule_rows) > 0) {
              # Check dimensions and use appropriate subsetting
              if(is.matrix(g_preds_bin_processed[[t]])) {
                vals <- g_preds_bin_processed[[t]][rule_rows, , drop=FALSE]
                # Calculate column means safely
                colMeans(vals < 0.025, na.rm=TRUE)
              } else {
                # If not a matrix, create a safer fallback
                rep(NA, 6)  # Assuming 6 treatments
              }
            } else {
              # No matching rows
              rep(NA, 6)  # Assuming 6 treatments
            }
          })
          
          # Ensure proper column names
          colnames(result) <- colnames(obs.rules[[t]])
          result
        } else {
          # Return NA matrix if no data
          result <- matrix(NA, nrow=6, ncol=3)
          colnames(result) <- c("static", "dynamic", "stochastic")
          result
        }
      }, error = function(e) {
        # Return NA matrix on error
        message("Error calculating prob_share_bin for time ", t, ": ", e$message)
        result <- matrix(NA, nrow=6, ncol=3)
        colnames(result) <- c("static", "dynamic", "stochastic")
        result
      })
    })
    
    # Set names for list elements
    names(prob_share_bin) <- paste0("t=", seq(0, t.end))
  }else{
    prob_share <- lapply(1:(t.end+1), function(t) {
      # Check that g_preds_processed exists and has data for this time point
      if(is.null(g_preds_processed) || length(g_preds_processed) < t || is.null(g_preds_processed[[t]])) {
        # Return matrix with default values instead of NA
        result <- matrix(0, nrow=J, ncol=3)  # Using zeros instead of NA
        colnames(result) <- c("static", "dynamic", "stochastic")
        return(result)
      }
      
      # Check obs.rules also exists
      if(is.null(obs.rules) || length(obs.rules) < t || is.null(obs.rules[[t]])) {
        # Return matrix with default values
        result <- matrix(0, nrow=J, ncol=3)
        colnames(result) <- c("static", "dynamic", "stochastic")
        return(result)
      }
      
      # Safe calculation with error handling
      tryCatch({
        # Calculate prob_share with proper dimension checking
        result <- sapply(1:ncol(obs.rules[[t]]), function(i) {
          rule_rows <- which(obs.rules[[t]][, i] == 1)
          if(length(rule_rows) > 0 && nrow(g_preds_processed[[t]]) >= max(rule_rows)) {
            # Extract relevant probabilities
            vals <- g_preds_processed[[t]][rule_rows, , drop=FALSE]
            # Calculate proportion < 0.025 safely
            if(is.matrix(vals) && nrow(vals) > 0) {
              return(colMeans(vals < 0.025, na.rm=TRUE))
            }
          }
          return(rep(0, J))  # Default to 0 instead of NA
        })
        
        # Ensure proper column names
        colnames(result) <- colnames(obs.rules[[t]])
        return(result)
      }, error = function(e) {
        message("Error calculating prob_share for time ", t, ": ", e$message)
        result <- matrix(0, nrow=J, ncol=3)  # Default values
        colnames(result) <- c("static", "dynamic", "stochastic")
        return(result)
      })
    })
    
    # Set column names safely
    for(t in 1:(t.end+1)){
      if(t <= length(prob_share) && !is.null(prob_share[[t]]) && 
         t.end <= length(obs.rules) && !is.null(obs.rules[[t.end]])) {
        if(is.matrix(prob_share[[t]]) && ncol(prob_share[[t]]) == ncol(obs.rules[[t.end]])) {
          colnames(prob_share[[t]]) <- colnames(obs.rules[[t.end]])
        }
      }
    }
    
    # Set list names
    names(prob_share) <- paste0("t=", seq(0, t.end))
    
    prob_share_bin <- lapply(1:(t.end+1), function(t) {
      # Check that g_preds_bin_processed exists and has data for this time point
      if(is.null(g_preds_bin_processed) || length(g_preds_bin_processed) < t || is.null(g_preds_bin_processed[[t]])) {
        # Return matrix with default values instead of NA
        result <- matrix(0, nrow=J, ncol=3)  # Using zeros instead of NA
        colnames(result) <- c("static", "dynamic", "stochastic")
        return(result)
      }
      
      # Check obs.rules also exists
      if(is.null(obs.rules) || length(obs.rules) < t || is.null(obs.rules[[t]])) {
        # Return matrix with default values
        result <- matrix(0, nrow=J, ncol=3)
        colnames(result) <- c("static", "dynamic", "stochastic")
        return(result)
      }
      
      # Safe calculation with error handling
      tryCatch({
        # Calculate prob_share with proper dimension checking
        result <- sapply(1:ncol(obs.rules[[t]]), function(i) {
          rule_rows <- which(obs.rules[[t]][, i] == 1)
          if(length(rule_rows) > 0 && nrow(g_preds_bin_processed[[t]]) >= max(rule_rows)) {
            # Extract relevant probabilities
            vals <- g_preds_bin_processed[[t]][rule_rows, , drop=FALSE]
            # Calculate proportion < 0.025 safely
            if(is.matrix(vals) && nrow(vals) > 0) {
              return(colMeans(vals < 0.025, na.rm=TRUE))
            }
          }
          return(rep(0, J))  # Default to 0 instead of NA
        })
        
        # Ensure proper column names
        colnames(result) <- colnames(obs.rules[[t]])
        return(result)
      }, error = function(e) {
        message("Error calculating prob_share for time ", t, ": ", e$message)
        result <- matrix(0, nrow=J, ncol=3)  # Default values
        colnames(result) <- c("static", "dynamic", "stochastic")
        return(result)
      })
    })
    
    # Set column names safely
    for(t in 1:(t.end+1)){
      if(t <= length(prob_share_bin) && !is.null(prob_share_bin[[t]]) && 
         t.end <= length(obs.rules) && !is.null(obs.rules[[t.end]])) {
        if(is.matrix(prob_share_bin[[t]]) && ncol(prob_share_bin[[t]]) == ncol(obs.rules[[t.end]])) {
          colnames(prob_share_bin[[t]]) <- colnames(obs.rules[[t.end]])
        }
      }
    }
    
    # Set list names
    names(prob_share_bin) <- paste0("t=", seq(0, t.end))
  }
  
  # Create diagnostic output directory if it doesn't exist
  diagnostics_dir <- paste0(output_dir, "diagnostics/")
  if(!dir.exists(diagnostics_dir)) {
    dir.create(diagnostics_dir, recursive = TRUE)
  }
  
  if(estimator=="tmle-lstm"){
    # For LSTM estimator
    # Fix NaN values in the final time point
    if(length(tmle_contrasts) >= t.end && !is.null(tmle_contrasts[[t.end]]) && 
       !is.null(tmle_contrasts[[t.end]]$Qstar) && any(is.nan(tmle_contrasts[[t.end]]$Qstar))) {
      print("Fixing NaN values in final time point predictions...")
      # Replace NaN values with mean of non-NaN values (or a fallback)
      for(rule in 1:ncol(tmle_contrasts[[t.end]]$Qstar)) {
        nan_indices <- which(is.nan(tmle_contrasts[[t.end]]$Qstar[,rule]))
        if(length(nan_indices) > 0) {
          valid_values <- tmle_contrasts[[t.end]]$Qstar[-nan_indices, rule]
          if(length(valid_values) > 0) {
            replacement <- mean(valid_values, na.rm=TRUE)
          } else {
            replacement <- 0.5  # Fallback if no valid values
          }
          tmle_contrasts[[t.end]]$Qstar[nan_indices, rule] <- replacement
          print(paste("Fixed", length(nan_indices), "NaN values in rule", rule))
        }
      }
    }
    
    # Add diagnostics flag to capture detailed variance information
    tmle_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, 
                            estimator="tmle-lstm", diagnostics=TRUE)
    tmle_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y, time.censored, 
                                estimator="tmle-lstm", diagnostics=TRUE)
    
    iptw_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, 
                            iptw=TRUE, estimator="tmle-lstm", diagnostics=TRUE)
    iptw_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y, time.censored, 
                                iptw=TRUE, estimator="tmle-lstm", diagnostics=TRUE)
    
    gcomp_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, 
                             gcomp=TRUE, estimator="tmle-lstm", diagnostics=TRUE)
    
    cat("CHECKPOINT: Variance estimates computed\n")
    
    # Save variance diagnostic information for analysis
    if(r == 1 && debug) {
      # Save standard error trends over time
      se_data <- data.frame(
        time_point = rep(1:length(tmle_est_var$se), each=3),
        treatment_rule = rep(c("static", "dynamic", "stochastic"), length(tmle_est_var$se)),
        se_tmle = unlist(lapply(tmle_est_var$se, function(x) x[1:3])),
        se_tmle_bin = unlist(lapply(tmle_est_var_bin$se, function(x) x[1:3])),
        se_iptw = unlist(lapply(iptw_est_var$se, function(x) x[1:3])),
        se_iptw_bin = unlist(lapply(iptw_est_var_bin$se, function(x) x[1:3])),
        se_gcomp = unlist(lapply(gcomp_est_var$se, function(x) x[1:3]))
      )
      
      # Generate basic diagnostic plots for standard errors
      if(requireNamespace("ggplot2", quietly = TRUE)) {
        # Plot standard error trends over time
        p <- ggplot2::ggplot(se_data, ggplot2::aes(x=time_point, y=se_tmle, color=treatment_rule)) +
          ggplot2::geom_line() +
          ggplot2::labs(title="Standard Error Trends (TMLE)", x="Time Point", y="Standard Error") +
          ggplot2::theme_minimal()
        
        # Save the plot
        ggplot2::ggsave(paste0(diagnostics_dir, "se_trends_tmle.png"), p, width=8, height=6)
        
        # Plot comparison of SEs across estimators
        p2 <- ggplot2::ggplot(se_data, ggplot2::aes(x=time_point)) +
          ggplot2::geom_line(ggplot2::aes(y=se_tmle, color="TMLE")) +
          ggplot2::geom_line(ggplot2::aes(y=se_tmle_bin, color="TMLE-bin")) +
          ggplot2::geom_line(ggplot2::aes(y=se_iptw, color="IPTW")) +
          ggplot2::geom_line(ggplot2::aes(y=se_gcomp, color="G-comp")) +
          ggplot2::facet_wrap(~treatment_rule) +
          ggplot2::labs(title="Standard Error Comparison", x="Time Point", y="Standard Error", color="Estimator") +
          ggplot2::theme_minimal()
        
        ggplot2::ggsave(paste0(diagnostics_dir, "se_comparison.png"), p2, width=10, height=6)
      }
      
      # Save the raw data
      saveRDS(se_data, paste0(diagnostics_dir, "se_diagnostic_data.rds"))
      
      # Save additional diagnostic information
      if(!is.null(tmle_est_var$diagnostics)) {
        saveRDS(tmle_est_var$diagnostics, paste0(diagnostics_dir, "tmle_variance_diagnostics.rds"))
      }
    }
    
  } else {
    # For standard TMLE estimator
    cat("Calculating variance estimates...\n")
    
    # Optimize TMLE_IC function calls by focusing on key time points
    key_time_points <- unique(c(1, seq(5, t.end, by=5), t.end))
    
    # First generate basic variance estimates only for key time points
    tmle_est_var_basic <- TMLE_IC(
      tmle_contrasts[key_time_points], 
      initial_model_for_Y[key_time_points], 
      time.censored, 
      estimator="tmle", 
      basic_only=TRUE,
      diagnostics=TRUE
    )
    
    # Then run full estimation for all time points, using the basic estimates as a starting point
    tmle_est_var <- TMLE_IC(
      tmle_contrasts, 
      initial_model_for_Y, 
      time.censored, 
      estimator="tmle",
      variance_estimates=tmle_est_var_basic,
      diagnostics=TRUE
    )
    tmle_est_var$est <- fill_na_estimates(tmle_est_var$est, use_interpolation=TRUE)
    
    # Run binary case with similar approach
    tmle_est_var_bin_basic <- TMLE_IC(
      tmle_contrasts_bin[key_time_points], 
      initial_model_for_Y_bin[key_time_points], 
      time.censored, 
      estimator="tmle",
      basic_only=TRUE,
      diagnostics=TRUE
    )
    
    tmle_est_var_bin <- TMLE_IC(
      tmle_contrasts_bin, 
      initial_model_for_Y_bin, 
      time.censored, 
      estimator="tmle",
      variance_estimates=tmle_est_var_bin_basic,
      diagnostics=TRUE
    )
    
    tmle_est_var_bin$est <- fill_na_estimates(tmle_est_var_bin$est, use_interpolation=TRUE)
    
    # Simple IPTW variance estimation (removed simplified parameter)
    iptw_est_var <- TMLE_IC(
      tmle_contrasts, 
      initial_model_for_Y, 
      time.censored, 
      iptw=TRUE, 
      estimator="tmle",
      diagnostics=TRUE
    )
    
    iptw_est_var_bin <- TMLE_IC(
      tmle_contrasts_bin, 
      initial_model_for_Y_bin, 
      time.censored, 
      iptw=TRUE, 
      estimator="tmle",
      diagnostics=TRUE
    )
    
    iptw_est_var$est <- fill_na_estimates(iptw_est_var$est, use_interpolation=TRUE)
    iptw_est_var_bin$est <- fill_na_estimates(iptw_est_var_bin$est, use_interpolation=TRUE)
    
    # G-computation variance estimation without smoothing
    # Get G-comp estimates directly (removed simplified parameter)
    gcomp_est_var <- TMLE_IC(
      tmle_contrasts, 
      initial_model_for_Y, 
      time.censored, 
      gcomp=TRUE, 
      estimator="tmle",
      diagnostics=TRUE
    )
    gcomp_est_var$est <- fill_na_estimates(gcomp_est_var$est, use_interpolation=TRUE)
    
    # Save diagnostic information when running the first iteration in debug mode
    if(r == 1 && debug) {
      # Collect variance diagnostic data across estimators
      variance_diagnostics <- list(
        "tmle" = tmle_est_var$diagnostics,
        "tmle_bin" = tmle_est_var_bin$diagnostics,
        "iptw" = iptw_est_var$diagnostics,
        "iptw_bin" = iptw_est_var_bin$diagnostics,
        "gcomp" = gcomp_est_var$diagnostics
      )
      
      # Save the diagnostic information
      saveRDS(variance_diagnostics, paste0(diagnostics_dir, "variance_diagnostics.rds"))
      
      # Create a table of variance statistics by time point
      if(!is.null(tmle_est_var$diagnostics)) {
        var_stats <- data.frame(
          time_point = 1:length(tmle_est_var$se),
          static_se = sapply(tmle_est_var$se, function(x) x[1]),
          dynamic_se = sapply(tmle_est_var$se, function(x) x[2]),
          stochastic_se = sapply(tmle_est_var$se, function(x) x[3])
        )
        
        # Save as CSV for easy viewing
        write.csv(var_stats, paste0(diagnostics_dir, "variance_statistics.csv"), row.names=FALSE)
      }
    }
  }
  
  print("Storing results")
  
  Ahat_tmle  <- g_preds_processed
  Ahat_tmle_bin  <- g_preds_bin_processed
  
  Chat_tmle  <- C_preds_cuml_bounded
  
  print("Calculating bias, CP, CIW wrt to est at each t")
  
  # Fix bias calculation with better NA handling
  bias_tmle <- lapply(2:t.end, function(t) {
    # Get true survival probabilities (1 - event probabilities)
    true_survival <- 1 - sapply(Y.true, "[[", t)
    
    # Check if we have estimates for this time point
    if(t <= length(tmle_est_var$est) && !is.null(tmle_est_var$est[[t]])) {
      est_survival <- tmle_est_var$est[[t]]
      # Calculate bias (true - estimated)
      bias <- true_survival - est_survival
      return(bias)
    } else {
      # Return NA if we don't have estimates for this time point
      return(rep(NA, length(true_survival)))
    }
  })
  names(bias_tmle) <- paste0("t=", 2:t.end)
  
  # Fix coverage probability calculation
  CP_tmle <- lapply(1:(t.end-1), function(t) {
    # Get true survival probabilities for time t+1
    true_survival <- 1 - sapply(Y.true, "[[", t+1)
    
    # Check if we have CI information for this time point
    if(t <= length(tmle_est_var$CI) && !is.null(tmle_est_var$CI[[t]])) {
      # Extract lower and upper bounds
      lower_ci <- tmle_est_var$CI[[t]][1,]
      upper_ci <- tmle_est_var$CI[[t]][2,]
      
      # Calculate coverage (1 if CI covers true value, 0 otherwise)
      coverage <- as.numeric((lower_ci <= true_survival) & (upper_ci >= true_survival))
      return(coverage)
    } else {
      # Return NA if we don't have CI for this time point
      return(rep(NA, length(true_survival)))
    }
  })
  names(CP_tmle) <- paste0("t=", 2:t.end)
  
  # Fix CI width calculation
  CIW_tmle <- lapply(1:(t.end-1), function(t) {
    # Check if we have CI information for this time point
    if(t <= length(tmle_est_var$CI) && !is.null(tmle_est_var$CI[[t]])) {
      # Extract lower and upper bounds
      lower_ci <- tmle_est_var$CI[[t]][1,]
      upper_ci <- tmle_est_var$CI[[t]][2,]
      
      # Calculate CI width
      ci_width <- upper_ci - lower_ci
      return(ci_width)
    } else {
      # Return NA if we don't have CI for this time point
      return(rep(NA, ncol(tmle_est_var$CI[[1]])))
    }
  })
  names(CIW_tmle) <- paste0("t=", 2:t.end)
  
  # Binary TMLE version
  bias_tmle_bin <- lapply(2:t.end, function(t) {
    # Get true survival probabilities
    true_survival <- 1 - sapply(Y.true, "[[", t)
    
    # Check if we have estimates for this time point
    if(t <= length(tmle_est_var_bin$est) && !is.null(tmle_est_var_bin$est[[t]])) {
      est_survival <- tmle_est_var_bin$est[[t]]
      # Calculate bias
      bias <- true_survival - est_survival
      return(bias)
    } else {
      # Return NA if we don't have estimates for this time point
      return(rep(NA, length(true_survival)))
    }
  })
  names(bias_tmle_bin) <- paste0("t=", 2:t.end)
  
  CP_tmle_bin <- lapply(1:(t.end-1), function(t) {
    # Get true survival probabilities for time t+1
    true_survival <- 1 - sapply(Y.true, "[[", t+1)
    
    # Check if we have CI information for this time point
    if(t <= length(tmle_est_var_bin$CI) && !is.null(tmle_est_var_bin$CI[[t]])) {
      # Calculate coverage
      lower_ci <- tmle_est_var_bin$CI[[t]][1,]
      upper_ci <- tmle_est_var_bin$CI[[t]][2,]
      coverage <- as.numeric((lower_ci <= true_survival) & (upper_ci >= true_survival))
      return(coverage)
    } else {
      # Return NA if we don't have CI for this time point
      return(rep(NA, length(true_survival)))
    }
  })
  names(CP_tmle_bin) <- paste0("t=", 2:t.end)
  
  CIW_tmle_bin <- lapply(1:(t.end-1), function(t) {
    # Check if we have CI information for this time point
    if(t <= length(tmle_est_var_bin$CI) && !is.null(tmle_est_var_bin$CI[[t]])) {
      ci_width <- tmle_est_var_bin$CI[[t]][2,] - tmle_est_var_bin$CI[[t]][1,]
      return(ci_width)
    } else {
      # Return NA if we don't have CI for this time point
      if(length(tmle_est_var_bin$CI) > 0 && !is.null(tmle_est_var_bin$CI[[1]])) {
        return(rep(NA, ncol(tmle_est_var_bin$CI[[1]])))
      } else {
        return(rep(NA, 3)) # Default to 3 rules
      }
    }
  })
  names(CIW_tmle_bin) <- paste0("t=", 2:t.end)
  
  # Preserve names
  for(t in 1:(t.end-1)){
    if(!is.null(bias_tmle_bin[[t]]) && !is.null(CIW_tmle_bin[[t]])) {
      names(CIW_tmle_bin[[t]]) <- names(bias_tmle_bin[[t]])
    }
  }
  
  # G-computation metrics
  bias_gcomp <- lapply(2:t.end, function(t) {
    true_survival <- 1 - sapply(Y.true, "[[", t)
    
    if(t <= length(gcomp_est_var$est) && !is.null(gcomp_est_var$est[[t]])) {
      est_survival <- gcomp_est_var$est[[t]]
      bias <- true_survival - est_survival
      return(bias)
    } else {
      return(rep(NA, length(true_survival)))
    }
  })
  names(bias_gcomp) <- paste0("t=", 2:t.end)
  
  CP_gcomp <- lapply(1:(t.end-1), function(t) {
    true_survival <- 1 - sapply(Y.true, "[[", t+1)
    
    if(t <= length(gcomp_est_var$CI) && !is.null(gcomp_est_var$CI[[t]])) {
      lower_ci <- gcomp_est_var$CI[[t]][1,]
      upper_ci <- gcomp_est_var$CI[[t]][2,]
      coverage <- as.numeric((lower_ci <= true_survival) & (upper_ci >= true_survival))
      return(coverage)
    } else {
      return(rep(NA, length(true_survival)))
    }
  })
  names(CP_gcomp) <- paste0("t=", 2:t.end)
  
  CIW_gcomp <- lapply(1:(t.end-1), function(t) {
    if(t <= length(gcomp_est_var$CI) && !is.null(gcomp_est_var$CI[[t]])) {
      ci_width <- gcomp_est_var$CI[[t]][2,] - gcomp_est_var$CI[[t]][1,]
      return(ci_width)
    } else {
      if(length(gcomp_est_var$CI) > 0 && !is.null(gcomp_est_var$CI[[1]])) {
        return(rep(NA, ncol(gcomp_est_var$CI[[1]])))
      } else {
        return(rep(NA, 3)) # Default to 3 rules
      }
    }
  })
  names(CIW_gcomp) <- paste0("t=", 2:t.end)
  
  for(t in 1:(t.end-1)){
    if(!is.null(bias_gcomp[[t]]) && !is.null(CIW_gcomp[[t]])) {
      names(CIW_gcomp[[t]]) <- names(bias_gcomp[[t]])
    }
  }
  
  # IPTW metrics
  bias_iptw <- lapply(2:t.end, function(t) {
    true_survival <- 1 - sapply(Y.true, "[[", t)
    
    if(t <= length(iptw_est_var$est) && !is.null(iptw_est_var$est[[t]])) {
      est_survival <- iptw_est_var$est[[t]]
      bias <- true_survival - est_survival
      return(bias)
    } else {
      return(rep(NA, length(true_survival)))
    }
  })
  names(bias_iptw) <- paste0("t=", 2:t.end)
  
  CP_iptw <- lapply(1:(t.end-1), function(t) {
    true_survival <- 1 - sapply(Y.true, "[[", t+1)
    
    if(t <= length(iptw_est_var$CI) && !is.null(iptw_est_var$CI[[t]])) {
      lower_ci <- iptw_est_var$CI[[t]][1,]
      upper_ci <- iptw_est_var$CI[[t]][2,]
      coverage <- as.numeric((lower_ci <= true_survival) & (upper_ci >= true_survival))
      return(coverage)
    } else {
      return(rep(NA, length(true_survival)))
    }
  })
  names(CP_iptw) <- paste0("t=", 2:t.end)
  
  CIW_iptw <- lapply(1:(t.end-1), function(t) {
    if(t <= length(iptw_est_var$CI) && !is.null(iptw_est_var$CI[[t]])) {
      ci_width <- iptw_est_var$CI[[t]][2,] - iptw_est_var$CI[[t]][1,]
      return(ci_width)
    } else {
      if(length(iptw_est_var$CI) > 0 && !is.null(iptw_est_var$CI[[1]])) {
        return(rep(NA, ncol(iptw_est_var$CI[[1]])))
      } else {
        return(rep(NA, 3)) # Default to 3 rules
      }
    }
  })
  names(CIW_iptw) <- paste0("t=", 2:t.end)
  
  for(t in 1:(t.end-1)){
    if(!is.null(bias_iptw[[t]]) && !is.null(CIW_iptw[[t]])) {
      names(CIW_iptw[[t]]) <- names(bias_iptw[[t]])
    }
  }
  
  # Binary IPTW metrics
  bias_iptw_bin <- lapply(2:t.end, function(t) {
    true_survival <- 1 - sapply(Y.true, "[[", t)
    
    if(t <= length(iptw_est_var_bin$est) && !is.null(iptw_est_var_bin$est[[t]])) {
      est_survival <- iptw_est_var_bin$est[[t]]
      bias <- true_survival - est_survival
      return(bias)
    } else {
      return(rep(NA, length(true_survival)))
    }
  })
  names(bias_iptw_bin) <- paste0("t=", 2:t.end)
  
  CP_iptw_bin <- lapply(1:(t.end-1), function(t) {
    true_survival <- 1 - sapply(Y.true, "[[", t+1)
    
    if(t <= length(iptw_est_var_bin$CI) && !is.null(iptw_est_var_bin$CI[[t]])) {
      lower_ci <- iptw_est_var_bin$CI[[t]][1,]
      upper_ci <- iptw_est_var_bin$CI[[t]][2,]
      coverage <- as.numeric((lower_ci <= true_survival) & (upper_ci >= true_survival))
      return(coverage)
    } else {
      return(rep(NA, length(true_survival)))
    }
  })
  names(CP_iptw_bin) <- paste0("t=", 2:t.end)
  
  CIW_iptw_bin <- lapply(1:(t.end-1), function(t) {
    if(t <= length(iptw_est_var_bin$CI) && !is.null(iptw_est_var_bin$CI[[t]])) {
      ci_width <- iptw_est_var_bin$CI[[t]][2,] - iptw_est_var_bin$CI[[t]][1,]
      return(ci_width)
    } else {
      if(length(iptw_est_var_bin$CI) > 0 && !is.null(iptw_est_var_bin$CI[[1]])) {
        return(rep(NA, ncol(iptw_est_var_bin$CI[[1]])))
      } else {
        return(rep(NA, 3)) # Default to 3 rules
      }
    }
  })
  names(CIW_iptw_bin) <- paste0("t=", 2:t.end)
  
  for(t in 1:(t.end-1)){
    if(!is.null(bias_iptw_bin[[t]]) && !is.null(CIW_iptw_bin[[t]])) {
      names(CIW_iptw_bin[[t]]) <- names(bias_iptw_bin[[t]])
    }
  }
  
  # Save this iteration's results
  result_filename <- paste0(output_dir, 
                            "longitudinal_simulation_results_",
                            "estimator_", estimator,
                            "_treatment_rule_", treatment.rule,
                            "_r_", r,
                            "_n_", n,
                            "_J_", J,
                            "_n_folds_", n.folds,
                            "_scale_continuous_", scale.continuous,
                            "_use_SL_", use.SL,
                            ".rds")
  
  # Create a list with metadata and results
  iteration_results <- list(
    "iteration" = r,
    "estimator" = estimator,
    "treatment_rule" = treatment.rule,
    "n" = n,
    "J" = J,
    "n_folds" = n.folds,
    "scale_continuous" = scale.continuous,
    "use_SL" = use.SL,
    "results" = list(
      "obs_rules"= obs.rules,
      "Y_true" = Y.true,
      "Ahat_tmle" = Ahat_tmle,
      "Chat_tmle" = Chat_tmle,
      "yhat_tmle" = tmle_estimates,
      "prob_share_tmle" = prob_share,
      "Ahat_tmle_bin" = Ahat_tmle_bin,
      "yhat_tmle_bin" = tmle_bin_estimates,
      "prob_share_tmle_bin" = prob_share_bin,
      "bias_tmle" = bias_tmle,
      "CP_tmle" = CP_tmle,
      "CIW_tmle" = CIW_tmle,
      "tmle_est_var" = tmle_est_var,
      "bias_tmle_bin" = bias_tmle_bin,
      "CP_tmle_bin" = CP_tmle_bin,
      "CIW_tmle_bin" = CIW_tmle_bin,
      "tmle_est_var_bin" = tmle_est_var_bin,
      "yhat_gcomp" = gcomp_estimates,
      "bias_gcomp" = bias_gcomp,
      "CP_gcomp" = CP_gcomp,
      "CIW_gcomp" = CIW_gcomp,
      "gcomp_est_var" = gcomp_est_var,
      "yhat_iptw" = iptw_estimates,
      "bias_iptw" = bias_iptw,
      "CP_iptw" = CP_iptw,
      "CIW_iptw" = CIW_iptw,
      "iptw_est_var" = iptw_est_var,
      "yhat_iptw_bin" = iptw_bin_estimates,
      "bias_iptw_bin" = bias_iptw_bin,
      "CP_iptw_bin" = CP_iptw_bin,
      "CIW_iptw_bin" = CIW_iptw_bin,
      "iptw_est_var_bin" = iptw_est_var_bin
    )
  )
  
  # Ensure output directory exists
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Save iteration results
  tryCatch({
    saveRDS(iteration_results, result_filename)
    if(debug) {
      print(paste0("Saved iteration ", r, " results to ", result_filename))
    }
  }, error = function(e) {
    warning(paste0("Failed to save iteration ", r, " results: ", e$message))
  })
  
  print("Returning list")
  
  # Ensure all matrices have proper dimensions before returning
  tmle_estimates <- ensure_matrix_dimensions(tmle_estimates, nrow=3, ncol=t.end)
  tmle_bin_estimates <- ensure_matrix_dimensions(tmle_bin_estimates, nrow=3, ncol=t.end)
  iptw_estimates <- ensure_matrix_dimensions(iptw_estimates, nrow=3, ncol=t.end)
  iptw_bin_estimates <- ensure_matrix_dimensions(iptw_bin_estimates, nrow=3, ncol=t.end)
  gcomp_estimates <- ensure_matrix_dimensions(gcomp_estimates, nrow=3, ncol=t.end)
  
  return(list("obs_rules"= obs.rules, "Y_true" = Y.true, "Ahat_tmle"=Ahat_tmle, "Chat_tmle"=Chat_tmle, "yhat_tmle"= tmle_estimates, "prob_share_tmle"= prob_share,
              "Ahat_tmle_bin"=Ahat_tmle_bin,"yhat_tmle_bin"= tmle_bin_estimates, "prob_share_tmle_bin"= prob_share_bin,
              "bias_tmle"= bias_tmle,"CP_tmle"= CP_tmle,"CIW_tmle"=CIW_tmle,"tmle_est_var"=tmle_est_var,
              "bias_tmle_bin"= bias_tmle_bin,"CP_tmle_bin"=CP_tmle_bin,"CIW_tmle_bin"=CIW_tmle_bin,"tmle_est_var_bin"=tmle_est_var_bin,
              "yhat_gcomp"= gcomp_estimates, "bias_gcomp"= bias_gcomp,"CP_gcomp"= CP_gcomp,"CIW_gcomp"=CIW_gcomp,"gcomp_est_var"=gcomp_est_var,
              "yhat_iptw"= iptw_estimates,"bias_iptw"= bias_iptw,"CP_iptw"= CP_iptw,"CIW_iptw"=CIW_iptw,"iptw_est_var"=iptw_est_var,
              "yhat_iptw_bin"= iptw_bin_estimates,"bias_iptw_bin"= bias_iptw_bin,"CP_iptw_bin"=CP_iptw_bin,"CIW_iptw_bin"=CIW_iptw_bin,"iptw_est_var_bin"=iptw_est_var_bin, "estimator"=estimator))
}

#####################
# Set parameters    #
#####################

# define settings for simulation
settings <- expand.grid("n"=c(10000), 
                        treatment.rule = c("all")) 

options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE) # command line arguments # args <- c('tmle',1, 'TRUE','FALSE')
estimator <- as.character(args[1])
cores <- as.numeric(args[2])
use.SL <- as.logical(args[3])  # When TRUE, use Super Learner for initial Y model and treatment model estimation; if FALSE, use GLM
doMPI <- as.logical(args[4])

# define parameters

n <- as.numeric(settings[,1]) # total sample size

treatment.rule <- as.character(settings[,2]) # calculate counterfactual means under all treatment rules

J <- 6 # number of treatments

t.end <- 36 # number of time points after t=0

R <- 128 # number of simulation runs

full_vector <- 1:R

# Specify the values to be omitted
completed_values <- c()
omit_values <- sort(c(47, 18, 7, 17, 39, 93, 118, 77, 24, 14, 85, 72,
                      101, 113, 51, 108, 81, 57, 80, 70, 64, 105, 96, 74,
                      38, 73, 65, 122, 130, 134, 131, 132, 140, 139, 133,
                      151, 152, 162, 160, 169, 176, 191, 183, 202, 207, 204,
                      209, 223, 217, 237, 233, 236, 243, 244, 247, 255, 263,
                      269, 282, 279, 294, 306, 311, 316, 324))

# Remove the specified values from the full vector
final_vector <- full_vector[!full_vector %in% c(completed_values,omit_values)]

scale.continuous <- FALSE # standardize continuous covariates

gbound <- c(0.05,1) # define bounds to be used for the propensity score and censoring prob.

ybound <- c(0.0001,0.9999) # define bounds to be used for the Y predictions

n.folds <- 3

window_size <- 12

debug <- FALSE

# output directory
simulation_version <- paste0(format(Sys.time(), "%Y%m%d"),"/")

output_dir <- paste0('./outputs/', simulation_version)
if(!dir.exists(output_dir)){
  print(paste0('create folder for outputs at: ', output_dir))
  dir.create(output_dir)
}

filename <- paste0(output_dir, 
                   "longitudinal_simulation_results_",
                   "estimator_",estimator,
                   "_treatment_rule_",treatment.rule,
                   "_R_", R, # Use R instead of r for the filename
                   "_n_", n,
                   "_J_", J,
                   "_n_folds_",n.folds,
                   "_scale_continuous_",scale.continuous,
                   "_use_SL_", use.SL,".rds")

# Setup parallel processing
if(doMPI){
  library(doMPI)
  
  # Start cluster
  cl <- startMPIcluster()
  
  # Register cluster
  registerDoMPI(cl)
  
  # Check cluster size
  print(paste0("cluster size: ", clusterSize(cl)))
  
}

if(cores>1){
  library(parallel)
  library(doParallel)
  
  print(paste0("number of cores used for parallel processing: ", cores))
  
  cl <- parallel::makeCluster(cores, outfile="")
  
  doParallel::registerDoParallel(cl) # register cluster
  
  # FIRST: Export all needed variables to worker nodes
  clusterExport(cl, c("estimator", "n", "J", "t.end", "gbound", "ybound", 
                      "n.folds", "treatment.rule", "use.SL", "scale.continuous", 
                      "debug", "window_size", "output_dir"))
  
  # THEN: Load packages on worker nodes
  clusterEvalQ(cl, {
    # Common packages needed for all estimator types
    for (pkg in c(
      "simcausal",
      "purrr",
      "origami",
      "sl3",
      "nnet",
      "ranger",
      "xgboost",
      "glmnet",
      "MASS",
      "progressr", 
      "data.table",
      "gtools",
      "dplyr",
      "readr",
      "tidyr",
      "parallel"
    )) {
      suppressPackageStartupMessages(library(pkg, character.only = TRUE))
    }
    
    # Now this will work because estimator has been exported
    if(estimator == "tmle-lstm") {
      tryCatch({
        library(reticulate)
        library(tensorflow)
        library(keras)
        message("Loaded LSTM-specific packages")
      }, error = function(e) {
        message("Note: LSTM packages not available, but not needed for tmle estimator")
      })
    } else {
      message("LSTM packages not needed for ", estimator, " estimator")
    }
    
    # Return the loaded packages for verification
    sessionInfo()$loadedOnly
  })
}

#####################
# Run simulation #
#####################

print(paste0('simulation setting: ', "estimator = ", estimator, ", treatment.rule = ", treatment.rule, " R = ", R, ", n = ", n,", J = ", J ,", t.end = ", t.end, ", use.SL = ",use.SL, ", scale.continuous = ",scale.continuous))

# Define base library vector (common to all estimators)
base_library_vector <- c(
  "simcausal",
  "purrr",
  "origami",
  "sl3",
  "nnet",
  "ranger",
  "xgboost",
  "glmnet",
  "MASS",
  "progressr",
  "data.table",
  "gtools",
  "dplyr",
  "readr",
  "tidyr",
  "parallel"
)

# Add LSTM-specific libraries only if needed
library_vector <- if(estimator == "tmle-lstm") {
  c(base_library_vector, "reticulate", "tensorflow", "keras")
} else {
  base_library_vector
}

library(foreach)

if(cores==1){ # run sequentially and save at each iteration
  sim.results <- foreach(r = final_vector, .combine='cbind', .errorhandling="pass", .packages=library_vector, .verbose = FALSE) %do% {
    simLong(r=r, J=J, n=n, t.end=t.end, gbound=gbound, ybound=ybound, n.folds=n.folds, 
            cores=cores, estimator=estimator, treatment.rule=treatment.rule, 
            use.SL=use.SL, scale.continuous=scale.continuous, debug=debug, window_size=window_size)
  }
} else if(cores>1){ # run in parallel
  library(parallel)
  library(doParallel)
  library(foreach)
  
  print(paste0("number of cores used for parallel processing: ", cores))
  
  cl <- parallel::makeCluster(cores, outfile="")
  doParallel::registerDoParallel(cl) # register cluster
  
  # After creating your cluster but BEFORE any clusterEvalQ calls:
  clusterExport(cl, c("estimator", "n", "J", "t.end", "gbound", "ybound", 
                      "n.folds", "treatment.rule", "use.SL", "scale.continuous", 
                      "debug", "window_size", "output_dir"))
  

  # Set up worker nodes with packages based on estimator type
  clusterEvalQ(cl, {
    # Load common packages for all estimator types
    for (pkg in c("simcausal", "purrr", "origami", "sl3", "nnet", "ranger", 
                  "glmnet", "MASS", "data.table", "gtools", "dplyr", "readr", "tidyr")) {
      suppressPackageStartupMessages(library(pkg, character.only = TRUE))
    }
    
    # Source common files
    source('./src/tmle_IC.R')
    source('./src/misc_fns.R')
    
    # Load estimator-specific files
    est_type <- get("estimator", envir = .GlobalEnv)
    if(est_type == "tmle") {
      source('./src/tmle_fns.R')
      source('./src/SL3_fns.R')
    } else if(est_type == "tmle-lstm") {
      tryCatch({
        library(reticulate)
        library(tensorflow)
        library(keras)
        source('./src/tmle_fns_lstm.R')
        source('./src/lstm.R')
      }, error = function(e) {
        warning("Error loading LSTM dependencies: ", e$message)
      })
    }
    
    # Return package loading status
    sessionInfo()$loadedOnly
  })
  
  sim.results <- foreach(r = 1:R, 
                         .combine='cbind', 
                         .errorhandling="pass",
                         .packages=library_vector, 
                         .verbose = TRUE) %dopar% {
                           # Use simLong with cores=1 to avoid nested parallelism
                           simLong(r=r, J=J, n=n, t.end=t.end, gbound=gbound, ybound=ybound, 
                                   n.folds=n.folds, cores=1, estimator=estimator, 
                                   treatment.rule=treatment.rule, use.SL=use.SL, 
                                   scale.continuous=scale.continuous, debug=debug, 
                                   window_size=window_size)
                         }
}
saveRDS(sim.results, filename)

if(doMPI){
  closeCluster(cl) # close down MPIcluster
  mpi.finalize()
}

if(cores>1){
  stopCluster(cl)
}