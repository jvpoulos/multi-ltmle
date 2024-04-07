############################################################################################
# Longitudinal setting (T>1) simulations: Compare multinomial TMLE with binary TMLE        #
############################################################################################

######################
# Simulation function #
######################

simLong <- function(r, J=6, n=12500, t.end=36, gbound=c(0.05,1), ybound=c(0.0001,0.9999), n.folds=5, cores=1, estimator="tmle", treatment.rule = "all", use.SL=TRUE, scale.continuous=FALSE){
  
  # libraries
  library(simcausal)
  library(purrr)
  library(origami)
  library(sl3)
  options(sl3.verbose = TRUE)
  library(nnet)
  library(ranger)
  library(xgboost)
  library(glmnet)
  library(MASS)
  library(progressr)
  library(data.table)
  library(gtools)
  library(dplyr)
  library(readr)
  library(tidyr)
  
  if(estimator=='tmle-lstm'){
    library(reticulate)
    use_python("~/multi-ltmle/env/bin/python")
    print(py_config()) # Check Python configuration
    
    library(tensorflow)
    library(keras)
    print(is_keras_available())
    print(tf_version())
    
    source('./src/lstm.R')
  }
  
  if(estimator%in%c("tmle-lstm")){
    source('./src/tmle_fns_lstm.R')
  }
  
  if(estimator%in%c("tmle")){
    source('./src/tmle_fns.R')
    source('./src/SL3_fns.R', local =TRUE)
  }
  
  source('./src/tmle_IC.R')
  source('./src/misc_fns.R')
  
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
  
  if(use.SL==FALSE){
    warning("not tested on use.SL=FALSE")
  }
  
  if(scale.continuous==TRUE){
    warning("real values of certain time-varying covariates needed to characterize treatment rules")
  }
  
  # define DGP
  source('./src/simcausal_fns.R')
  source('./src/simcausal_dgp.R', local =TRUE)
  
  # specify intervention rules (t=0 is same as observed)
  Dset <- set.DAG(D, vecfun=c("StochasticFun")) # locks DAG, consistency checks
  
  if(r==1){
    png(paste0(output_dir,"DAG_plot.png"))
    plotDAG(Dset, 
            excludeattrs=c("C_0","Y_0"), 
            xjitter=0.95, 
            yjitter=0.2, 
            tmax = 3,
            customvlabs = c("V^1_0", "V^2_0", "V^3_0",
                            "L^1_0", "L^2_0", "L^3_0",
                            "L^1_1", "L^2_1", "L^3_1",
                            "L^1_2", "L^2_2", "L^3_2",
                            "L^1_3", "L^2_3", "L^3_3",
                            "A_0", "A_1", "A_2", "A_3", 
                            "C_1", "C_2","C_3","Y_1","Y_2","Y_3")) # plot DAG
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
  
  dat[["A_th1"]] <-sim(DAG = D.dyn1, actions = "A_th1", n = n, LTCF = "Y", rndseed = r) # static
  dat[["A_th2"]] <-sim(DAG = D.dyn2, actions = "A_th2", n = n, LTCF = "Y", rndseed = r)  # dynamic
  dat[["A_th3"]] <-sim(DAG = D.dyn3, actions = "A_th3", n = n, LTCF = "Y", rndseed = r) # stochastic
  
  # true parameter values
  
  D.dyn1 <- set.targetE(D.dyn1, outcome = "Y", t=1:t.end, param = "A_th1") # vector of counterfactual means of "Y" over all time periods
  D.dyn2 <- set.targetE(D.dyn2, outcome = "Y", t=1:t.end, param = "A_th2")
  D.dyn3 <- set.targetE(D.dyn3, outcome = "Y", t=1:t.end, param = "A_th3") 
  
  Y.true <- list()
  Y.true[["static"]] <- eval.target(D.dyn1, data = dat[["A_th1"]])$res
  Y.true[["dynamic"]] <- eval.target(D.dyn2, data = dat[["A_th2"]])$res
  Y.true[["stochastic"]] <- eval.target(D.dyn3, data = dat[["A_th3"]])$res
  
  # simulate observed data (under censoring) - sampled from (pre-intervention) distribution specified by DAG
  
  Odat <- sim(DAG = Dset, n = n, LTCF = "Y", rndseed = r) # survival outcome =1 after first occurance
  
  anodes <- grep("A",colnames(Odat),value=TRUE)
  cnodes <- grep("C",colnames(Odat),value=TRUE)
  ynodes <- grep("Y",colnames(Odat),value=TRUE)
  
  # store observed treatment assignment
  obs.treatment <- Odat[,anodes] # t=0,2,...,T
  
  treatments <- lapply(1:(t.end+1), function(t) as.data.frame(dummify(obs.treatment[,t])))
  
  for(t in 1:t.end){
    obs.treatment[,(1+t)] <- factor(obs.treatment[,(1+t)], levels = levels(addNA(obs.treatment[,(1+t)])), labels = c(levels(obs.treatment[,(1+t)]), 0), exclude = NULL)
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
                main = "Treatment rule adherence (simulated data)",
                legend.xyloc = "topright", xaxt="n")
    axis(1, at = seq(1, (t.end+1), by = 3))
    dev.off()
  }
  
  # store observed Ys
  Y.observed <- list()
  Y.observed[["static"]] <- sapply(1:t.end, function(t) mean(Odat[,ynodes][,paste0("Y_",t)][which(obs.rules[[t+1]][,"static"]==1)], na.rm=TRUE))
  Y.observed[["dynamic"]] <- sapply(1:t.end, function(t) mean(Odat[,ynodes][,paste0("Y_",t)][which(obs.rules[[t+1]][,"dynamic"]==1)], na.rm=TRUE))
  Y.observed[["stochastic"]] <- sapply(1:t.end, function(t) mean(Odat[,ynodes][,paste0("Y_",t)][which(obs.rules[[t+1]][,"stochastic"]==1)], na.rm=TRUE))
  Y.observed[["overall"]] <- sapply(1:t.end, function(t) mean(Odat[,ynodes][,paste0("Y_",t)], na.rm=TRUE))
  
  if(r==1){
    png(paste0(output_dir,paste0("survival_plot_truth_",n, ".png")))
    plotSurvEst(surv = list("Static"=1-Y.true[["static"]], "Dynamic"=1-Y.true[["dynamic"]], "Stochastic"=1-Y.true[["stochastic"]]),
                ylab = "Share of patients without diabetes diagnosis", 
                xlab = "Month",
                main = "Counterfactual outcomes (simulated data)",
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_observed_",n, ".png")))
    plotSurvEst(surv = list("Static"=1-Y.observed[["static"]], "Dynamic"=1-Y.observed[["dynamic"]], "Stochastic"=1-Y.observed[["stochastic"]]),
                ylab = "Share of patients without diabetes diagnosis", 
                xlab = "Month",
                main = "Observed outcomes (simulated data)",
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
  
  tmle_estimates <- list()
  iptw_estimates <- list()
  iptw_bin_estimates <- list()
  gcomp_estimates <- list()
  
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
  }else if(estimator=="tmle-lstm"){
    tmle_dat <- cbind(tmle_dat[,!colnames(tmle_dat)%in%c("V1","V2")], dummify(tmle_dat$V1), dummify(tmle_dat$V2)) # binarize categorical covariates
    
    if(scale.continuous){
      tmle_dat[c("V3","L1")] <- scale(tmle_dat[c("V3","L1")]) # scale continuous variables
    }
    
    colnames(tmle_dat) <- c("ID", "V3", "t", "L1", "L2", "L3", "A", "C", "Y", "white", "black", "latino", "other", "mdd", "bipolar", "schiz")
    
    tmle_dat$A <- factor(tmle_dat$A)
    
    # Reshape the time-varying variables to wide format
    #tmle_dat <- reshape(tmle_dat, idvar = "t", timevar = "ID", direction = "wide") # T x N
    tmle_dat <- reshape(tmle_dat, idvar = "ID", timevar = "t", direction = "wide") # N x T
    
    tmle_covars_Y <- tmle_covars_A <- tmle_covars_C <- c()
    tmle_covars_Y <- c(grep("L",colnames(tmle_dat),value = TRUE), 
                       grep("A", colnames(tmle_dat), value = TRUE), 
                       grep("V", colnames(tmle_dat), value = TRUE),
                       grep("white", colnames(tmle_dat), value = TRUE),
                       grep("black", colnames(tmle_dat), value = TRUE),
                       grep("latino", colnames(tmle_dat), value = TRUE),
                       grep("other", colnames(tmle_dat), value = TRUE),
                       grep("mdd", colnames(tmle_dat), value = TRUE),
                       grep("bipolar", colnames(tmle_dat), value = TRUE),
                       grep("schiz", colnames(tmle_dat), value = TRUE))
    tmle_covars_A <- setdiff(tmle_covars_Y, grep("A", colnames(tmle_dat), value = TRUE))
    tmle_covars_C <- tmle_covars_A
    
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
    
    tmle_dat[grep("A", colnames(tmle_dat), value = TRUE)] <- lapply(tmle_dat[grep("A", colnames(tmle_dat), value = TRUE)], function(x){`levels<-`(addNA(x), c(0,levels(x)))})
  }
  
  ##  fit initial treatment model
  
  # multinomial
  
  if(estimator=="tmle"){
    
    initial_model_for_A_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                           learners = if(use.SL) learner_stack_A else make_learner(Lrnr_glm),
                                           metalearner = metalearner_A,
                                           keep_extra=FALSE)
    
    initial_model_for_A <- lapply(0:t.end, function(t){ # going forward in time
      
      tmle_dat_sub <- tmle_dat[tmle_dat$t==t,][!colnames(tmle_dat)%in%c("Y","C")]
      
      folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
      
      # define task 
      
      initial_model_for_A_task <- make_sl3_Task(tmle_dat_sub, 
                                                covariates = tmle_covars_A,
                                                outcome = "A", 
                                                outcome_type="categorical", 
                                                folds = folds) 
      
      # train
      initial_model_for_A_sl_fit <- initial_model_for_A_sl$train(initial_model_for_A_task)
      
      return(list("preds"=initial_model_for_A_sl_fit$predict(initial_model_for_A_task),
                  "folds"= folds,
                  "task"=initial_model_for_A_task,
                  "fit"=initial_model_for_A_sl_fit,
                  "data"=tmle_dat_sub)) 
    })
    
    g_preds <- lapply(1:length(initial_model_for_A), function(i) data.frame(matrix(unlist(lapply(initial_model_for_A[[i]]$preds, unlist)), nrow=length(lapply(initial_model_for_A[[i]]$preds, unlist)), byrow=TRUE)) ) # t length list of estimated propensity scores 
    g_preds <- lapply(1:length(initial_model_for_A), function(x) setNames(g_preds[[x]], grep("A[0-9]$",colnames(tmle_dat), value=TRUE)) )
    
    g_preds_ID <- lapply(1:length(initial_model_for_A), function(i) unlist(lapply(initial_model_for_A[[i]]$data$ID, unlist))) 
    
    g_preds_cuml <- vector("list", length(g_preds))
    
    g_preds_cuml[[1]] <- g_preds[[1]]
    
    for (i in 2:length(g_preds)) {
      g_preds_cuml[[i]] <- g_preds[[i]][which(g_preds_ID[[i-1]]%in%g_preds_ID[[i]]),] * g_preds_cuml[[i-1]][which(g_preds_ID[[i-1]]%in%g_preds_ID[[i]]),]
    }
    
    g_preds_cuml_bounded <- lapply(1:length(initial_model_for_A), function(x) boundProbs(g_preds_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores                             
    
  } else if(estimator=="tmle-lstm"){ 
    window.size <- 7
    
    lstm_A_preds <- lstm(data=tmle_dat[c(grep("A",colnames(tmle_dat),value = TRUE),tmle_covars_A)], outcome=grep("A", colnames(tmle_dat), value = TRUE), covariates = tmle_covars_A, t_end=t.end, window_size=window.size, out_activation="softmax", loss_fn = "sparse_categorical_crossentropy", output_dir, J=7) # list of 27 matrices
    
    lstm_A_preds <- c(replicate((window.size), lstm_A_preds[[window.size+1]], simplify = FALSE), lstm_A_preds[(window.size+1):length(lstm_A_preds)]) # extend to list of 37 matrices by assuming predictions in t=1....window.size is the same as window_size+1    
    initial_model_for_A <- list("preds"= lstm_A_preds,
                                "data"=tmle_dat)
    
    g_preds <- lapply(lstm_A_preds, function(prediction_matrix) {
      # Check if the matrix has the expected dimensions (n by J+1)
      dims <- dim(prediction_matrix)
      if (length(dims) == 2 && dims[2] == J + 1) {
        # Extract the columns corresponding to classes 1 to J
        reshaped_matrix <- prediction_matrix[, -1, drop = FALSE]
        
        # Naming the columns
        colnames(reshaped_matrix) <- paste0("A", 1:J)
        
        return(reshaped_matrix)
      } else {
        warning("Incorrect dimensions for prediction matrix")
        return(NULL)
      }
    })
    
    g_preds_cuml <- vector("list", length(g_preds))
    
    g_preds_cuml[[1]] <- g_preds[[1]]
    
    for (i in 2:length(g_preds)) {
      g_preds_cuml[[i]] <- g_preds[[i]] * g_preds_cuml[[i-1]]
    }
    
    g_preds_cuml_bounded <- lapply(1:length(g_preds), function(x) boundProbs(g_preds_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores                             
  }
  
  # binomial
  
  if(estimator=="tmle"){
    
    initial_model_for_A_sl_bin <- make_learner(Lrnr_sl, # cross-validates base models
                                               learners = if(use.SL) learner_stack_A_bin else make_learner(Lrnr_glm),
                                               metalearner = metalearner_A_bin,
                                               keep_extra=FALSE)
    
    initial_model_for_A_bin <- lapply(0:t.end, function(t){
      
      tmle_dat_sub <- tmle_dat[tmle_dat$t==t,][!colnames(tmle_dat)%in%c("C","Y")]
      
      # define cross-validation
      folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
      
      # define task and candidate learners
      
      initial_model_for_A_task_bin <- lapply(1:J, function(j) make_sl3_Task(cbind("A"=dummify(tmle_dat_sub$A)[,j],tmle_dat_sub[tmle_covars_A]), 
                                                                            covariates = tmle_covars_A, 
                                                                            outcome = "A",
                                                                            outcome_type="binomial",
                                                                            folds = folds)) 
      # train
      
      initial_model_for_A_sl_fit_bin <- lapply(1:J, function(j) initial_model_for_A_sl_bin$train(initial_model_for_A_task_bin[[j]]))
      
      return(list("preds"=sapply(1:J, function(j) initial_model_for_A_sl_fit_bin[[j]]$predict(initial_model_for_A_task_bin[[j]])), 
                  "folds"= folds,
                  "task"=initial_model_for_A_task_bin,
                  "fit"=initial_model_for_A_sl_fit_bin,
                  "data"=tmle_dat_sub)) 
    })
    
    g_preds_bin <- lapply(1:length(initial_model_for_A_bin), function(i) data.frame(initial_model_for_A_bin[[i]]$preds) ) # t length list of estimated propensity scores 
    g_preds_bin <- lapply(1:length(initial_model_for_A_bin), function(x) setNames(g_preds_bin[[x]], grep("A[0-9]$",colnames(tmle_dat), value=TRUE)) )
    
    g_preds_bin_ID <- lapply(1:length(initial_model_for_A_bin), function(i) unlist(lapply(initial_model_for_A_bin[[i]]$data$ID, unlist)))
    
    g_preds_bin_cuml <- vector("list", length(g_preds_bin))
    
    g_preds_bin_cuml[[1]] <- g_preds_bin[[1]]
    
    for (i in 2:length(g_preds_bin)) {
      g_preds_bin_cuml[[i]] <- g_preds_bin[[i]][which(g_preds_bin_ID[[i-1]]%in%g_preds_bin_ID[[i]]),] * g_preds_bin_cuml[[i-1]][which(g_preds_bin_ID[[i-1]]%in%g_preds_bin_ID[[i]]),]
    }    
    g_preds_bin_cuml_bounded <- lapply(1:length(initial_model_for_A_bin), function(x) boundProbs(g_preds_bin_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores                             
  } else if(estimator=="tmle-lstm"){ 
    
    # Define the number of treatment classes
    num_classes <- J
    
    lstm_A_preds_bin <- list()
    for (k in 1:num_classes) {
      print(paste0("Class ", k, " in ", num_classes))
      
      # Create a binary matrix for the outcomes where each 'A' column is compared to class k
      A_binary_matrix <- sapply(grep("A", colnames(tmle_dat), value = TRUE), function(col_name) {
        as.numeric(tmle_dat[[col_name]] == k)  # 1 if treatment is class k, 0 otherwise
      })
      
      print(dim(A_binary_matrix))
      
      # Combine with covariates data
      lstm_input_data <- cbind(A_binary_matrix, tmle_dat[tmle_covars_A])
      
      # Train the LSTM model with the prepared data
      lstm_A_preds_bin[[k]] <- lstm(data = lstm_input_data, 
                                    outcome = colnames(A_binary_matrix), 
                                    covariates = tmle_covars_A, 
                                    t_end = t.end, 
                                    window_size = window.size, 
                                    out_activation = "sigmoid", 
                                    loss_fn = "binary_crossentropy", 
                                    output_dir) # [[J (1:6)][[timesteps (1:27)]]]
    }
    
    # Optionally, force garbage collection after the loop
    gc()
    
    # Initialize the transformed list
    transformed_preds_bin <- vector("list", length = 6) # for 6 treatment classes
    
    for (class in 1:J) {
      # Extract all time-step predictions for the current treatment class
      time_step_preds <- lstm_A_preds_bin[[class]]
      
      # Combine the predictions into a single matrix (n by timesteps)
      # Assuming each element in time_step_preds is a vector of length n
      combined_matrix <- do.call(cbind, time_step_preds)
      
      # Assign the combined matrix to the transformed list
      transformed_preds_bin[[class]] <- combined_matrix
    } # Now, transformed_preds_bin is a list of 6 elements, each a n by 27 matrix
    
    for (i in 1:length(transformed_preds_bin)) {
      # Get the first column and replicate it (window.size) times
      first_col_replicated <- matrix(rep(transformed_preds_bin[[i]][, 1], window.size), 
                                     nrow = n, ncol = window.size)
      
      # Combine the replicated columns with the original matrix
      extended_matrix <- cbind(first_col_replicated, transformed_preds_bin[[i]])
      
      # Add the extended matrix to lstm_A_preds_bin
      lstm_A_preds_bin[[i]] <- extended_matrix
    }
    
    # Check the dimension of the first matrix in lstm_A_preds_bin
    dim(lstm_A_preds_bin[[1]])
    
    initial_model_for_A_bin <- list("preds" = lstm_A_preds_bin, "data" = tmle_dat)
    
    # Process each matrix in lstm_A_preds_bin
    g_preds_bin <- lstm_A_preds_bin  # No need to reshape
    
    g_preds_bin_ID <- tmle_dat$ID
    
    # Initialize g_preds_bin_cuml and compute cumulative predictions
    g_preds_bin_cuml <- vector("list", length(g_preds_bin))
    g_preds_bin_cuml[[1]] <- g_preds_bin[[1]]
    
    for (i in 2:length(g_preds_bin)) {
      # Check if the dimensions match
      if (nrow(g_preds_bin[[i]]) == nrow(g_preds_bin_cuml[[i - 1]]) && ncol(g_preds_bin[[i]]) == ncol(g_preds_bin_cuml[[i - 1]])) {
        # Perform element-wise multiplication
        g_preds_bin_cuml[[i]] <- g_preds_bin[[i]] * g_preds_bin_cuml[[i - 1]]
      } else {
        warning("Dimension mismatch between g_preds_bin[", i, "] and g_preds_bin_cuml[", i - 1, "]")
      }
    }
    
    # Initialize an empty list for the reshaped predictions
    reshaped_preds_bin <- vector("list", length = t.end)
    
    # Loop over each time point
    for (i in 1:t.end) {
      # Extract predictions for each treatment class at time point i
      temp_matrices <- lapply(lstm_A_preds_bin, function(x) {
        # Convert each element of lstm_A_preds_bin into a n x 37 matrix
        matrix(x, nrow = n, ncol = t.end)
      })
      
      # Extract the ith column from each matrix and combine them to create a single matrix
      combined_matrix <- do.call(cbind, lapply(temp_matrices, function(m) m[, i]))
      
      # Assign the combined matrix to the reshaped_preds_bin list
      reshaped_preds_bin[[i]] <- combined_matrix
    }
    
    # Apply boundProbs to each cumulative prediction
    g_preds_bin_cuml_bounded <- lapply(reshaped_preds_bin, function(x) boundProbs(x, bounds = gbound))
  }
  
  ##  fit initial censoring model
  ## implicitly fit on those that are uncensored until t-1
  
  if(estimator=="tmle"){
    
    initial_model_for_C_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                           learners = if(use.SL) learner_stack_A_bin else make_learner(Lrnr_glm),
                                           metalearner = metalearner_A_bin,
                                           keep_extra=FALSE)
    
    initial_model_for_C <- lapply(0:t.end, function(t){
      
      tmle_dat_sub <- tmle_dat[tmle_dat$t==t,][!colnames(tmle_dat)%in%c("A","Y")]
      
      folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
      
      # define task and candidate learners
      initial_model_for_C_task <- make_sl3_Task(data=tmle_dat_sub,
                                                covariates = tmle_covars_C, 
                                                outcome = "C",
                                                outcome_type="binomial", 
                                                folds = folds) 
      
      # train
      
      initial_model_for_C_sl_fit <- initial_model_for_C_sl$train(initial_model_for_C_task)
      
      return(list("preds"=initial_model_for_C_sl_fit$predict(initial_model_for_C_task),
                  "folds"= folds,
                  "task"=initial_model_for_C_task,
                  "fit"=initial_model_for_C_sl_fit,
                  "data"=tmle_dat_sub)) 
    })
    
    C_preds <- lapply(1:length(initial_model_for_C), function(i) 1-initial_model_for_C[[i]]$preds) # t length list # C=1 if uncensored; C=0 if censored  
    
    C_preds_ID <- lapply(1:length(initial_model_for_C), function(i) unlist(lapply(initial_model_for_C[[i]]$data$ID, unlist))) 
    
    C_preds_cuml <- vector("list", length(C_preds))
    
    C_preds_cuml[[1]] <- C_preds[[1]]
    
    for (i in 2:length(C_preds)) {
      C_preds_cuml[[i]] <- C_preds[[i]][which(C_preds_ID[[i-1]]%in%C_preds_ID[[i]])] * C_preds_cuml[[i-1]][which(C_preds_ID[[i-1]]%in%C_preds_ID[[i]])]
    }    
    C_preds_cuml_bounded <- lapply(1:length(initial_model_for_C), function(x) boundProbs(C_preds_cuml[[x]],bounds=gbound))  # winsorized cumulative bounded censoring predictions, 1=Censored                            
  }else if(estimator=="tmle-lstm"){ 
    
    lstm_C_preds <- lstm(data=tmle_dat[c(grep("C",colnames(tmle_dat),value = TRUE),tmle_covars_C)], outcome=grep("C",colnames(tmle_dat),value = TRUE), covariates=tmle_covars_C, t_end=t.end, window_size=window.size, out_activation="sigmoid", loss_fn = "binary_crossentropy", output_dir) # list of 27, n preds
    
    # Transform lstm_C_preds into a data matrix of n x 37
    transformed_C_preds <- c(replicate(window.size, lstm_C_preds[[1]], simplify = FALSE), lstm_C_preds)
    
    # Initialize initial_model_for_C with the transformed predictions
    initial_model_for_C <- list("preds" = transformed_C_preds, "data" = tmle_dat)
    
    # Process the transformed predictions
    C_preds <- lapply(1:length(initial_model_for_C$preds), function(i) 1 - initial_model_for_C$preds[[i]])
    
    # Assuming tmle_dat has a column 'ID' that contains the IDs
    C_preds_ID <- replicate(length(C_preds), 1:n, simplify = FALSE)
    
    # Initialize C_preds_cuml and compute cumulative predictions
    C_preds_cuml <- vector("list", length(C_preds))
    C_preds_cuml[[1]] <- C_preds[[1]]
    
    for (i in 2:length(C_preds)) {
      # Find common IDs
      common_ids <- intersect(C_preds_ID[[i]], C_preds_ID[[i-1]])
      
      if (length(common_ids) > 0) {
        # Calculate cumulative predictions for common IDs
        common_indices_i <- match(common_ids, C_preds_ID[[i]])
        common_indices_i_minus_1 <- match(common_ids, C_preds_ID[[i-1]])
        
        # Check if C_preds[[i]] is a vector or a matrix
        if (is.vector(C_preds[[i]]) || length(dim(C_preds[[i]])) == 1) {
          # Handle the vector case
          C_preds_cuml[[i]] <- C_preds[[i]][common_indices_i] * C_preds_cuml[[i-1]][common_indices_i_minus_1]
        } else {
          # Handle the matrix case
          C_preds_cuml[[i]] <- C_preds[[i]][common_indices_i, ] * C_preds_cuml[[i-1]][common_indices_i_minus_1, ]
        }
      } else {
        warning("No common IDs found for iteration ", i)
      }
    }
    
    # Apply boundProbs to each cumulative prediction
    C_preds_cuml_bounded <- lapply(C_preds_cuml, function(x) boundProbs(x, bounds = gbound))
  }
  
  ## sequential g-formula
  ## model is fit on all uncensored and alive (until t-1)
  ## the outcome is the observed Y for t=T and updated Y if t<T
  
  if(estimator=="tmle"){
    
    initial_model_for_Y_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                           learners = if(use.SL) learner_stack_Y else make_learner(Lrnr_glm),
                                           metalearner = metalearner_Y,
                                           keep_extra=FALSE)
    
    initial_model_for_Y_sl_cont <- make_learner(Lrnr_sl, # cross-validates base models
                                                learners = if(use.SL) learner_stack_Y_cont else make_learner(Lrnr_glm),
                                                metalearner = metalearner_Y_cont,
                                                keep_extra=FALSE)
    
    initial_model_for_Y <- list()
    initial_model_for_Y_bin <- list() # updated Y's are used as outcomes for t<T
    initial_model_for_Y[[t.end]] <- sequential_g(t=t.end, tmle_dat=tmle_dat[!tmle_dat$ID%in%time.censored$ID[which(time.censored$time_censored<(t.end+1))],], n.folds=n.folds, tmle_covars_Y=tmle_covars_Y, initial_model_for_Y_sl, ybound) # for t=T fit on measured Y
    initial_model_for_Y_bin[[t.end]] <- initial_model_for_Y[[t.end]] # same
    
    # Update equations, calculate point estimate and variance
    # backward in time: T.end, ..., 1
    # fit on thoese uncesnored until t-1
    
    tmle_rules <- list("static"=static_mtp,
                       "dynamic"=dynamic_mtp,
                       "stochastic"=stochastic_mtp)
    
    tmle_contrasts <-list()
    tmle_contrasts_bin <- list()
    tmle_contrasts[[t.end]] <- getTMLELong(initial_model_for_Y=initial_model_for_Y[[t.end]], tmle_rules=tmle_rules, tmle_covars_Y=tmle_covars_Y, g_preds_bounded=g_preds_cuml_bounded[[t.end+1]], C_preds_bounded=C_preds_cuml_bounded[[t.end+1]], obs.treatment=treatments[[t.end+1]], obs.rules=obs.rules[[t.end+1]], gbound=gbound, ybound=ybound, t.end=t.end)
    tmle_contrasts_bin[[t.end]] <- getTMLELong(initial_model_for_Y=initial_model_for_Y_bin[[t.end]], tmle_rules=tmle_rules, tmle_covars_Y=tmle_covars_Y, g_preds_bounded=g_preds_bin_cuml_bounded[[t.end+1]], C_preds_bounded=C_preds_cuml_bounded[[t.end+1]], obs.treatment=treatments[[t.end+1]], obs.rules=obs.rules[[t.end+1]], gbound=gbound, ybound=ybound, t.end=t.end)
    
    for(t in (t.end-1):1){
      initial_model_for_Y[[(t)]] <- sapply(1:length(tmle_rules), function(i) sequential_g(t=t, tmle_dat=tmle_dat[!tmle_dat$ID%in%time.censored$ID[which(time.censored$time_censored<t)],], n.folds=n.folds, tmle_covars_Y=tmle_covars_Y, initial_model_for_Y_sl=initial_model_for_Y_sl_cont, ybound=ybound, Y_pred = tmle_contrasts[[t+1]]$Qstar[[i]]))
      initial_model_for_Y_bin[[(t)]] <- sapply(1:length(tmle_rules), function(i) sequential_g(t=t, tmle_dat=tmle_dat[!tmle_dat$ID%in%time.censored$ID[which(time.censored$time_censored<t)],], n.folds=n.folds, tmle_covars_Y=tmle_covars_Y,  initial_model_for_Y_sl=initial_model_for_Y_sl_cont, ybound=ybound, Y_pred =tmle_contrasts_bin[[t+1]]$Qstar[[i]]))
      
      tmle_contrasts[[t]] <- sapply(1:length(tmle_rules), function(i) getTMLELong(initial_model_for_Y=initial_model_for_Y[[t]][,i], tmle_rules=tmle_rules, tmle_covars_Y=tmle_covars_Y, g_preds_bounded=g_preds_cuml_bounded[[t+1]], C_preds_bounded=C_preds_cuml_bounded[[t+1]], obs.treatment=treatments[[t+1]], obs.rules=obs.rules[[t+1]], gbound=gbound, ybound=ybound, t.end=t.end))
      tmle_contrasts_bin[[t]] <- sapply(1:length(tmle_rules), function(i) getTMLELong(initial_model_for_Y=initial_model_for_Y_bin[[t]][,i], tmle_rules=tmle_rules, tmle_covars_Y=tmle_covars_Y, g_preds_bounded=g_preds_bin_cuml_bounded[[t+1]], C_preds_bounded=C_preds_cuml_bounded[[t+1]], obs.treatment=treatments[[t+1]], obs.rules=obs.rules[[t+1]], gbound=gbound, ybound=ybound, t.end=t.end))
    }
  } else if(estimator=='tmle-lstm'){
    
    lstm_Y_preds <- lstm(data=tmle_dat[c(grep("Y",colnames(tmle_dat),value = TRUE),tmle_covars_Y)], outcome=grep("Y",colnames(tmle_dat),value = TRUE), covariates=tmle_covars_Y, t_end=t.end, window_size=window.size, out_activation="sigmoid", loss_fn = "binary_crossentropy", output_dir)
    
    # Transform lstm_Y_preds into a data matrix of n x t.end
    transformed_Y_preds <- do.call(cbind, c(replicate(window.size, lstm_Y_preds[[1]], simplify = FALSE), lstm_Y_preds))
    
    # Reshape tmle_dat from wide to long format
    long_data <- tmle_dat %>%
      # Convert all columns to character type to avoid data type conflicts
      mutate(across(-t, as.character)) %>%
      pivot_longer(
        cols = -t,  # Select all columns except 't'
        names_to = "covariate_id",
        values_to = "value"
      ) %>%
      separate(covariate_id, into = c("covariate", "ID"), sep = "\\.") %>%
      pivot_wider(
        names_from = covariate,
        values_from = value,
        id_cols = c(t, ID)
      )
    
    # Convert all columns to numeric except 'A', which is converted to a factor
    long_data <- long_data %>%
      mutate(across(where(is.character), as.numeric),  # Convert all character columns to numeric
             A = as.factor(A))  # Convert 'A' to a factor
    
    long_data <- as.data.frame(long_data)
    
    # Rename the columns to remove the "ID" suffix
    names(long_data) <- gsub("\\.ID", "", names(long_data))
    
    # Create initial_model_for_Y using the transformed predictions
    initial_model_for_Y <- list("preds" = boundProbs(transformed_Y_preds, ybound), "data" = long_data)
    
    tmle_rules <- list("static" = static_mtp_lstm,
                       "dynamic" = dynamic_mtp_lstm,
                       "stochastic" = stochastic_mtp_lstm)
    
    tmle_contrasts <- list()
    tmle_contrasts_bin <- list()
    
    tmle_contrasts[[t.end]] <- getTMLELongLSTM(
      initial_model_for_Y_preds = initial_model_for_Y$preds[, t.end], 
      initial_model_for_Y_data = initial_model_for_Y$data, 
      tmle_rules = tmle_rules,
      tmle_covars_Y=tmle_covars_Y, 
      g_preds_bounded=g_preds_cuml_bounded[[t.end+1]], 
      C_preds_bounded=C_preds_cuml_bounded[[t.end]], 
      obs.treatment=treatments[[t.end+1]], 
      obs.rules=obs.rules[[t.end+1]], 
      gbound=gbound, ybound=ybound, t.end=t.end, window.size=window.size
    )
    
    tmle_contrasts_bin[[t.end]] <- getTMLELongLSTM(
      initial_model_for_Y_preds = initial_model_for_Y$preds[, t.end], 
      initial_model_for_Y_data = initial_model_for_Y$data,
      tmle_rules = tmle_rules,
      tmle_covars_Y=tmle_covars_Y, 
      g_preds_bounded=g_preds_bin_cuml_bounded[[t.end]], 
      C_preds_bounded=C_preds_cuml_bounded[[t.end]], 
      obs.treatment=treatments[[t.end+1]], 
      obs.rules=obs.rules[[t.end+1]], 
      gbound=gbound, ybound=ybound, t.end=t.end, window.size=window.size
    )
    
    # Pre-compute lengths and other constants outside the loop
    tmle_rules_length <- length(tmle_rules)
    initial_model_for_Y_data <- initial_model_for_Y$data
    
    # Use parallel processing to compute tmle_contrasts
    tmle_contrasts <- mclapply(1:(t.end - 1), function(t) {
      sapply(1:tmle_rules_length, function(i) {
        getTMLELongLSTM(
          initial_model_for_Y_preds = initial_model_for_Y$preds[, t],
          initial_model_for_Y_data = initial_model_for_Y_data,
          tmle_rules = tmle_rules,
          tmle_covars_Y = tmle_covars_Y,
          g_preds_bounded = g_preds_cuml_bounded[[t + 1]],
          C_preds_bounded = C_preds_cuml_bounded[[t]],
          obs.treatment = treatments[[t + 1]],
          obs.rules = obs.rules[[t + 1]],
          gbound = gbound, ybound = ybound, t.end = t.end, window.size = window.size
        )
      })
    }, mc.cores = cores)
    
    # Use parallel processing to compute tmle_contrasts_bin
    tmle_contrasts_bin <- mclapply(1:(t.end - 1), function(t) {
      sapply(1:tmle_rules_length, function(i) {
        getTMLELongLSTM(
          initial_model_for_Y_preds = initial_model_for_Y$preds[, t],
          initial_model_for_Y_data = initial_model_for_Y_data,
          tmle_rules = tmle_rules,
          tmle_covars_Y = tmle_covars_Y,
          g_preds_bounded = g_preds_bin_cuml_bounded[[t]],
          C_preds_bounded = C_preds_cuml_bounded[[t]],
          obs.treatment = treatments[[t + 1]],
          obs.rules = obs.rules[[t + 1]],
          gbound = gbound, ybound = ybound, t.end = t.end, window.size = window.size
        )
      })
    }, mc.cores = cores)
  }
    
  # plot estimated survival curves
  
  tmle_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar[[x]]))) # static, dynamic, stochastic
  tmle_bin_estimates <-  cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t]][,x]$Qstar[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t.end]]$Qstar[[x]]))) 
  
  if(r==1){
    png(paste0(output_dir,paste0("survival_plot_tmle_estimates_",n, estimator,".png")))
    plotSurvEst(surv = list("Static"= tmle_estimates[1,], "Dynamic"= tmle_estimates[2,], "Stochastic"= tmle_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "TMLE (ours, multinomial) estimated counterfactuals",
                xlab = "Month",
                ylim = c(0.5,1),
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_tmle_estimates_bin_",n, estimator,".png")))
    plotSurvEst(surv = list("Static"= tmle_bin_estimates[1,], "Dynamic"= tmle_bin_estimates[2,], "Stochastic"= tmle_bin_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "TMLE (ours, binomial) estimated counterfactuals",
                xlab = "Month",
                ylim = c(0.5,1),
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
  }
  
  iptw_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar_iptw[[x]], na.rm=TRUE))) # static, dynamic, stochastic
  iptw_bin_estimates <-  cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t.end]]$Qstar_iptw[[x]], na.rm=TRUE))) 
  
  if(r==1){
    png(paste0(output_dir,paste0("survival_plot_iptw_estimates_",n, estimator,".png")))
    plotSurvEst(surv = list("Static"= iptw_estimates[1,], "Dynamic"= iptw_estimates[2,], "Stochastic"= iptw_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "IPTW (ours, multinomial) estimated counterfactuals",
                xlab = "Month",
                ylim = c(0.5,1),
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_iptw_bin_estimates_",n, estimator,".png")))
    plotSurvEst(surv = list("Static"= iptw_bin_estimates[1,], "Dynamic"= iptw_bin_estimates[2,], "Stochastic"= iptw_bin_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "IPTW (ours, binomial) estimated counterfactuals",
                xlab = "Month",
                ylim = c(0.5,1),
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
  }
  
  gcomp_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar_gcomp[[x]]))) # static, dynamic, stochastic
  
  if(r==1){
    png(paste0(output_dir,paste0("survival_plot_gcomp_estimates_",n, estimator,".png")))
    plotSurvEst(surv = list("Static"= gcomp_estimates[1,], "Dynamic"= gcomp_estimates[2,], "Stochastic"= gcomp_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "G-comp. (ours) estimated counterfactuals",
                xlab = "Month",
                ylim = c(0.5,1),
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
  }
  
  # calc share of cumulative probabilities of continuing to receive treatment according to the assigned treatment rule which are smaller than 0.025
  
  prob_share <- lapply(1:(t.end+1), function(t) sapply(1:ncol(obs.rules[[(t)]]), function(i) colMeans(g_preds_cuml_bounded[[(t)]][which(obs.rules[[(t)]][na.omit(tmle_dat[tmle_dat$t==(t),])$ID,][,i]==1),]<0.025, na.rm=TRUE)))
  for(t in 1:(t.end+1)){
    colnames(prob_share[[t]]) <- colnames(obs.rules[[(t.end)]])
  }
  
  names(prob_share) <- paste0("t=",seq(0,t.end))
  
  prob_share.bin <- lapply(1:(t.end+1), function(t) sapply(1:ncol(obs.rules[[(t)]]), function(i) colMeans(g_preds_bin_cuml_bounded[[(t)]][which(obs.rules[[(t)]][na.omit(tmle_dat[tmle_dat$t==(t),])$ID,][,i]==1),]<0.025, na.rm=TRUE)))
  for(t in 1:(t.end+1)){
    colnames(prob_share.bin[[t]]) <- colnames(obs.rules[[(t.end)]])
  }
  
  names(prob_share.bin) <- paste0("t=",seq(0,t.end))
  
  # calc CIs 
  
  tmle_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored)
  tmle_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y_bin, time.censored)
  
  iptw_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, iptw=TRUE)
  iptw_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y_bin, time.censored, iptw=TRUE)
  
  gcomp_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, gcomp=TRUE)
  
  # store results
  
  Ahat_tmle  <- g_preds_cuml_bounded
  Ahat_tmle_bin  <- g_preds_bin_cuml_bounded
  
  Chat_tmle  <- C_preds_cuml_bounded
  
  # calculate bias, CP, CIW wrt to est at each t
  bias_tmle  <- lapply(2:t.end, function(t) sapply(Y.true,"[[",t) - tmle_est_var$est[[t]])
  names(bias_tmle) <- paste0("t=",2:t.end)
  
  CP_tmle <- lapply(1:(t.end-1), function(t) as.numeric((tmle_est_var$CI[[t]][1,] < sapply(Y.true,"[[",t)) & (tmle_est_var$CI[[t]][2,] > sapply(Y.true,"[[",t))))
  names(CP_tmle) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CP_tmle[[t]]) <- names(bias_tmle[[t]])
  }
  
  CIW_tmle  <- lapply(1:(t.end-1), function(t) tmle_est_var$CI[[t]][2,]- tmle_est_var$CI[[t]][1,])
  names(CIW_tmle) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CIW_tmle[[t]]) <- names(bias_tmle[[t]])
  }
  
  # binomial version
  
  bias_tmle_bin  <- lapply(2:t.end, function(t) sapply(Y.true,"[[",t) - tmle_est_var_bin$est[[t]])
  names(bias_tmle_bin) <- paste0("t=",2:t.end)
  
  CP_tmle_bin <- lapply(1:(t.end-1), function(t) as.numeric((tmle_est_var_bin$CI[[t]][1,] < sapply(Y.true,"[[",t)) & (tmle_est_var_bin$CI[[t]][2,] > sapply(Y.true,"[[",t))))
  names(CP_tmle_bin) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CP_tmle_bin[[t]]) <- names(bias_tmle_bin[[t]])
  }
  
  CIW_tmle_bin  <- lapply(1:(t.end-1), function(t) tmle_est_var_bin$CI[[t]][2,]- tmle_est_var_bin$CI[[t]][1,])
  names(CIW_tmle_bin) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CIW_tmle_bin[[t]]) <- names(bias_tmle_bin[[t]])
  }
  
  # gcomp metrics
  bias_gcomp  <- lapply(2:t.end, function(t) sapply(Y.true,"[[",t) - gcomp_est_var$est[[t]])
  names(bias_gcomp) <- paste0("t=",2:t.end)
  
  CP_gcomp <- lapply(1:(t.end-1), function(t) as.numeric((gcomp_est_var$CI[[t]][1,] < sapply(Y.true,"[[",t)) & (gcomp_est_var$CI[[t]][2,] > sapply(Y.true,"[[",t))))
  names(CP_gcomp) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CP_gcomp[[t]]) <- names(bias_gcomp[[t]])
  }
  
  CIW_gcomp  <- lapply(1:(t.end-1), function(t) gcomp_est_var$CI[[t]][2,]- gcomp_est_var$CI[[t]][1,])
  names(CIW_gcomp) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CIW_gcomp[[t]]) <- names(bias_gcomp[[t]])
  }
  
  # IPTW metrics
  bias_iptw  <- lapply(2:t.end, function(t) sapply(Y.true,"[[",t) - iptw_est_var$est[[t]])
  names(bias_iptw) <- paste0("t=",2:t.end)
  
  CP_iptw <- lapply(1:(t.end-1), function(t) as.numeric((iptw_est_var$CI[[t]][1,] < sapply(Y.true,"[[",t)) & (iptw_est_var$CI[[t]][2,] > sapply(Y.true,"[[",t))))
  names(CP_iptw) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CP_iptw[[t]]) <- names(bias_iptw[[t]])
  }
  
  CIW_iptw  <- lapply(1:(t.end-1), function(t) iptw_est_var$CI[[t]][2,]- iptw_est_var$CI[[t]][1,])
  names(CIW_iptw) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CIW_iptw[[t]]) <- names(bias_iptw[[t]])
  }
  
  # binomial version
  
  bias_iptw_bin  <- lapply(2:t.end, function(t) sapply(Y.true,"[[",t) - iptw_est_var_bin$est[[t]])
  names(bias_iptw_bin) <- paste0("t=",2:t.end)
  
  CP_iptw_bin <- lapply(1:(t.end-1), function(t) as.numeric((iptw_est_var_bin$CI[[t]][1,] < sapply(Y.true,"[[",t)) & (iptw_est_var_bin$CI[[t]][2,] > sapply(Y.true,"[[",t))))
  names(CP_iptw_bin) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CP_iptw_bin[[t]]) <- names(bias_iptw_bin[[t]])
  }
  
  CIW_iptw_bin  <- lapply(1:(t.end-1), function(t) iptw_est_var_bin$CI[[t]][2,]- iptw_est_var_bin$CI[[t]][1,])
  names(CIW_iptw_bin) <- paste0("t=",2:t.end)
  
  for(t in 1:(t.end-1)){
    names(CIW_iptw_bin[[t]]) <- names(bias_iptw_bin[[t]])
  }
  
  return(list("Ahat_tmle"=Ahat_tmle, "Chat_tmle"=Chat_tmle, "yhat_tmle"= tmle_estimates, "prob_share_tmle"= prob_share,
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
settings <- expand.grid("n"=c(12500), 
                        treatment.rule = c("static","dynamic","stochastic")) 

options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE) # command line arguments # args <- c('tmle-lstm',1,'TRUE','FALSE')
estimator <- as.character(args[1])
thisrun <- settings[as.numeric(args[2]),]
use.SL <- as.logical(args[3])  # When TRUE, use Super Learner for initial Y model and treatment model estimation; if FALSE, use GLM
doMPI <- as.logical(args[4])

# define parameters

n <- as.numeric(thisrun[,1]) # total sample size

treatment.rule <- ifelse(estimator=="tmle" | estimator=="tmle-lstm", "all", as.character(thisrun[,2])) # calculate counterfactual means under all treatment rules

J <- 6 # number of treatments

t.end <- 36 # number of time points after t=0

R <- 1#325 # number of simulation runs

# full_vector <- 1:R
# 
# # Specify the values to be omitted
# omit_values <- c(47, 18, 7, 17, 39, 93, 118, 77, 24, 14, 85, 72,
#                  101, 113, 51, 108, 81, 57, 80, 70, 64, 105, 96, 74,
#                  38, 73, 65, 122, 130, 134, 131, 132, 140, 139, 133,
#                  151, 152, 162, 160, 169, 176, 191, 183, 202, 207, 204,
#                  209, 223, 217, 237, 233, 236, 243, 244, 247, 255, 263,
#                  269, 282, 279, 294, 306, 311, 316, 324)
# 
# # Remove the specified values from the full vector
# final_vector <- full_vector[!full_vector %in% omit_values]

scale.continuous <- FALSE # standardize continuous covariates

gbound <- c(0.05,1) # define bounds to be used for the propensity score and censoring prob.

ybound <- c(0.0001,0.9999) # define bounds to be used for the Y predictions

n.folds <- 5

# Setup parallel processing
if(doMPI){
  library(doMPI)
  
  # Start cluster
  cl <- startMPIcluster()
  
  # Register cluster
  registerDoMPI(cl)
  
  # Check cluster size
  print(paste0("cluster size: ", clusterSize(cl)))
  
} else{
  library(foreach)
  library(parallel)
  library(doParallel)
  
  cores <- (parallel::detectCores())
  print(paste0("number of cores used: ", cores))
  
  if(estimator!='tmle-lstm'){
    cl <- parallel::makeCluster(cores, outfile="")
  
    doParallel::registerDoParallel(cl) # register cluster
}
}

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
                   "_R_", R,
                   "_n_", n,
                   "_J_", J,
                   "_n_folds_",n.folds,
                   "_scale_continuous_",scale.continuous,
                   "_use_SL_", use.SL,".rds")

#####################
# Run simulation #
#####################

print(paste0('simulation setting: ', "estimator = ", estimator, ", treatment.rule = ", treatment.rule, " R = ", R, ", n = ", n,", J = ", J ,", t.end = ", t.end, ", use.SL = ",use.SL, ", scale.continuous = ",scale.continuous))

library_vector <- c(
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
  "reticulate",
  "tensorflow",
  "keras",
  "parallel"
)

if(estimator=='tmle-lstm'){ # run sequentially
  sim.results <- foreach(r = 1:R, .combine='cbind', .errorhandling="pass", .packages=library_vector, .verbose = TRUE) %do% {
    simLong(r=r, J=J, n=n, t.end=t.end, gbound=gbound, ybound=ybound, n.folds=n.folds, cores=cores, estimator=estimator, treatment.rule=treatment.rule, use.SL=use.SL, scale.continuous=scale.continuous)
  }
}else{ # run in parallel
  sim.results <- foreach(r = 1:R, .combine='cbind', .errorhandling="pass", .packages=library_vector, .verbose = TRUE, .inorder=FALSE) %dopar% {
    simLong(r=r, J=J, n=n, t.end=t.end, gbound=gbound, ybound=ybound, n.folds=n.folds, cores=cores, estimator=estimator, treatment.rule=treatment.rule, use.SL=use.SL, scale.continuous=scale.continuous)
  }
}

saveRDS(sim.results, filename)

if(doMPI){
  closeCluster(cl) # close down MPIcluster
  mpi.finalize()
}else{
  if(estimator!='tmle-lstm'){
    stopCluster(cl)
  }
}