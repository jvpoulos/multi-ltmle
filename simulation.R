############################################################################################
# Longitudinal setting (T>1) simulations: Compare multinomial TMLE with binary TMLE        #
############################################################################################

######################
# Simulation function #
######################

simLong <- function(r, J=6, n=10000, t.end=36, gbound=c(0.025,1), ybound=c(0.0001,0.9999), n.folds=5, estimator="tmle", treatment.rule = "all", use.SL=TRUE){
  
  # libraries
  library(simcausal)
  library(dplyr)
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
  library(weights)
  library(gtools)
  
  if(estimator=='tmle-lstm'){
    library(tensorflow)
    library(keras)
    print(is_keras_available())
    print(tf_version())
  }
  
  if(estimator%in%c("tmle", "tmle-lstm", "lmtp-tmle","lmtp-iptw","lmtp-gcomp","lmtp-sdr")){
    source('./src/misc_fns.R')
    source('./src/tmle_fns.R')
    source('./src/SL3_fns.R', local =TRUE)
  }
  
  if(estimator%in%c("lmtp-tmle","lmtp-iptw","lmtp-gcomp","lmtp-sdr")){
    library(lmtp)
  }
  
  if(estimator%in%c("ltmle-tmle","ltmle-gcomp")){
    library(ltmle)
    library(SuperLearner)
    source('./src/SL_fns.R')
    source('./src/misc_fns.R')
    source('./src/ltmle_fns.R')
  }
  
  if(J!=6){
    stop("J must be 6")
  }
  
  if(t.end<4 && t.end >36){
    stop("t.end must be at least 4 and no more than 36")
  }
  
  if(t.end!=36 & estimator!="tmle"){
    stop("need to manually change t.end in shift functions in lmtp_fns.R or ltmle.R, and the number of lags in tmle_dat, and IC in tmle_fns")
  }
  
  if(n.folds<3){
    stop("n.folds needs to be greater than 3")
  }
  
  if(use.SL==FALSE){
    warning("not tested on use.SL=FALSE")
  }
  
  if(estimator%in%c("lmtp-tmle","lmtp-iptw","lmtp-gcomp","lmtp-sdr","ltmle-tmle","ltmle-gcomp","tmle-lstm")){
    warning("estimator not functional")
  }
  
  # define DGP
  source('./src/simcausal_fns.R')
  source('./src/simcausal_dgp.R', local =TRUE)
  
  # specify intervention rules (t=0 is same as observed)
  Dset <- set.DAG(D, vecfun=c("StochasticFun")) # locks DAG, consistency checks
  
  if(r==1){
    png(paste0(output_dir,"DAG_plot.png"))
    plotDAG(Dset, excludeattrs=c("C_0","Y_0"), xjitter=0.8, tmax = 3) # plot DAG
    dev.off()
  }
  
  int.static <-c(node("A", t = 0:t.end, distr = "rconst", # Static: Everyone gets quetiap (if bipolar=2), halo (if schizophrenia=3), ari (if MDD=1) and stays on it
                      const = ifelse(V2[0]==3, 2, ifelse(V2[0]==1, 1, 4))),
                 node("C", t = 1:t.end, distr = "rbern", prob = 0)) # under no censoring
  
  int.dynamic <- c(node("A", t = 0, distr = "rconst",   # Dynamic: Everyone starts with quetiap. # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to risp (if bipolar), halo. (if schizophrenia), ari (if MDD)
                        const= 4),
                   node("A", t = 1:t.end, distr = "rconst",
                        const=ifelse((L1[t] >0 | L2[t] >0 | L3[t] >0), ifelse(V2[0]==3, 2, ifelse(V2[0]==2, 4, 1)), 4)),
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
        #        ylim = c(0.5,1),
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_observed_",n, ".png")))
    plotSurvEst(surv = list("Static"=1-Y.observed[["static"]], "Dynamic"=1-Y.observed[["dynamic"]], "Stochastic"=1-Y.observed[["stochastic"]]),
                ylab = "Share of patients without diabetes diagnosis", 
                xlab = "Month",
                main = "Observed outcomes (simulated data)",
            #    ylim = c(0.5,1),
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
  
  if(estimator%in%c("tmle", "tmle-lstm")){
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
    tmle_dat[c("V3","L1","L1.lag","L1.lag2","L1.lag3")] <- scale(tmle_dat[c("V3","L1","L1.lag","L1.lag2","L1.lag3")]) # scale continuous variables
    
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
      tmle_dat[c("V3","L1")] <- scale(tmle_dat[c("V3","L1")]) # scale continuous variables
      
      tmle_dat[is.na(tmle_dat)] <- -10 # set NAs to -10 (add masking layer to LSTM)
      
      tmle_dat$A <- factor(tmle_dat$A)
      tmle_dat$V1 <- factor(tmle_dat$V1)
      tmle_dat$V2 <- factor(tmle_dat$V2)
      
      tmle_dat <- reshape(tmle_dat[c("t","ID", "V1", "V2", "V3", "L1", "L2","L3","A","Y","C")], idvar = "t", timevar = "ID", direction = "wide") # reshape wide so it is T x N
      
      tmle_covars_Y <- c(grep("L",colnames(tmle_dat),value = TRUE), grep("A",colnames(tmle_dat),value = TRUE), grep("V",colnames(tmle_dat),value = TRUE))  
      tmle_covars_A <- c(grep("L",colnames(tmle_dat),value = TRUE), grep("V",colnames(tmle_dat),value = TRUE))
      tmle_covars_C <- tmle_covars_A
    }
    
    ##  fit initial treatment model
    
    # multinomial
    
    initial_model_for_A_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                           learners = if(use.SL) learner_stack_A else make_learner(Lrnr_glm),
                                           metalearner = metalearner_A,
                                           keep_extra=FALSE)
    
    if(estimator=="tmle"){
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
    } else if(estimator=="tmle-lstm"){ 
      folds <- origami::make_folds(tmle_dat[-(grep("Y",colnames(tmle_dat)))], fold_fun=folds_rolling_window, window_size = ceiling(t.end*0.5), validation_size = ceiling(t.end*0.1), gap = 0, batch = 1) # define cross-validation appropriate for dependent data
      
      options('datatable.alloccol' = 270003)
      initial_model_for_A_task <- make_sl3_Task(tmle_dat[-(grep("Y",colnames(tmle_dat)))],
                                                covariates = tmle_covars_A,
                                                outcome = grep("A",colnames(tmle_dat),value = TRUE), 
                                                outcome_type="categorical", 
                                                folds = folds) 

      
      # train
      initial_model_for_A_sl_fit <- initial_model_for_A_sl$train(initial_model_for_A_task)
      
      initial_model_for_A <- list("preds"=initial_model_for_A_sl_fit$predict(initial_model_for_A_task),
                                  "folds"= folds,
                                  "task"=initial_model_for_A_task,
                                  "fit"=initial_model_for_A_sl_fit,
                                  "data"=tmle_dat)
    }
    
    g_preds <- lapply(1:length(initial_model_for_A), function(i) data.frame(matrix(unlist(lapply(initial_model_for_A[[i]]$preds, unlist)), nrow=length(lapply(initial_model_for_A[[i]]$preds, unlist)), byrow=TRUE)) ) # t length list of estimated propensity scores 
    g_preds <- lapply(1:length(initial_model_for_A), function(x) setNames(g_preds[[x]], grep("A[0-9]$",colnames(tmle_dat), value=TRUE)) )
    
    g_preds_ID <- lapply(1:length(initial_model_for_A), function(i) unlist(lapply(initial_model_for_A[[i]]$data$ID, unlist))) 
    
    g_preds_cuml <- vector("list", length(g_preds))
    
    g_preds_cuml[[1]] <- g_preds[[1]]
    
    for (i in 2:length(g_preds)) {
      g_preds_cuml[[i]] <- g_preds[[i]][which(g_preds_ID[[i-1]]%in%g_preds_ID[[i]]),] * g_preds_cuml[[i-1]][which(g_preds_ID[[i-1]]%in%g_preds_ID[[i]]),]
    }
    
    g_preds_cuml_bounded <- lapply(1:length(initial_model_for_A), function(x) boundProbs(g_preds_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores                             
    
    # binomial
    
    initial_model_for_A_sl_bin <- make_learner(Lrnr_sl, # cross-validates base models
                                               learners = if(use.SL) learner_stack_A_bin else make_learner(Lrnr_glm),
                                               metalearner = metalearner_A_bin,
                                               keep_extra=FALSE)
    
    if(estimator=="tmle"){
      
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
    } else if(estimator=="tmle-lstm"){ 
      initial_model_for_A_task_bin <- lapply(1:J, function(j) make_sl3_Task(cbind("A"=dummify(tmle_dat$A)[,j],tmle_dat[tmle_covars_A]), 
                                                                            covariates = tmle_covars_A, 
                                                                            outcome = "A",
                                                                            outcome_type="binomial",
                                                                            folds = folds)) 
      # train
      
      initial_model_for_A_sl_fit_bin <- lapply(1:J, function(j) initial_model_for_A_sl_bin$train(initial_model_for_A_task_bin[[j]]))
      
      initial_model_for_A_sl_bin <- list("preds"=sapply(1:J, function(j) initial_model_for_A_sl_fit_bin[[j]]$predict(initial_model_for_A_task_bin[[j]])), 
                                         "folds"= folds,
                                         "task"=initial_model_for_A_task_bin,
                                         "fit"=initial_model_for_A_sl_fit_bin,
                                         "data"=tmle_dat)
    }
    
    g_preds_bin <- lapply(1:length(initial_model_for_A_bin), function(i) data.frame(initial_model_for_A_bin[[i]]$preds) ) # t length list of estimated propensity scores 
    g_preds_bin <- lapply(1:length(initial_model_for_A_bin), function(x) setNames(g_preds_bin[[x]], grep("A[0-9]$",colnames(tmle_dat), value=TRUE)) )
    
    g_preds_bin_ID <- lapply(1:length(initial_model_for_A_bin), function(i) unlist(lapply(initial_model_for_A_bin[[i]]$data$ID, unlist)))
    
    g_preds_bin_cuml <- vector("list", length(g_preds_bin))
    
    g_preds_bin_cuml[[1]] <- g_preds_bin[[1]]
    
    for (i in 2:length(g_preds_bin)) {
      g_preds_bin_cuml[[i]] <- g_preds_bin[[i]][which(g_preds_bin_ID[[i-1]]%in%g_preds_bin_ID[[i]]),] * g_preds_bin_cuml[[i-1]][which(g_preds_bin_ID[[i-1]]%in%g_preds_bin_ID[[i]]),]
    }    
    g_preds_bin_cuml_bounded <- lapply(1:length(initial_model_for_A), function(x) boundProbs(g_preds_bin_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores                             
    
    ##  fit initial censoring model
    ## implicitly fit on those that are uncensored until t-1
    
    initial_model_for_C_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                           learners = if(use.SL) learner_stack_A_bin else make_learner(Lrnr_glm),
                                           metalearner = metalearner_A_bin,
                                           keep_extra=FALSE)
    
    if(estimator=="tmle"){
      
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
    }else if(estimator=="tmle-lstm"){
      
      # define task and candidate learners
      initial_model_for_C_task <- make_sl3_Task(data=tmle_dat,
                                                covariates = tmle_covars_C, 
                                                outcome = "C",
                                                outcome_type="binomial", 
                                                folds = folds) 
      
      # train
      
      initial_model_for_C_sl_fit <- initial_model_for_C_sl$train(initial_model_for_C_task)
      
      initial_model_for_C <- list("preds"=initial_model_for_C_sl_fit$predict(initial_model_for_C_task),
                                  "folds"= folds,
                                  "task"=initial_model_for_C_task,
                                  "fit"=initial_model_for_C_sl_fit,
                                  "data"=tmle_dat)
    }
    
    C_preds <- lapply(1:length(initial_model_for_C), function(i) 1-initial_model_for_C[[i]]$preds) # t length list # C=1 if uncensored; C=0 if censored  
    
    C_preds_ID <- lapply(1:length(initial_model_for_C), function(i) unlist(lapply(initial_model_for_C[[i]]$data$ID, unlist))) 
    
    C_preds_cuml <- vector("list", length(C_preds))
    
    C_preds_cuml[[1]] <- C_preds[[1]]
    
    for (i in 2:length(C_preds)) {
      C_preds_cuml[[i]] <- C_preds[[i]][which(C_preds_ID[[i-1]]%in%C_preds_ID[[i]])] * C_preds_cuml[[i-1]][which(C_preds_ID[[i-1]]%in%C_preds_ID[[i]])]
    }    
    C_preds_cuml_bounded <- lapply(1:length(initial_model_for_C), function(x) boundProbs(C_preds_cuml[[x]],bounds=ybound))  # winsorized cumulative bounded censoring predictions, 1=Censored                            
    
    ## sequential g-formula
    ## model is fit on all uncensored and alive (until t-1)
    ## the outcome is the observed Y for t=T and updated Y if t<T
    
    initial_model_for_Y_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                           learners = if(use.SL) learner_stack_Y else make_learner(Lrnr_glm),
                                           metalearner = metalearner_Y,
                                           keep_extra=FALSE)
    
    if(estimator=="tmle"){
      
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
      tmle_dat_sub <- tmle_dat[!is.na(tmle_dat$Y),] # drop rows with missing Y
      
      # define task and candidate learners
      initial_model_for_Y_task <- make_sl3_Task(data=tmle_dat_sub,
                                                covariates = tmle_covars_Y, 
                                                outcome = "Y",
                                                outcome_type="binomial", 
                                                folds = folds) 
      # train
      initial_model_for_Y_sl_fit <- initial_model_for_Y_sl$train(initial_model_for_Y_task)
      
      # predict on everyone
      Y_preds <- initial_model_for_Y_sl_fit$predict(sl3_Task$new(data=tmle_dat, covariates = tmle_covars_Y, outcome="Y", outcome_type="binomial"))
      
      initial_model_for_Y <- list("preds"=boundProbs(Y_preds,ybound),
                                  "fit"=initial_model_for_Y_sl_fit,
                                  "data"=tmle_dat) # evaluation data (fit on everyone)
    }
    
    # plot estimated survival curves
    
    tmle_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar[[x]]))) # static, dynamic, stochastic
    tmle_bin_estimates <-  cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t]][,x]$Qstar[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t.end]]$Qstar[[x]]))) 
    
    if(r==1){
      png(paste0(output_dir,paste0("survival_plot_tmle_estimates_",n, ".png")))
      plotSurvEst(surv = list("Static"= tmle_estimates[1,], "Dynamic"= tmle_estimates[2,], "Stochastic"= tmle_estimates[3,]),  
                  ylab = "Estimated share of patients without diabetes diagnosis", 
                  main = "LTMLE (ours, multinomial) estimated counterfactuals",
                  xlab = "Month",
                  ylim = c(0.5,1),
                  legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
      axis(1, at = seq(1, t.end, by = 5))
      dev.off()
      
      png(paste0(output_dir,paste0("survival_plot_tmle_estimates_bin_",n, ".png")))
      plotSurvEst(surv = list("Static"= tmle_bin_estimates[1,], "Dynamic"= tmle_bin_estimates[2,], "Stochastic"= tmle_bin_estimates[3,]),  
                  ylab = "Estimated share of patients without diabetes diagnosis", 
                  main = "LTMLE (ours, binomial) estimated counterfactuals",
                  xlab = "Month",
                  ylim = c(0.5,1),
                  legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
      axis(1, at = seq(1, t.end, by = 5))
      dev.off()
    }
    
    iptw_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar_iptw[[x]]))) # static, dynamic, stochastic
    iptw_bin_estimates <-  cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t]][,x]$Qstar_iptw[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t.end]]$Qstar_iptw[[x]]))) 
    
    if(r==1){
      png(paste0(output_dir,paste0("survival_plot_iptw_estimates_",n, ".png")))
      plotSurvEst(surv = list("Static"= iptw_estimates[1,], "Dynamic"= iptw_estimates[2,], "Stochastic"= iptw_estimates[3,]),  
                  ylab = "Estimated share of patients without diabetes diagnosis", 
                  main = "IPTW (ours, multinomial) estimated counterfactuals",
                  xlab = "Month",
                  ylim = c(0.5,1),
                  legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
      axis(1, at = seq(1, t.end, by = 5))
      dev.off()
      
      png(paste0(output_dir,paste0("survival_plot_iptw_bin_estimates_",n, ".png")))
      plotSurvEst(surv = list("Static"= iptw_bin_estimates[1,], "Dynamic"= iptw_bin_estimates[2,], "Stochastic"= iptw_bin_estimates[3,]),  
                  ylab = "Estimated share of patients without diabetes diagnosis", 
                  main = "IPTW (ours, binomial) estimated counterfactuals",
                  xlab = "Month",
                  ylim = c(0.5,1))
      dev.off()
    }
    
    gcomp_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar_gcomp[[x]]))) # static, dynamic, stochastic
    
    if(r==1){
      png(paste0(output_dir,paste0("survival_plot_gcomp_estimates_",n, ".png")))
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
  }
  
  ## LMTP
  
  lmtp_tmle_results <- list()
  lmtp_iptw_results <- list()
  lmtp_gcomp_results <- list()
  lmtp_sdr_results <- list()
  
  results_bias_lmtp_tmle <- list() 
  results_CP_lmtp_tmle <- list()
  results_CIW_lmtp_tmle <- list()
  
  results_bias_lmtp_iptw <- list() 
  results_CP_lmtp_iptw <- list()
  results_CIW_lmtp_iptw <- list()
  
  results_bias_lmtp_gcomp <- list() 
  results_CP_lmtp_gcomp <- list()
  results_CIW_lmtp_gcomp <- list()
  
  results_bias_lmtp_sdr <- list() 
  results_CP_lmtp_sdr <- list()
  results_CIW_lmtp_sdr <- list()
  
  if(estimator%in%c("lmtp-tmle", "lmtp-iptw","lmtp-gcomp","lmtp-sdr")){
    
    # define treatment and covariates
    baseline <- c("V1_0", "V2_0", "V3_0")
    tv <- lapply(0:t.end, function(i){paste0("L",c(1,2,3),"_",i)})
    
    lmtp_dat <- Odat[!colnames(Odat)%in%c("ID")] # columns in time-ordering of model: A < C < Y
    lmtp_dat[,cnodes][is.na(lmtp_dat[,cnodes])] <- 1
    lmtp_dat[,cnodes] <- ifelse(lmtp_dat[,cnodes]==1, 0, 1) # C=0 indicates censored
    
    lmtp_dat[c("V3_0",grep("L1",colnames(lmtp_dat), value=TRUE))] <- scale(lmtp_dat[c("V3_0",grep("L1",colnames(lmtp_dat), value=TRUE))] ) # center and scale continuous/count vars
    
    # define treatment rules
    
    lmtp_rules <- list("static"=static_mtp,
                       "dynamic"=dynamic_mtp,
                       "stochastic"=stochastic_mtp) 
    
    # estimate outcomes under treatment rule, for all periods
    
    if(estimator=="lmtp-tmle"){
      lmtp_tmle_results[[treatment.rule]] <- lmtp_tmle(data = lmtp_dat,
                                                       trt = anodes,
                                                       outcome = ynodes, 
                                                       baseline = baseline, 
                                                       time_vary = tv,
                                                       cens= cnodes,
                                                       shift=lmtp_rules[[treatment.rule]],
                                                       intervention_type = "mtp",
                                                       outcome_type = "survival", 
                                                       learners_trt= if(use.SL) learner_stack_A else make_learner(Lrnr_glm),
                                                       learners_outcome= if(use.SL) learner_stack_Y else make_learner(Lrnr_glm),
                                                       .SL_folds = n.folds)
      
      results_bias_lmtp_tmle[[treatment.rule]] <- lmtp_tmle_results[[treatment.rule]]$theta - Y.true[[treatment.rule]][t.end] # LMTP  point and variance estimates
      results_CP_lmtp_tmle[[treatment.rule]] <- as.numeric((lmtp_tmle_results[[treatment.rule]]$low < Y.true[[treatment.rule]][t.end]) & (lmtp_tmle_results[[treatment.rule]]$high > Y.true[[treatment.rule]][t.end]))
      results_CIW_lmtp_tmle[[treatment.rule]]<- lmtp_tmle_results[[treatment.rule]]$high-lmtp_tmle_results[[treatment.rule]]$low
    } else if(estimator=="lmtp-iptw"){
      lmtp_iptw_results[[treatment.rule]] <- lmtp_ipw(data = lmtp_dat,
                                                      trt = anodes,
                                                      outcome = ynodes, 
                                                      baseline = baseline, 
                                                      time_vary = tv,
                                                      cens= cnodes,
                                                      shift=lmtp_rules[[treatment.rule]],
                                                      intervention_type = "mtp",
                                                      outcome_type = "survival", 
                                                      learners= if(use.SL) learner_stack_A else make_learner(Lrnr_glm),
                                                      .SL_folds = n.folds)
      
      results_bias_lmtp_iptw[[treatment.rule]] <- lmtp_iptw_results[[treatment.rule]]$theta - Y.true[[treatment.rule]][t.end] # LMTP  point and variance estimates
      results_CP_lmtp_iptw[[treatment.rule]] <- as.numeric((lmtp_iptw_results[[treatment.rule]]$low < Y.true[[treatment.rule]][t.end]) & (lmtp_iptw_results[[treatment.rule]]$high > Y.true[[treatment.rule]][t.end]))
      results_CIW_lmtp_iptw[[treatment.rule]]<- lmtp_iptw_results[[treatment.rule]]$high-lmtp_iptw_results[[treatment.rule]]$low
    } else if(estimator=="lmtp-gcomp"){
      lmtp_gcomp_results[[treatment.rule]] <- lmtp_sub(data = lmtp_dat,
                                                       trt = anodes,
                                                       outcome = ynodes, 
                                                       baseline = baseline, 
                                                       time_vary = tv,
                                                       cens= cnodes,
                                                       shift=lmtp_rules[[treatment.rule]],
                                                       outcome_type = "survival", 
                                                       learners= if(use.SL) learner_stack_Y else make_learner(Lrnr_glm),
                                                       .SL_folds = n.folds)
      
      results_bias_lmtp_gcomp[[treatment.rule]] <- lmtp_gcomp_results[[treatment.rule]]$theta - Y.true[[treatment.rule]][t.end] # LMTP  point and variance estimates
      results_CP_lmtp_gcomp[[treatment.rule]] <- as.numeric((lmtp_gcomp_results[[treatment.rule]]$low < Y.true[[treatment.rule]][t.end]) & (lmtp_gcomp_results[[treatment.rule]]$high > Y.true[[treatment.rule]][t.end]))
      results_CIW_lmtp_gcomp[[treatment.rule]]<- lmtp_gcomp_results[[treatment.rule]]$high-lmtp_gcomp_results[[treatment.rule]]$low
    } else if(estimator=="lmtp-sdr"){
      lmtp_sdr_results[[treatment.rule]] <- lmtp_sdr(data = lmtp_dat,
                                                     trt = anodes,
                                                     outcome = ynodes, 
                                                     baseline = baseline, 
                                                     time_vary = tv,
                                                     cens= cnodes,
                                                     shift=lmtp_rules[[treatment.rule]],
                                                     intervention_type = "mtp",
                                                     outcome_type = "survival", 
                                                     learners_trt= if(use.SL) learner_stack_A else make_learner(Lrnr_glm),
                                                     learners_outcome= if(use.SL) learner_stack_Y else make_learner(Lrnr_glm),
                                                     .SL_folds = n.folds)
      # store results
      
      results_bias_lmtp_sdr[[treatment.rule]] <- lmtp_sdr_results[[treatment.rule]]$theta - Y.true[[treatment.rule]][t.end] # LMTP  point and variance estimates
      results_CP_lmtp_sdr[[treatment.rule]] <- as.numeric((lmtp_sdr_results[[treatment.rule]]$low < Y.true[[treatment.rule]][t.end]) & (lmtp_sdr_results[[treatment.rule]]$high > Y.true[[treatment.rule]][t.end]))
      results_CIW_lmtp_sdr[[treatment.rule]]<- lmtp_sdr_results[[treatment.rule]]$high-lmtp_sdr_results[[treatment.rule]]$low
    }
  }
  
  ## LTMLE
  
  ltmle_tmle_results <- list()
  ltmle_iptw_results <- list()
  ltmle_gcomp_results <- list()
  
  results_CI_ltmle_tmle <- list() 
  results_bias_ltmle_tmle <- list() 
  results_CP_ltmle_tmle <- list()
  results_CIW_ltmle_tmle <- list()
  
  results_CI_ltmle_iptw <- list() 
  results_bias_ltmle_iptw <- list() 
  results_CP_ltmle_iptw <- list()
  results_CIW_ltmle_iptw <- list()
  
  results_CI_ltmle_gcomp <- list() 
  results_bias_ltmle_gcomp <- list() 
  results_CP_ltmle_gcomp <- list()
  results_CIW_ltmle_gcomp <- list()
  
  if(estimator%in%c("ltmle-tmle","ltmle-gcomp")){
    
    # define treatments
    obs.treatment[obs.treatment==0] <- NA
    obs.treatment <- droplevels(obs.treatment)
    treatments <- lapply(1:(t.end+1), function(t) dummify(obs.treatment[,t]))
    treatments <- do.call("cbind", treatments)
    colnames(treatments) <- c(sapply(0:t.end, function(t) paste0(c("A1_","A2_","A3_","A4_","A5_","A6_"), rep(t,3))))
    
    # define treatment and covariates 
    
    baseline <- c("V1_0", "V2_0", "V3_0")
    tv <- c(sapply(0:t.end, function(i){paste0("L",c(1,2,3),"_",i)}))
    
    ltmle_dat <- data.frame(Odat[baseline],Odat[tv],treatments,
                            sapply(1:length(cnodes), function(t) BinaryToCensoring(is.censored=Odat[,cnodes[t]])),  # converts to 1=uncensored, 0=censored
                            Odat[ynodes], stringsAsFactors=TRUE)
    colnames(ltmle_dat)[grep("X",colnames(ltmle_dat))] <- colnames(Odat[cnodes])
    
    ltmle_dat[c("V3_0",grep("L1",colnames(ltmle_dat), value=TRUE))] <- scale(ltmle_dat[c("V3_0",grep("L1",colnames(ltmle_dat), value=TRUE))]) # center and scale continuous/count vars
    
    column_order <- c(baseline, c(sapply(0:t.end, function(i){
      c(paste0("L",c(1,2,3),"_",i), paste0("A",c(1:6),"_",i), paste0("C","_",i), paste0("Y","_",i))
    })))
    
    ltmle_dat <-  ltmle_dat[mixedorder(names(ltmle_dat))][column_order] # columns in time-ordering of model: A < C < Y
    
    # define treatment rules
    
    ltmle_rules <- list("static"=static_mtp,
                        "dynamic"=dynamic_mtp,
                        "stochastic"=stochastic_mtp) 
    
    # estimate outcomes under treatment regime
    
    if(estimator=="ltmle-tmle"){
      ltmle_tmle_results[[treatment.rule]] <- ltmle(ltmle_dat, 
                                                    Anodes=colnames(treatments),
                                                    Cnodes=cnodes,
                                                    Lnodes=tv,
                                                    Ynodes=ynodes, 
                                                    survivalOutcome = TRUE,
                                                    rule=ltmle_rules[[treatment.rule]],
                                                    stratify=FALSE, 
                                                    variance.method="ic",
                                                    estimate.time=TRUE, 
                                                    SL.library=if(use.SL) SL.library else "glm",
                                                    SL.cvControl=if(use.SL) list(V = n.folds) else list(),
                                                    gbounds=gbound)
      
      results_bias_ltmle_tmle[[treatment.rule]] <- ltmle_tmle_results[[treatment.rule]]$estimates['tmle'] - Y.true[[treatment.rule]][t.end] # ltmle  point and variance estimates
      results_CI_ltmle_tmle[[treatment.rule]] <- CI(est=ltmle_tmle_results[[treatment.rule]]$estimates['tmle'], infcurv = ltmle_tmle_results[[treatment.rule]]$IC$tmle, alpha=0.05)
      
      results_CP_ltmle_tmle[[treatment.rule]] <- as.numeric((results_CI_ltmle_tmle[[treatment.rule]][[1]] < sapply(Y.true,"[[",t.end)[[treatment.rule]])  & (results_CI_ltmle_tmle[[treatment.rule]][[2]] > sapply(Y.true,"[[",t.end)[[treatment.rule]])) 
      results_CIW_ltmle_tmle[[treatment.rule]] <- results_CI_ltmle_tmle[[treatment.rule]][[2]] - results_CI_ltmle_tmle[[treatment.rule]][[1]]  # wrt to est at T
      
      results_bias_ltmle_iptw[[treatment.rule]] <- ltmle_iptw_results[[treatment.rule]]$estimates['iptw'] - Y.true[[treatment.rule]][t.end] # ltmle  point and variance estimates
      results_CI_ltmle_iptw[[treatment.rule]] <- CI(est=ltmle_iptw_results[[treatment.rule]]$estimates['iptw'], infcurv = ltmle_iptw_results[[treatment.rule]]$IC$iptw, alpha=0.05)
      
      results_CP_ltmle_iptw[[treatment.rule]] <- as.numeric((results_CI_ltmle_iptw[[treatment.rule]][[1]] < sapply(Y.true,"[[",t.end)[[treatment.rule]])  & (results_CI_ltmle_iptw[[treatment.rule]][[2]] > sapply(Y.true,"[[",t.end)[[treatment.rule]])) 
      results_CIW_ltmle_iptw[[treatment.rule]] <- results_CI_ltmle_iptw[[treatment.rule]][[2]] - results_CI_ltmle_iptw[[treatment.rule]][[1]]  # wrt to est at T
      
    }else if(estimator=="ltmle-gcomp"){
      ltmle_gcomp_results[[treatment.rule]] <- ltmle(ltmle_dat, 
                                                     Anodes=colnames(treatments),
                                                     Cnodes=cnodes,
                                                     Lnodes=tv,
                                                     Ynodes=ynodes, 
                                                     survivalOutcome = TRUE,
                                                     rule=ltmle_rules[[treatment.rule]],
                                                     stratify=FALSE, 
                                                     variance.method="ic",
                                                     estimate.time=TRUE, 
                                                     gcomp=TRUE,
                                                     SL.library=if(use.SL) SL.library else "glm",
                                                     SL.cvControl=if(use.SL) list(V = n.folds) else list(),
                                                     gbounds=gbound)
      
      results_bias_ltmle_gcomp[[treatment.rule]] <- ltmle_gcomp_results[[treatment.rule]]$estimates['gcomp'] - Y.true[[treatment.rule]][t.end] # ltmle  point and variance estimates
      results_CI_ltmle_gcomp[[treatment.rule]] <- CI(est=ltmle_gcomp_results[[treatment.rule]]$estimates['gcomp'], infcurv = ltmle_gcomp_results[[treatment.rule]]$IC$gcomp, alpha=0.05)
      
      results_CP_ltmle_gcomp[[treatment.rule]] <- as.numeric((results_CI_ltmle_gcomp[[treatment.rule]][[1]] < sapply(Y.true,"[[",t.end)[[treatment.rule]])  & (results_CI_ltmle_gcomp[[treatment.rule]][[2]] > sapply(Y.true,"[[",t.end)[[treatment.rule]])) 
      results_CIW_ltmle_gcomp[[treatment.rule]] <- results_CI_ltmle_gcomp[[treatment.rule]][[2]] - results_CI_ltmle_gcomp[[treatment.rule]][[1]]  # wrt to est at T
    }
  }
  
  return(list("Ahat_tmle"=Ahat_tmle, "Chat_tmle"=Chat_tmle, "yhat_tmle"= tmle_estimates, "prob_share_tmle"= prob_share,
              "Ahat_tmle_bin"=Ahat_tmle_bin,"yhat_tmle_bin"= tmle_bin_estimates, "prob_share_tmle_bin"= prob_share_bin,
              "bias_tmle"= bias_tmle,"CP_tmle"= CP_tmle,"CIW_tmle"=CIW_tmle,"tmle_est_var"=tmle_est_var,
              "bias_tmle_bin"= bias_tmle_bin,"CP_tmle_bin"=CP_tmle_bin,"CIW_tmle_bin"=CIW_tmle_bin,"tmle_est_var_bin"=tmle_est_var_bin,
              "yhat_gcomp"= gcomp_estimates, "bias_gcomp"= bias_gcomp,"CP_gcomp"= CP_gcomp,"CIW_gcomp"=CIW_gcomp,"gcomp_est_var"=gcomp_est_var,
              "yhat_iptw"= iptw_estimates,"bias_iptw"= bias_iptw,"CP_iptw"= CP_iptw,"CIW_iptw"=CIW_iptw,"iptw_est_var"=iptw_est_var,
              "yhat_iptw_bin"= iptw_bin_estimates,"bias_iptw_bin"= bias_iptw_bin,"CP_iptw_bin"=CP_iptw_bin,"CIW_iptw_bin"=CIW_iptw_bin,"iptw_est_var_bin"=iptw_est_var_bin,
              "lmtp_tmle_results"=lmtp_tmle_results[[treatment.rule]],"bias_lmtp_tmle"=results_bias_lmtp_tmle,"CP_lmtp_tmle"=results_CP_lmtp_tmle,"CIW_lmtp_tmle"=results_CIW_lmtp_tmle,
              "lmtp_iptw_results"=lmtp_iptw_results[[treatment.rule]],"bias_lmtp_iptw"=results_bias_lmtp_iptw,"CP_lmtp_iptw"=results_CP_lmtp_iptw,"CIW_lmtp_iptw"=results_CIW_lmtp_iptw,
              "lmtp_gcomp_results"=lmtp_gcomp_results[[treatment.rule]],"bias_lmtp_gcomp"=results_bias_lmtp_gcomp,"CP_lmtp_gcomp"=results_CP_lmtp_gcomp,"CIW_lmtp_gcomp"=results_CIW_lmtp_gcomp,
              "lmtp_sdr_results"=lmtp_sdr_results[[treatment.rule]],"bias_lmtp_sdr"=results_bias_lmtp_sdr,"CP_lmtp_sdr"=results_CP_lmtp_sdr,"CIW_lmtp_sdr"=results_CIW_lmtp_sdr,
              "ltmle_tmle_results"=ltmle_tmle_results[[treatment.rule]],"CI_ltmle_tmle"=results_CI_ltmle_tmle,"bias_ltmle_tmle"=results_bias_ltmle_tmle, "CP_ltmle_tmle"=results_CP_ltmle_tmle,"CIW_ltmle_tmle"=results_CIW_ltmle_tmle,
              "ltmle_iptw_results"=ltmle_iptw_results[[treatment.rule]],"CI_ltmle_iptw"=results_CI_ltmle_iptw,"bias_ltmle_iptw"=results_bias_ltmle_iptw, "CP_ltmle_iptw"=results_CP_ltmle_iptw,"CIW_ltmle_iptw"=results_CIW_ltmle_iptw,
              "ltmle_gcomp_results"=ltmle_gcomp_results[[treatment.rule]],"CI_ltmle_gcomp"=results_CI_ltmle_gcomp,"bias_ltmle_gcomp"=results_bias_ltmle_gcomp, "CP_ltmle_gcomp"=results_CP_ltmle_gcomp,"CIW_ltmle_gcomp"=results_CIW_ltmle_gcomp))
}

#####################
# Set parameters    #
#####################

# define settings for simulation
settings <- expand.grid("n"=c(10000), 
                        treatment.rule = c("static","dynamic","stochastic")) 

options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE) # command line arguments
# args <- c('tmle-lstm',1,'TRUE','FALSE')
estimator <- as.character(args[1])
thisrun <- settings[as.numeric(args[2]),]
use.SL <- as.logical(args[3])  # When TRUE, use Super Learner for initial Y model and treatment model estimation; if FALSE, use GLM
doMPI <- as.logical(args[4])

# define parameters

n <- as.numeric(thisrun[,1]) # total sample size

treatment.rule <- ifelse(estimator=="tmle", "all", as.character(thisrun[,2])) # tmle calculates calculates counterfactual means under all treatment rules

J <- 6 # number of treatments

t.end <- 36 # number of time points after t=0

R <- 40 # number of simulation runs

gbound <- c(0.025,1) # define bounds to be used for the propensity score

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
  library(parallel)
  library(doParallel)
  library(foreach)
  
  cores <- parallel::detectCores()
  print(paste0("number of cores used: ", cores))
  
  cl <- parallel::makeCluster(cores, outfile="")
  
  doParallel::registerDoParallel(cl) # register cluster
}

# output directory
output_dir <- './outputs/'
simulation_version <- paste0(format(Sys.time(), "%Y%m%d"),"/")
if(!dir.exists(output_dir)){
  print(paste0('create folder for outputs at: ', output_dir))
  
  dir.create(output_dir)
}
output_dir <- paste0(output_dir, simulation_version)
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
                   "_use_SL_", use.SL,".rds")

#####################
# Run simulation #
#####################

print(paste0('simulation setting: ', "estimator = ", estimator, "treatment.rule = ", treatment.rule, " R = ", R, ", n = ", n,", J = ", J ,", t.end = ", t.end, ", use.SL = ",use.SL))

sim.results <- foreach(r = 1:R, .combine='cbind', .verbose = TRUE, .errorhandling="pass") %dopar% {
  simLong(r=r, J=J, n=n, t.end=t.end, gbound=gbound, ybound=ybound, n.folds=n.folds, estimator=estimator, treatment.rule=treatment.rule, use.SL=use.SL)
}
sim.results

saveRDS(sim.results, filename)

if(doMPI){
  closeCluster(cl) # close down MPIcluster
  mpi.finalize()
}else{
  stopCluster(cl)
}