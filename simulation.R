############################################################################################
# Longitudinal setting (T>1) simulations: Compare multinomial TMLE with binary TMLE        #
############################################################################################

######################
# Simulation function #
######################

simLong <- function(r, J=6, n=12500, t.end=36, gbound=c(0.05,1), ybound=c(0.0001,0.9999), n.folds=5, cores=1, estimator="tmle", treatment.rule = "all", use.SL=TRUE, scale.continuous=FALSE, debug =TRUE, window_size=7){
  
  # libraries
  library(simcausal)
  options(simcausal.verbose=FALSE)
  library(purrr)
  library(origami)
  library(sl3)
  options(sl3.verbose = FALSE)
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
  library(latex2exp)
  
  if(estimator=='tmle-lstm'){
    library(reticulate)
    use_python("/media/jason/Dropbox/github/multi-ltmle/myenv/bin/python", required = TRUE)
    print(py_config()) # Check Python configuration
    
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
  
  if(estimator=="tmle_lstm"){
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
  Y.observed[["static"]] <- sapply(1:t.end, function(t) mean(Odat[,ynodes][,paste0("Y_",t)][which(obs.rules[[t+1]][,"static"]==1)], na.rm=TRUE))
  Y.observed[["dynamic"]] <- sapply(1:t.end, function(t) mean(Odat[,ynodes][,paste0("Y_",t)][which(obs.rules[[t+1]][,"dynamic"]==1)], na.rm=TRUE))
  Y.observed[["stochastic"]] <- sapply(1:t.end, function(t) mean(Odat[,ynodes][,paste0("Y_",t)][which(obs.rules[[t+1]][,"stochastic"]==1)], na.rm=TRUE))
  Y.observed[["overall"]] <- sapply(1:t.end, function(t) mean(Odat[,ynodes][,paste0("Y_",t)], na.rm=TRUE))
  
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

    # Safe reshaping function
    safe_reshape_data <- function(data, t_end, debug) {
      if(debug){
        # Debug output
        print("Data column names at start:")
        print(names(data))
      }

      # Get unique IDs
      unique_ids <- unique(data$ID)
      n_ids <- length(unique_ids)

      # Ensure A column exists and is properly formatted
      if(!"A" %in% names(data)) {
        print("Looking for alternative treatment columns...")
        A_cols <- grep("^A[0-9]$", names(data), value=TRUE)
        if(length(A_cols) > 0) {
          print(paste("Found treatment columns:", paste(A_cols, collapse=", ")))
          # Use first treatment column if multiple exist
          data$A <- data[[A_cols[1]]]
        } else {
          stop("No treatment column found in data")
        }
      }

      # Convert treatment to factor if not already
      data$A <- factor(data$A)

      # Get unique IDs
      unique_ids <- unique(data$ID)
      n_ids <- length(unique_ids)

      # Create time sequence
      time_seq <- 0:t_end

      # Create expanded grid of IDs and times
      wide_base <- expand.grid(
        ID = unique_ids,
        t = time_seq,
        stringsAsFactors = FALSE
      )

      # Sort by ID and time
      wide_base <- wide_base[order(wide_base$ID, wide_base$t), ]

      # Merge with original data
      wide_data <- merge(wide_base, data, by = c("ID", "t"), all.x = TRUE)

      # Get time-varying columns
      time_varying_cols <- c("L1", "L2", "L3", "A", "C", "Y")

      # Reshape each time-varying variable
      wide_reshaped <- wide_data[!duplicated(wide_data$ID), setdiff(names(wide_data), c("t", time_varying_cols))]

      for(col in time_varying_cols) {
        if(col %in% names(wide_data)) {
          temp_data <- wide_data[, c("ID", "t", col)]
          temp_wide <- reshape(temp_data,
                               timevar = "t",
                               idvar = "ID",
                               direction = "wide",
                               sep = ".")

          # Add reshaped columns
          wide_cols <- grep(paste0("^", col, "\\."), names(temp_wide), value = TRUE)
          if(length(wide_cols) > 0) {
            wide_reshaped[wide_cols] <- temp_wide[, wide_cols]
          }
        }
      }

      # Handle missing values
      na_cols <- c(
        grep("^Y\\.", names(wide_reshaped), value = TRUE),
        grep("^C\\.", names(wide_reshaped), value = TRUE),
        grep("^L[1-3]\\.", names(wide_reshaped), value = TRUE),
        grep("^V3\\.", names(wide_reshaped), value = TRUE)
      )

      wide_reshaped[na_cols][is.na(wide_reshaped[na_cols])] <- -1

      # Handle treatment columns
      A_cols <- grep("^A\\.", names(wide_reshaped), value = TRUE)
      if(length(A_cols) > 0) {
        wide_reshaped[A_cols] <- lapply(wide_reshaped[A_cols], function(x) {
          if(is.factor(x)) {
            x <- addNA(x)
            levels(x) <- c("0", levels(x)[-length(levels(x))])
          } else {
            x <- factor(x)
            x <- addNA(x)
            levels(x) <- c("0", levels(x)[-length(levels(x))])
          }
          as.numeric(as.character(x))
        })
      }

      return(wide_reshaped)
    }

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

    g_preds_cuml <- vector("list", length(g_preds_bin))
    g_preds_cuml[[1]] <- g_preds_bin[[1]]
    
    for (i in 2:length(g_preds)) {
      # CORRECT SCALING: Scale to [0.5, 1] range to prevent underflow
      g_preds_cuml[[i]] <- g_preds[[i]][which(g_preds_ID[[i-1]]%in%g_preds_ID[[i]]),] * 
        (0.5 + (g_preds_cuml[[i-1]][which(g_preds_ID[[i-1]]%in%g_preds_ID[[i]]),] / 2))
    }

    g_preds_cuml_bounded <- lapply(1:length(initial_model_for_A), function(x) boundProbs(g_preds_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores

  } else if(estimator=="tmle-lstm"){
    window.size <- window_size

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

    # Modified window_predictions function
    window_predictions <- function(preds, window_size, n_ids) {
      if(is.null(preds) || length(preds) == 0) {
        return(replicate(window_size, matrix(1/J, nrow=n_ids, ncol=J), simplify=FALSE))
      }

      # Ensure we have enough predictions
      if(length(preds) <= window_size) {
        return(replicate(length(preds), preds[[1]], simplify=FALSE))
      }

      # Process predictions with proper dimensions
      c(replicate(window_size, {
        first_valid <- preds[[window_size + 1]]
        if(!is.matrix(first_valid)) {
          first_valid <- matrix(first_valid, nrow=n_ids)
        }
        first_valid
      }, simplify=FALSE),
      preds[(window_size + 1):length(preds)])
    }

    lstm_A_preds <- window_predictions(lstm_A_preds, window_size, n_ids)

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
      # CORRECT SCALING: Scale to [0.5, 1] range to prevent underflow
      g_preds_bin_cuml[[i]] <- g_preds_bin[[i]][which(g_preds_bin_ID[[i-1]]%in%g_preds_bin_ID[[i]]),] * 
        (0.5 + (g_preds_bin_cuml[[i-1]][which(g_preds_bin_ID[[i-1]]%in%g_preds_bin_ID[[i]]),] / 2))
    }
    g_preds_bin_cuml_bounded <- lapply(1:length(initial_model_for_A_bin), function(x) boundProbs(g_preds_bin_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores
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
      # Ensure matrix format
      if(is.null(dim(x))) {
        x <- matrix(x, ncol=1)
      }
      boundProbs(x, bounds = gbound)
    })
  }

  # sequential g-formula
  # model is fit on all uncensored and alive (until t-1)
  # the outcome is the observed Y for t=T and updated Y if t<T
  
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
    # fit on those uncensored until t-1
    
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
    
    # Process g predictions
    g_preds_processed <- safe_get_cuml_preds(g_preds, n_ids)
    g_preds_bin_processed <- safe_get_cuml_preds(g_preds_bin, n_ids)
    print("G predictions processed")
    
    # Process C predictions
    C_preds_processed <- safe_get_cuml_preds(C_preds, n_ids)
    print("C predictions processed")
    
    n_ids <- length(unique(initial_model_for_Y$data$ID))
    
    results <- process_time_points(
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
  
  if(estimator=='tmle-lstm'){
    # Calculate TMLE estimates - fix dimensions and calculations 
    # Conversion to survival probabilities (1-p) only happens at the visualization stage, which is correct.
    tmle_estimates <- matrix(NA, nrow=3, ncol=t.end)
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]])) {
        for(rule in 1:3) {
          tmle_estimates[rule,t] <- 1-mean(tmle_contrasts[[t]]$Qstar[,rule], na.rm=TRUE) # convert event prob. into survival prob.
        }
      }
    }
    
    # Calculate binary TMLE estimates
    tmle_bin_estimates <- matrix(NA, nrow=3, ncol=t.end)
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts_bin[[t]])) {
        for(rule in 1:3) {
          tmle_bin_estimates[rule,t] <- 1-mean(tmle_contrasts_bin[[t]]$Qstar[,rule], na.rm=TRUE)
        }
      }
    }
    
    # Calculate IPTW estimates
    iptw_estimates <- matrix(NA, nrow=3, ncol=t.end)
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]]$Qstar_iptw)) {
        iptw_means <- tmle_contrasts[[t]]$Qstar_iptw[1,]
        for(rule in 1:3) {
          iptw_estimates[rule,t] <- 1-iptw_means[rule]
        }
      }
    }
    
    # Calculate binary IPTW estimates  
    iptw_bin_estimates <- matrix(NA, nrow=3, ncol=t.end)
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts_bin[[t]]$Qstar_iptw)) {
        iptw_means <- tmle_contrasts_bin[[t]]$Qstar_iptw[1,]
        for(rule in 1:3) {
          iptw_bin_estimates[rule,t] <- 1-iptw_means[rule]
        }
      }
    }
    
    # Calculate G-computation estimates
    gcomp_estimates <- matrix(NA, nrow=3, ncol=t.end)
    for(t in 1:t.end) {
      if(!is.null(tmle_contrasts[[t]]$Qstar_gcomp)) {
        for(rule in 1:3) {
          gcomp_estimates[rule,t] <- 1-mean(tmle_contrasts[[t]]$Qstar_gcomp[,rule], na.rm=TRUE)
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
    
    if(debug) {
      cat("\nRaw Qstar values from first timepoint:\n")
      if(!is.null(tmle_contrasts[[1]])) {
        print(head(tmle_contrasts[[1]]$Qstar))
        cat("\nMeans by rule:\n")
        print(colMeans(tmle_contrasts[[1]]$Qstar, na.rm=TRUE))
      }
      
      cat("\nQstar_iptw values from first timepoint:\n")
      if(!is.null(tmle_contrasts[[1]])) {
        print(tmle_contrasts[[1]]$Qstar_iptw)
      }
      
      cat("\nFirst few rows of final estimates matrices:\n")
      cat("TMLE estimates:\n")
      print(head(tmle_estimates))
      cat("\nIPTW estimates:\n") 
      print(head(iptw_estimates))
      cat("\nG-comp estimates:\n")
      print(head(gcomp_estimates))
    }
  } else{
    tmle_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar[[x]]))) # static, dynamic, stochastic
    tmle_bin_estimates <-  cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t]][,x]$Qstar[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t.end]]$Qstar[[x]]))) 
    
    iptw_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar_iptw[[x]], na.rm=TRUE))) # static, dynamic, stochastic
    iptw_bin_estimates <-  cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t.end]]$Qstar_iptw[[x]], na.rm=TRUE))) 
    
    gcomp_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar_gcomp[[x]]))) # static, dynamic, stochastic
  }
  
  if(r==1){
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
  
  if(estimator == "tmle-lstm") {
    prob_share <- vector("list", t.end + 1)
    prob_share_bin <- vector("list", t.end + 1)
    
    for(t in 1:(t.end+1)) {
      # Get predictions and ensure matrix format
      g_preds_t <- g_preds_processed[[t]]
      g_preds_bin_t <- g_preds_bin_processed[[t]]
      
      # Force matrix format with proper dimensions
      if(!is.matrix(g_preds_t)) {
        g_preds_t <- matrix(g_preds_t, ncol=6)  # J=6 treatments
      }
      if(!is.matrix(g_preds_bin_t)) {
        g_preds_bin_t <- matrix(g_preds_bin_t, ncol=6)
      }
      
      # Get current rules and ensure matrix format
      current_rules <- obs.rules[[t]]
      if(!is.matrix(current_rules)) {
        current_rules <- matrix(current_rules, ncol=3)  # 3 rules: static, dynamic, stochastic
      }
      
      # Calculate shares with proper dimension handling
      prob_share[[t]] <- matrix(0, nrow=ncol(g_preds_t), ncol=ncol(current_rules))
      for(i in seq_len(ncol(current_rules))) {
        rule_indices <- which(current_rules[,i] == 1)
        if(length(rule_indices) > 0) {
          g_preds_subset <- g_preds_t[rule_indices,, drop=FALSE]
          prob_share[[t]][,i] <- colMeans(g_preds_subset < 0.025, na.rm=TRUE)
        }
      }
      colnames(prob_share[[t]]) <- c("static", "dynamic", "stochastic")
      
      # Same for binary predictions
      prob_share_bin[[t]] <- matrix(0, nrow=ncol(g_preds_bin_t), ncol=ncol(current_rules))
      for(i in seq_len(ncol(current_rules))) {
        rule_indices <- which(current_rules[,i] == 1)
        if(length(rule_indices) > 0) {
          g_preds_subset <- g_preds_bin_t[rule_indices,, drop=FALSE]
          prob_share_bin[[t]][,i] <- colMeans(g_preds_subset < 0.025, na.rm=TRUE)
        }
      }
      colnames(prob_share_bin[[t]]) <- c("static", "dynamic", "stochastic")
    }
    
    # Add names to list elements
    names(prob_share) <- paste0("t=", seq(0, t.end))
    names(prob_share_bin) <- paste0("t=", seq(0, t.end))
  }else{
    prob_share <- lapply(1:(t.end+1), function(t) sapply(1:ncol(obs.rules[[(t)]]), function(i) colMeans(g_preds_processed[[(t)]][which(obs.rules[[(t)]][na.omit(tmle_dat[tmle_dat$t==(t),])$ID,][,i]==1),]<0.025, na.rm=TRUE)))
    for(t in 1:(t.end+1)){
      colnames(prob_share[[t]]) <- colnames(obs.rules[[(t.end)]])
    }
    
    names(prob_share) <- paste0("t=",seq(0,t.end))
    
    prob_share_bin <- lapply(1:(t.end+1), function(t) sapply(1:ncol(obs.rules[[(t)]]), function(i) colMeans(g_preds_bin_processed[[(t)]][which(obs.rules[[(t)]][na.omit(tmle_dat[tmle_dat$t==(t),])$ID,][,i]==1),]<0.025, na.rm=TRUE)))
    for(t in 1:(t.end+1)){
      colnames(prob_share_bin[[t]]) <- colnames(obs.rules[[(t.end)]])
    }
    
    names(prob_share_bin) <- paste0("t=",seq(0,t.end))
  }
  
  print("Calculating CIs") 
  
  if(estimator=="tmle-lstm"){
    tmle_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, estimator="tmle-lstm")
    tmle_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y, time.censored, estimator="tmle-lstm")
    
    iptw_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, iptw=TRUE, estimator="tmle-lstm")
    iptw_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y, time.censored, iptw=TRUE, estimator="tmle-lstm")
    
    gcomp_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, gcomp=TRUE, estimator="tmle-lstm")
  }else{
    tmle_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, estimator="tmle")
    tmle_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y, time.censored, estimator="tmle")
    
    iptw_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, iptw=TRUE, estimator="tmle")
    iptw_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y, time.censored, iptw=TRUE, estimator="tmle")
    
    gcomp_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, gcomp=TRUE, estimator="tmle")
  }
  
  print("Storing results")
  
  Ahat_tmle  <- g_preds_processed
  Ahat_tmle_bin  <- g_preds_bin_processed
  
  Chat_tmle  <- C_preds_cuml_bounded
  
  print("Calculating bias, CP, CIW wrt to est at each t")
  
  bias_tmle  <- lapply(2:t.end, function(t) sapply(Y.true,"[[",t) - tmle_est_var$est[[t]])
  names(bias_tmle) <- paste0("t=",2:t.end)
  
  CP_tmle <- lapply(1:(t.end-1), function(t) as.numeric((tmle_est_var$CI[[t]][1,] < sapply(Y.true,"[[",t)) & 
                                                          (tmle_est_var$CI[[t]][2,] > sapply(Y.true,"[[",t))))
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
settings <- expand.grid("n"=c(20000), 
                        treatment.rule = c("all")) 

options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE) # command line arguments # args <- c('tmle','TRUE','FALSE')
estimator <- as.character(args[1])
use.SL <- as.logical(args[2])  # When TRUE, use Super Learner for initial Y model and treatment model estimation; if FALSE, use GLM
doMPI <- as.logical(args[3])

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

n.folds <- 5

window_size <- 12

debug <- TRUE

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
  
  cores <- ceiling((parallel::detectCores())/2)
  print(paste0("number of cores used: ", cores))
  
  cl <- parallel::makeCluster(cores, outfile="")
  
  doParallel::registerDoParallel(cl) # register cluster
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

if(estimator=='tmle-lstm'){ # run sequentially and save at each iteration
  sim.results <- foreach(r = final_vector, .combine='cbind', .errorhandling="pass", .packages=library_vector, .verbose = FALSE, .export = c("output_dir", "simLong", "lstm", "process_time_points", "TMLE_IC", "prepare_lstm_data")) %do% {
    simLong(r=r, J=J, n=n, t.end=t.end, gbound=gbound, ybound=ybound, n.folds=n.folds, 
            cores=cores, estimator=estimator, treatment.rule=treatment.rule, 
            use.SL=use.SL, scale.continuous=scale.continuous, debug=debug, window_size=window_size)
  }
}else{ # run in parallel
  sim.results <- foreach(r = 1:R, .combine='cbind', .errorhandling="pass", .packages=library_vector, .verbose = FALSE, .inorder=FALSE) %dopar% {
    simLong(r=r, J=J, n=n, t.end=t.end, gbound=gbound, ybound=ybound, n.folds=n.folds, 
            cores=cores, estimator=estimator, treatment.rule=treatment.rule, 
            use.SL=use.SL, scale.continuous=scale.continuous, debug=debug, window_size=window_size)
  }
}

saveRDS(sim.results, filename)

if(doMPI){
  closeCluster(cl) # close down MPIcluster
  mpi.finalize()
}else{
  stopCluster(cl)
}