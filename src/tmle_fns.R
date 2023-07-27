###################################################################
# treatment regime functions                                      #
###################################################################

static_arip_on <- function(row,lags=TRUE) {
  #  binary treatment is set to aripiprazole at all time points for all observations
  if(lags){
    treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  }else{
    treats <- row[grep("A[0-9]$",colnames(row), value=TRUE)]
  }
  if(row$t==1){ # first-, second-, and third-order lags are 0
    shifted <- ifelse(names(treats)%in%grep("^A1$",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }else if(row$t==2){ #second- and third-order lags are zero
    shifted <- ifelse(names(treats)%in%c(grep("^A1$",colnames(row), value=TRUE),grep("^A1.lag$",colnames(row), value=TRUE)),1,0)
    names(shifted) <- names(treats)
  }else if (row$t>2){ #turn on all lags
    shifted <- ifelse(names(treats)%in%grep("A1",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }
  return(shifted)
}

static_halo_on <- function(row,lags=TRUE) {
  if(lags){
    treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  }else{
    treats <- row[grep("A[0-9]$",colnames(row), value=TRUE)]
  }
  #  binary treatment is set to haloperidol at all time points for all observations
  if(row$t==1){ # first-, second-, and third-order lags are 0
    shifted <- ifelse(names(treats)%in%grep("^A2$",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }else if(row$t==2){ #second- and third-order lags are zero
    shifted <- ifelse(names(treats)%in%c(grep("^A2$",colnames(row), value=TRUE),grep("^A2.lag$",colnames(row), value=TRUE)),1,0)
    names(shifted) <- names(treats)
  }else if (row$t>2){ #turn on all lags
    shifted <- ifelse(names(treats)%in%grep("A2",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }
  return(shifted)
}

static_olanz_on <- function(row,lags=TRUE) {
  if(lags){
    treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  }else{
    treats <- row[grep("A[0-9]$",colnames(row), value=TRUE)]
  }
  #  binary treatment is set to olanzapine at all time points for all observations
  if(row$t==1){ # first-, second-, and third-order lags are 0
    shifted <- ifelse(names(treats)%in%grep("^A3$",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }else if(row$t==2){ #second- and third-order lags are zero
    shifted <- ifelse(names(treats)%in%c(grep("^A3$",colnames(row), value=TRUE),grep("^A3.lag$",colnames(row), value=TRUE)),1,0)
    names(shifted) <- names(treats)
  }else if (row$t>2){ #turn on all lags
    shifted <- ifelse(names(treats)%in%grep("A3",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }
  return(shifted)
}

static_risp_on <- function(row,lags=TRUE) {
  if(lags){
    treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  }else{
    treats <- row[grep("A[0-9]$",colnames(row), value=TRUE)]
  }
  #  binary treatment is set to risperidone at all time points for all observations
  if(row$t==1){ # first-, second-, and third-order lags are 0
    shifted <- ifelse(names(treats)%in%grep("^A5$",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }else if(row$t==2){ #second- and third-order lags are zero
    shifted <- ifelse(names(treats)%in%c(grep("^A5$",colnames(row), value=TRUE),grep("^A5.lag$",colnames(row), value=TRUE)),1,0)
    names(shifted) <- names(treats)
  }else if (row$t>2){ #turn on all lags
    shifted <- ifelse(names(treats)%in%grep("A5",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }
  return(shifted)
}

static_quet_on <- function(row,lags=TRUE) {
  if(lags){
    treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  }else{
    treats <- row[grep("A[0-9]$",colnames(row), value=TRUE)]
  }
  #  binary treatment is set to quetiapine at all time points for all observations
  if(row$t==1){ # first-, second-, and third-order lags are 0
    shifted <- ifelse(names(treats)%in%grep("^A4$",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }else if(row$t==2){ #second- and third-order lags are zero
    shifted <- ifelse(names(treats)%in%c(grep("^A4$",colnames(row), value=TRUE),grep("^A4.lag$",colnames(row), value=TRUE)),1,0)
    names(shifted) <- names(treats)
  }else if (row$t>2){ #turn on all lags
    shifted <- ifelse(names(treats)%in%grep("A4",colnames(row), value=TRUE),1,0)
    names(shifted) <- names(treats)
  }
  return(shifted)
}

static_mtp <- function(row){ 
  # Static: Everyone gets quetiap (if bipolar), halo (if schizophrenia), ari (if MDD) and stays on it
  if(row$t==0){ # first-, second-, and third-order lags are 0
    if(row$schiz==1){
      shifted <- static_halo_on(row,lags=TRUE)
    }else if(row$bipolar==1){
      shifted <- static_quet_on(row,lags=TRUE)
    }else if(row$mdd==1){
      shifted <- static_arip_on(row,lags=TRUE)
    }
  }else if(row$t>=1){
    lags <- row[grep("A",grep("lag",colnames(row), value=TRUE), value=TRUE)]
    if(row$schiz==1){
      shifted <- unlist(c(static_risp_on(row,lags = FALSE),lags)) # switch to risp
    }else if(row$bipolar==1){
      shifted <- unlist(c(static_quet_on(row,lags = FALSE),lags)) # switch to quet
    }else if(row$mdd==1){
      shifted <- unlist(c(static_arip_on(row,lags = FALSE),lags)) # switch to arip
    }
  }
  return(shifted)
}

dynamic_mtp <- function(row){ 
  # Dynamic: Everyone starts with risp.
  # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to quetiap. (if bipolar), halo. (if schizophrenia), ari (if MDD); otherwise stay on risp.
  if(row$t==0){ # first-, second-, and third-order lags are 0
    shifted <- static_risp_on(row,lags=TRUE)
  }else if(row$t>=1){
    lags <- row[grep("A",grep("lag",colnames(row), value=TRUE), value=TRUE)]
    if((row$L1 >0 | row$L2 >0 | row$L3 >0)){
      if(row$schiz==1){
        shifted <- unlist(c(static_halo_on(row,lags = FALSE),lags)) # switch to halo
      }else if(row$bipolar==1){
        shifted <- unlist(c(static_quet_on(row,lags = FALSE),lags)) # switch to quet.
      }else if(row$mdd==1){
        shifted <- unlist(c(static_arip_on(row,lags = FALSE),lags)) # switch to arip
      }
    }else{
      shifted <- unlist(c(static_risp_on(row,lags=FALSE),lags))  # otherwise stay on risp.
    }
  }
  return(shifted)
}

stochastic_mtp <- function(row){
  # Stochastic: at each t>0, 95% chance of staying with treatment at t-1, 5% chance of randomly switching according to Multinomial distibution
  if(row$t==0){ # do nothing first period
    shifted <- row[grep("A[0-9]",colnames(row), value=TRUE)]  # first and second-order lags are 0
  }else if(row$t>=1){
    lags <- row[grep("A",grep("lag",colnames(row), value=TRUE), value=TRUE)]
    random_treat <- Multinom(1, StochasticFun(row[grep("A[0-9]$",colnames(row), value=TRUE)], d=c(0,0,0,0,0,0))[which(row[grep("A[0-9]$",colnames(row), value=TRUE)]==1),])
    shifted <- row[grep("A[0-9]$",colnames(row), value=TRUE)]
    shifted[shifted>0] <- 0
    shifted[as.numeric(random_treat)] <- 1
    shifted <- unlist(c(shifted,lags))
  }
  return(shifted)
}

###################################################################
# Sequential-g estimator                                          #
###################################################################

sequential_g <- function(t, tmle_dat, n.folds, tmle_covars_Y, initial_model_for_Y_sl, ybound, Y_pred=NULL){
  
  tmle_dat_sub <- tmle_dat[tmle_dat$t==t & !is.na(tmle_dat$Y),] # drop rows with missing Y
  
  if(!is.null(Y_pred)){ # for t<T
    tmle_dat_sub <- tmle_dat_sub[tmle_dat_sub$ID %in% as.numeric(names(Y_pred)) %in% tmle_dat_sub$ID,] # use ppl who were uncensored until t-1
    tmle_dat_sub$Y <- Y_pred[as.numeric(names(Y_pred)) %in% tmle_dat_sub$ID]
  }
  
  folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
  
  # define task and candidate learners
  initial_model_for_Y_task <- make_sl3_Task(data=tmle_dat_sub,
                                            covariates = tmle_covars_Y, 
                                            outcome = "Y",
                                            outcome_type=ifelse(!is.null(Y_pred), "continuous", "binomial"), 
                                            folds = folds) 
  # train
  initial_model_for_Y_sl_fit <- initial_model_for_Y_sl$train(initial_model_for_Y_task)
  
  # predict on everyone
  Y_preds <- initial_model_for_Y_sl_fit$predict(sl3_Task$new(data=tmle_dat[tmle_dat$t==t,], covariates = tmle_covars_Y, outcome="Y", outcome_type=ifelse(!is.null(Y_pred), "continuous", "binomial")))
  
  return(list("preds"=boundProbs(Y_preds,ybound),
              "fit"=initial_model_for_Y_sl_fit,
              "data"=tmle_dat[tmle_dat$t==t,])) # evaluation data (fit on everyone)
}

###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

getTMLELong <- function(initial_model_for_Y, tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end){

  initial_model_for_Y_preds <- initial_model_for_Y$preds # t length list
  initial_model_for_Y_data <- initial_model_for_Y$data
  initial_model_for_Y_sl_fit <- initial_model_for_Y$fit
  
  C <- initial_model_for_Y$data$C # 1=Censored
  
  for(rule in names(tmle_rules)){
    data <- initial_model_for_Y_data[grep("A",colnames(initial_model_for_Y_data), value=TRUE, invert=TRUE)]
    shifted <- tmle_rules[[rule]](initial_model_for_Y_data[1,]) # first row
    for(i in 2:nrow(initial_model_for_Y_data)){ # bind row-by-row
      shifted <- rbind(shifted, tmle_rules[[rule]](initial_model_for_Y_data[i,]))
    }
    newdata <- cbind(data,shifted)
    assign(paste0("Q_",rule), initial_model_for_Y_sl_fit$predict(sl3_Task$new(newdata, covariates = tmle_covars_Y)))
    rm(data,shifted,newdata)
  }
  
  Qs <- mget(c(paste0("Q_", names(tmle_rules)))) # list of treatment.rule
  Qs <- do.call(cbind, Qs)
  
  QAW <- data.frame(apply(cbind(QA=initial_model_for_Y_preds,Qs), 2, boundProbs, bounds=ybound)) # bound predictions
  
  # inverse probs. only used for subset of patients following treatment rule and uncensored (=0 for those excluded)
  clever_covariates <- (obs.rules[initial_model_for_Y_data$ID,]*(1-C)) # numerator
  
  if(dim(g_preds_bounded)[1]>length(initial_model_for_Y_data$ID)){
    weights <- clever_covariates/rowSums(obs.treatment[initial_model_for_Y_data$ID,]*boundProbs(g_preds_bounded[initial_model_for_Y_data$ID,]*C_preds_bounded[initial_model_for_Y_data$ID], bounds = gbound)) # denominator: clever covariate used as weight in regression
  }else{
    weights <- clever_covariates/rowSums(obs.treatment[initial_model_for_Y_data$ID,]*boundProbs(g_preds_bounded*C_preds_bounded, bounds = gbound)) # denominator: clever covariate used as weight in regression
  }
  
  # targeting step - refit outcome model using clever covariates
  if(unique(initial_model_for_Y$data$t)<t.end){ # use actual Y for t=T
    updated_model_for_Y <- lapply(1:ncol(clever_covariates), function(i) glm(QAW$QA ~ 1 + offset(qlogis(QAW[,(i+1)])), weights=weights[,i], family=quasibinomial())) # plug-in predicted outcome used as offset
  }else{
    updated_model_for_Y <- lapply(1:ncol(clever_covariates), function(i) glm(initial_model_for_Y$data$Y~ 1 + offset(qlogis(QAW[,(i+1)])), weights=weights[,i], family=quasibinomial())) # plug-in predicted outcome used as offset
  }
  
  Qstar <- lapply(1:ncol(clever_covariates), function(i) predict(updated_model_for_Y[[i]], type="response"))
  names(Qstar) <- colnames(obs.rules)
  
  # IPTW estimate
  Qstar_iptw <- lapply(1:ncol(clever_covariates), function(i) boundProbs(weights[,i]*initial_model_for_Y$data$Y, bound=ybound)) 
  names(Qstar_iptw) <- colnames(obs.rules)
  
  # gcomp estimate
  Qstar_gcomp <- lapply(1:ncol(clever_covariates), function(i) QAW[,(i+1)])
  names(Qstar_gcomp) <- colnames(obs.rules)
  
  return(list("Qs"=Qs,"QAW"=QAW,"clever_covariates"=clever_covariates,"weights"=weights,"updated_model_for_Y"=updated_model_for_Y, "Qstar"=Qstar, "Qstar_iptw"=Qstar_iptw, "Qstar_gcomp"=Qstar_gcomp, "ID"=initial_model_for_Y_data$ID))
}

###################################################################
# Influence Curve                                                 #
###################################################################

TMLE_IC <- function(tmle_contrasts, initial_model_for_Y, time.censored, alpha=0.05, iptw=FALSE, gcomp=FALSE){
  
  t.end <- length(initial_model_for_Y)
  n.rules <- ncol(tmle_contrasts[[t.end]]$weights)
  
  # calcuate final TMLE estimate
  if(iptw){
    tmle_final <- lapply(1:t.end, function(t) sapply(1:n.rules, function(x) ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE), mean(tmle_contrasts[[t]]$Qstar_iptw[[x]], na.rm=TRUE))))
  } else if(gcomp){
    tmle_final <- lapply(1:t.end, function(t) sapply(1:n.rules, function(x) ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]], na.rm=TRUE), mean(tmle_contrasts[[t]]$Qstar_gcomp[[x]], na.rm=TRUE))))
  } else{
    tmle_final <- lapply(1:t.end, function(t) sapply(1:n.rules, function(x) ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar[[x]], na.rm=TRUE), mean(tmle_contrasts[[t]]$Qstar[[x]], na.rm=TRUE))))
  }
  
  for(t in 1:t.end){
    names(tmle_final[[t]]) <- colnames(tmle_contrasts[[t.end]]$weights)
  }
  
  # calculate influence curve at each t
  Y_uncensored <- lapply(1:(t.end-1), function(t) initial_model_for_Y[[t]][,1]$data$Y[!initial_model_for_Y[[t]][,1]$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) # Y_T is actual Y # same for each treatment rule
  Y_uncensored[[t.end]] <- initial_model_for_Y[[t.end]]$data$Y[!initial_model_for_Y[[t.end]]$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]
  
  for(t in 1:t.end){
    Y_uncensored[[t]][is.na(Y_uncensored[[t]])] <- 0 # replace missing with 0
  }
  
  infcurv <- list()
  if(gcomp){
    infcurv[[t.end]] <- sapply(1:n.rules, function(x) # last time period
      tmle_contrasts[[t.end]]$Qstar_gcomp[[x]][!tmle_contrasts[[t.end]]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_final[[t.end]][[x]] +
        (Y_uncensored[[t.end]] - tmle_contrasts[[(t.end-1)]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) + # final time period (use real Y)
        (tmle_contrasts[[2]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[1]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) + # first time period
        rowSums(sapply((t.end-1):2, function(t)
          (tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[(t-1)]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]])
        )))
    
    for(t in (t.end-1):2){   # time periods 2 to t.end-1
      infcurv[[t]] <- sapply(1:n.rules, function(x)
        tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_final[[t]][[x]] +
          (Y_uncensored[[t]] - tmle_contrasts[[(t-1)]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) + # final time period (use real Y)
          (tmle_contrasts[[2]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[1]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) + # first time period
          ifelse(t>2, rowSums(sapply((t-1):2, function(i)
            (tmle_contrasts[[i]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[i]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[(i-1)]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]])
          )), 0))
    }
  }else if(iptw){
    infcurv[[t.end]] <- sapply(1:n.rules, function(x) # last time period
      tmle_contrasts[[t.end]]$Qstar_iptw[[x]][!tmle_contrasts[[t.end]]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_final[[t.end]][[x]] +
        tmle_contrasts[[(t.end-1)]][,x]$weights[,x][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] *(Y_uncensored[[t.end]] - tmle_contrasts[[(t.end-1)]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) + # final time period (use real Y)
        tmle_contrasts[[1]][,x]$weights[,x][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]*(tmle_contrasts[[2]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[1]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) + # first time period
        rowSums(sapply((t.end-1):2, function(t)
          tmle_contrasts[[(t-1)]][,x]$weights[,x][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]*(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[(t-1)]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]])
        )))
    
    for(t in (t.end-1):2){   # time periods 2 to t.end-1
      infcurv[[t]] <- sapply(1:n.rules, function(x)
        tmle_contrasts[[t]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_final[[t]][[x]] +
          tmle_contrasts[[(t-1)]][,x]$weights[,x][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] *(Y_uncensored[[t]] - tmle_contrasts[[(t-1)]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) + # final time period (use real Y)
          tmle_contrasts[[1]][,x]$weights[,x][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]*(tmle_contrasts[[2]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[1]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) + # first time period
          ifelse(t>2, rowSums(sapply((t-1):2, function(i)
            tmle_contrasts[[(i-1)]][,x]$weights[,x][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]*(tmle_contrasts[[i]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[i]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[(i-1)]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]])
          )), 0))
    }
  } else{
    infcurv[[t.end]] <- sapply(1:n.rules, function(x) # last time period
      tmle_contrasts[[t.end]]$Qstar[[x]][!tmle_contrasts[[t.end]]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_final[[t.end]][[x]] +
        tmle_contrasts[[(t.end-1)]][,x]$weights[,x][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] *(Y_uncensored[[t.end]] - tmle_contrasts[[(t.end-1)]][,x]$Qstar[[x]][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) + # final time period (use real Y)
        tmle_contrasts[[1]][,x]$weights[,x][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]*(tmle_contrasts[[2]][,x]$Qstar[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[1]][,x]$Qstar[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) + # first time period
        rowSums(sapply((t.end-1):2, function(t)
          tmle_contrasts[[(t-1)]][,x]$weights[,x][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]*(tmle_contrasts[[t]][,x]$Qstar[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[(t-1)]][,x]$Qstar[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]])
        )))
    
    for(t in (t.end-1):2){   # time periods 2 to t.end-1
      infcurv[[t]] <- sapply(1:n.rules, function(x)
        tmle_contrasts[[t]][,x]$Qstar[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_final[[t]][[x]] +
          tmle_contrasts[[(t-1)]][,x]$weights[,x][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] *(Y_uncensored[[t]] - tmle_contrasts[[(t-1)]][,x]$Qstar[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) + # final time period (use real Y)
          tmle_contrasts[[1]][,x]$weights[,x][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]*(tmle_contrasts[[2]][,x]$Qstar[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[1]][,x]$Qstar[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) + # first time period
          ifelse(t>2, rowSums(sapply((t-1):2, function(i)
            tmle_contrasts[[(i-1)]][,x]$weights[,x][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]*(tmle_contrasts[[i]][,x]$Qstar[[x]][!tmle_contrasts[[i]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[(i-1)]][,x]$Qstar[[x]][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]])
          )), 0))
    }
  }
  
  for(t in 2:t.end){
    colnames(infcurv[[t]]) <- colnames(tmle_contrasts[[t.end]]$weights)
  }
  
  CIs <- lapply(2:t.end, function (t) (sapply(1:n.rules, function(x) CI(est=tmle_final[[t]][x], infcurv = infcurv[[t]][,x], alpha=0.05))))
  names(CIs) <- paste0("t=",2:t.end)
  
  vars <- lapply(2:t.end, function (t) (sapply(1:n.rules, function(x) var(infcurv[[t]][,x], na.rm = TRUE))))
  names(vars) <- paste0("t=",2:t.end)
  
  for(t in 2:(t.end-1)){
    colnames(CIs[[t]]) <- colnames(tmle_contrasts[[t.end]]$weights)
    names(vars[[t]]) <- colnames(tmle_contrasts[[t.end]]$weights)
  }
  return(list("infcurv"=infcurv, "CI"=CIs, "var"=vars,"est"=tmle_final))
}