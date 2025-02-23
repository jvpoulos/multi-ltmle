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
  shifted <- factor(0, levels=levels(row$A))
  if(row$t==0){ # first-, second-, and third-order lags are 0
    if(row$schiz==1){
      shifted <- static_halo_on(row,lags=TRUE)
    }else if(row$bipolar==1){
      shifted <- static_quet_on(row,lags=TRUE)
    }else if(row$mdd==1){
      shifted <- static_arip_on(row,lags=TRUE)
    }else{
      shifted <- factor(0, levels=levels(row$A))
    }
  }else if(row$t>=1){
    lags <- row[grep("A",grep("lag",colnames(row), value=TRUE), value=TRUE)]
    if(row$schiz==1){
      shifted <- unlist(c(static_risp_on(row,lags = FALSE),lags)) # switch to risp
    }else if(row$bipolar==1){
      shifted <- unlist(c(static_quet_on(row,lags = FALSE),lags)) # switch to quet
    }else if(row$mdd==1){
      shifted <- unlist(c(static_arip_on(row,lags = FALSE),lags)) # switch to arip
    }else{
      shifted <- factor(0, levels=levels(row$A))
    }
  }
  return(shifted)
}

dynamic_mtp <- function(row){ 
  # Dynamic: Everyone starts with risp.
  # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to quetiap. (if bipolar), halo. (if schizophrenia), ari (if MDD); otherwise stay on risp.
  shifted <- factor(0, levels=levels(row$A))
  if(row$t==0){ # first-, second-, and third-order lags are 0
    shifted <- static_risp_on(row,lags=TRUE)
  }else if(row$t>=1){
    lags <- row[grep("A",grep("lag",colnames(row), value=TRUE), value=TRUE)]
    if (!is.na(row$L1) && !is.na(row$L2) && !is.na(row$L3) && (row$L1 > 0 | row$L2 > 0 | row$L3 > 0)) {
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
  shifted <- factor(0, levels=levels(row$A))
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

getTMLELong <- function(initial_model_for_Y, tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end, analysis=FALSE){
  
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
  if(analysis){
    clever_covariates <- lapply(obs.rules, function(rule_matrix) {
      (rule_matrix[initial_model_for_Y_data$ID, ] * (1 - C))
    })
    
    # Compute weights as a list
    weights <- lapply(seq_along(clever_covariates), function(i) {
      clever_covariates[[i]] / rowSums(obs.treatment[initial_model_for_Y_data$ID, ] * boundProbs(g_preds_bounded[initial_model_for_Y_data$ID, ] * C_preds_bounded[initial_model_for_Y_data$ID], bounds = gbound))
    })
    
    # Targeting step - refit outcome model using clever covariates
    if (all(initial_model_for_Y$data$t < t.end)) { # use actual Y for t=T
      updated_model_for_Y <- lapply(seq_along(clever_covariates), function(i) {
        glm(QAW$QA ~ 1 + offset(qlogis(QAW[, (i + 1)])), weights = weights[[i]], family = quasibinomial())
      })
    } else {
      updated_model_for_Y <- lapply(seq_along(clever_covariates), function(i) {
        glm(initial_model_for_Y$data$Y ~ 1 + offset(qlogis(QAW[, (i + 1)])), weights = weights[[i]], family = quasibinomial())
      })
    }
    
    Qstar <- lapply(seq_along(clever_covariates), function(i) {
      predict(updated_model_for_Y[[i]], type = "response")
    })
    names(Qstar) <- colnames(obs.rules)
    
    # IPTW estimate
    Qstar_iptw <- lapply(seq_along(clever_covariates), function(i) {
      boundProbs(weights[[i]] * initial_model_for_Y$data$Y, bound = ybound)
    })
    names(Qstar_iptw) <- colnames(obs.rules)
    
    # gcomp estimate
    Qstar_gcomp <- lapply(seq_along(clever_covariates), function(i) {
      QAW[, (i + 1)]
    })
    names(Qstar_gcomp) <- colnames(obs.rules)
  }else{
    clever_covariates <- (obs.rules[initial_model_for_Y_data$ID,]*(1-C)) # numerator
    
    if(dim(g_preds_bounded)[1]>length(initial_model_for_Y_data$ID)){
      weights <- clever_covariates/rowSums(obs.treatment[initial_model_for_Y_data$ID,]*boundProbs(g_preds_bounded[initial_model_for_Y_data$ID,]*C_preds_bounded[initial_model_for_Y_data$ID], bounds = gbound)) # denominator: clever covariate used as weight in regression
    }else{
      weights <- clever_covariates/rowSums(obs.treatment[initial_model_for_Y_data$ID,]*boundProbs(g_preds_bounded*C_preds_bounded, bounds = gbound)) # denominator: clever covariate used as weight in regression
    }
    
    # targeting step - refit outcome model using clever covariates
    if(all(initial_model_for_Y$data$t<t.end)){ # use actual Y for t=T
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
  }
  return(list("Qs"=Qs,"QAW"=QAW,"clever_covariates"=clever_covariates,"weights"=weights,"updated_model_for_Y"=updated_model_for_Y, "Qstar"=Qstar, "Qstar_iptw"=Qstar_iptw, "Qstar_gcomp"=Qstar_gcomp, "ID"=initial_model_for_Y_data$ID))
}