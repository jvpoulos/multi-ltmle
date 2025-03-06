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
    
    # Convert Y_pred to numeric if it's a list
    if(is.list(Y_pred)) {
      Y_pred <- unlist(Y_pred)
    }
    
    # Ensure IDs match between Y_pred and tmle_dat_sub
    matching_ids <- as.numeric(names(Y_pred)) %in% tmle_dat_sub$ID
    tmle_dat_sub$Y <- Y_pred[matching_ids]
  }
  
  # Use fewer folds and simpler cross-validation
  folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
  
  # Define task with appropriate outcome type
  initial_model_for_Y_task <- make_sl3_Task(
    data = tmle_dat_sub,
    covariates = tmle_covars_Y, 
    outcome = "Y",
    outcome_type = ifelse(!is.null(Y_pred), "continuous", "binomial"), 
    folds = folds
  ) 
  
  # Try-catch block to handle potential errors gracefully
  initial_model_for_Y_sl_fit <- tryCatch({
    # Train the model
    fit <- initial_model_for_Y_sl$train(initial_model_for_Y_task)
    fit
  }, error = function(e) {
    # Fallback to a simpler model if SL fails
    message("SuperLearner failed with error: ", e$message)
    message("Falling back to mean learner")
    mean_learner <- make_learner(Lrnr_mean)
    mean_learner$train(initial_model_for_Y_task)
  })
  
  # Predict on everyone
  prediction_task <- sl3_Task$new(
    data = tmle_dat[tmle_dat$t==t,], 
    covariates = tmle_covars_Y, 
    outcome = "Y", 
    outcome_type = ifelse(!is.null(Y_pred), "continuous", "binomial")
  )
  
  Y_preds <- tryCatch({
    initial_model_for_Y_sl_fit$predict(prediction_task)
  }, error = function(e) {
    message("Prediction failed with error: ", e$message)
    message("Using default predictions")
    rep(mean(tmle_dat_sub$Y, na.rm=TRUE), nrow(tmle_dat[tmle_dat$t==t,]))
  })
  
  # Ensure Y_preds is numeric
  Y_preds <- as.numeric(Y_preds)
  
  # Apply bounds safely
  Y_preds_bounded <- pmin(pmax(Y_preds, ybound[1]), ybound[2])
  
  return(list(
    "preds" = Y_preds_bounded,
    "fit" = initial_model_for_Y_sl_fit,
    "data" = tmle_dat[tmle_dat$t==t,]
  ))
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
  
  # Create a unique data frame for each rule's shifted treatment values
  rule_data_frames <- list()
  for(rule in names(tmle_rules)){
    data <- initial_model_for_Y_data[grep("A",colnames(initial_model_for_Y_data), value=TRUE, invert=TRUE)]
    shifted <- tmle_rules[[rule]](initial_model_for_Y_data[1,]) # first row
    for(i in 2:nrow(initial_model_for_Y_data)){ # bind row-by-row
      shifted <- rbind(shifted, tmle_rules[[rule]](initial_model_for_Y_data[i,]))
    }
    newdata <- cbind(data, shifted)
    rule_data_frames[[rule]] <- newdata
    assign(paste0("Q_",rule), initial_model_for_Y_sl_fit$predict(sl3_Task$new(newdata, covariates = tmle_covars_Y)))
    rm(data, shifted, newdata)
  }
  
  Qs <- mget(c(paste0("Q_", names(tmle_rules)))) # list of treatment.rule
  Qs <- do.call(cbind, Qs)
  
  QAW <- data.frame(apply(cbind(QA=initial_model_for_Y_preds,Qs), 2, boundProbs, bounds=ybound)) # bound predictions
  
  # For each treatment rule, calculate rule-specific weights
  clever_covariates <- list()
  weights <- list()
  
  for(rule_idx in 1:length(names(tmle_rules))){
    rule <- names(tmle_rules)[rule_idx]
    
    # Create rule-specific clever covariate
    clever_covariates[[rule]] <- (obs.rules[initial_model_for_Y_data$ID, rule_idx] * (1-C))
    
    # Create rule-specific weights with improved stability
    # For treatment probabilities under this specific rule
    rule_data <- rule_data_frames[[rule]]
    
    # Better handling of dimension matching
    if(dim(g_preds_bounded)[1] > length(initial_model_for_Y_data$ID)){
      # Calculate product first, then bound to avoid numerical issues
      weights[[rule]] <- clever_covariates[[rule]] / 
        rowSums(obs.treatment[initial_model_for_Y_data$ID,] * 
                  boundProbs(g_preds_bounded[initial_model_for_Y_data$ID,] * 
                               C_preds_bounded[initial_model_for_Y_data$ID], bounds = gbound))
    } else {
      # Use more stable calculation - bound each component first 
      g_preds_bounded_safe <- boundProbs(g_preds_bounded, bounds = c(0.01, 0.99))
      C_preds_bounded_safe <- boundProbs(C_preds_bounded, bounds = c(0.01, 0.99))
      
      # Calculate product with stronger bounds to prevent extreme ratios
      weights[[rule]] <- clever_covariates[[rule]] / 
        pmax(0.01, rowSums(obs.treatment[initial_model_for_Y_data$ID,] * 
                             (g_preds_bounded_safe * C_preds_bounded_safe)))
    }
    
    # Apply weight normalization and trimming for stability
    # Trim extreme weights (similar to LSTM approach)
    if(sum(weights[[rule]] > 0) > 0) {
      max_weight <- quantile(weights[[rule]][weights[[rule]] > 0], 0.95, na.rm=TRUE)
      weights[[rule]] <- pmin(weights[[rule]], max_weight)
      # Normalize weights to sum to 1
      if(sum(weights[[rule]], na.rm=TRUE) > 0) {
        weights[[rule]] <- weights[[rule]] / sum(weights[[rule]], na.rm=TRUE)
      }
    }
  }
  
  # Targeting step - refit outcome model using clever covariates
  updated_model_for_Y <- list()
  for(rule_idx in 1:length(names(tmle_rules))){
    rule <- names(tmle_rules)[rule_idx]
    
    if(all(initial_model_for_Y$data$t < t.end)){ # use actual Y for t=T
      # Use adaptive epsilon based on rule-specific weights
      model_data <- data.frame(
        y = QAW$QA,
        offset = qlogis(pmax(pmin(QAW[,(rule_idx+1)], 0.9999), 0.0001)),
        weights = weights[[rule]]
      )
      
      # Better handling of numeric issues in GLM
      valid_rows <- complete.cases(model_data) & 
        is.finite(model_data$y) &
        is.finite(model_data$offset) &
        is.finite(model_data$weights) &
        model_data$weights > 0
      
      if(sum(valid_rows) > 0) {
        updated_model_for_Y[[rule]] <- tryCatch({
          glm(y ~ 1 + offset(offset), 
              weights = weights, 
              family = quasibinomial(),
              data = model_data[valid_rows,])
        }, error = function(e) {
          # Fallback if GLM fails
          NULL
        })
      } else {
        updated_model_for_Y[[rule]] <- NULL
      }
    } else {
      # For t=T, use actual outcomes
      model_data <- data.frame(
        y = initial_model_for_Y$data$Y,
        offset = qlogis(pmax(pmin(QAW[,(rule_idx+1)], 0.9999), 0.0001)),
        weights = weights[[rule]]
      )
      
      valid_rows <- complete.cases(model_data) & 
        is.finite(model_data$y) &
        is.finite(model_data$offset) &
        is.finite(model_data$weights) &
        model_data$weights > 0 &
        model_data$y != -1  # Not censored
      
      if(sum(valid_rows) > 0) {
        updated_model_for_Y[[rule]] <- tryCatch({
          glm(y ~ 1 + offset(offset), 
              weights = weights, 
              family = quasibinomial(),
              data = model_data[valid_rows,])
        }, error = function(e) {
          # Fallback if GLM fails
          NULL
        })
      } else {
        updated_model_for_Y[[rule]] <- NULL
      }
    }
  }
  
  # Generate targeted predictions
  Qstar <- list()
  for(rule_idx in 1:length(names(tmle_rules))){
    rule <- names(tmle_rules)[rule_idx]
    
    if(is.null(updated_model_for_Y[[rule]])) {
      # If model fitting failed, fall back to initial prediction
      Qstar[[rule]] <- QAW[,(rule_idx+1)]
    } else {
      # Get epsilon from model
      epsilon <- coef(updated_model_for_Y[[rule]])[1]
      
      # Apply targeting transformation
      expit <- function(x) 1/(1+exp(-x))
      logit <- function(p) log(p/(1-p))
      
      # Apply targeting with bounded values
      Qstar[[rule]] <- pmin(pmax(
        expit(logit(pmax(pmin(QAW[,(rule_idx+1)], 0.9999), 0.0001)) + epsilon),
        ybound[1]), ybound[2])
    }
  }
  
  # IPTW estimate - improve weighting scheme
  Qstar_iptw <- list()
  for(rule_idx in 1:length(names(tmle_rules))){
    rule <- names(tmle_rules)[rule_idx]
    
    valid_idx <- clever_covariates[[rule]] > 0
    if(any(valid_idx)) {
      # Use weights to calculate IPTW estimate
      outcome_vals <- initial_model_for_Y$data$Y[valid_idx]
      weight_vals <- weights[[rule]][valid_idx]
      
      # Handle NAs and censoring
      valid_outcome <- !is.na(outcome_vals) & outcome_vals != -1
      if(any(valid_outcome)) {
        Qstar_iptw[[rule]] <- weighted.mean(
          outcome_vals[valid_outcome], 
          weight_vals[valid_outcome],
          na.rm = TRUE)
      } else {
        Qstar_iptw[[rule]] <- mean(QAW[,(rule_idx+1)], na.rm=TRUE)
      }
    } else {
      Qstar_iptw[[rule]] <- mean(QAW[,(rule_idx+1)], na.rm=TRUE)
    }
  }
  
  # G-computation estimate - keep original calculation
  Qstar_gcomp <- list()
  for(rule_idx in 1:length(names(tmle_rules))){
    rule <- names(tmle_rules)[rule_idx]
    Qstar_gcomp[[rule]] <- QAW[,(rule_idx+1)]
  }
  
  # Return results
  return(list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = clever_covariates,
    "weights" = weights,
    "updated_model_for_Y" = updated_model_for_Y, 
    "Qstar" = Qstar, 
    "Qstar_iptw" = Qstar_iptw, 
    "Qstar_gcomp" = Qstar_gcomp, 
    "ID" = initial_model_for_Y_data$ID
  ))
}