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
    
    # Create rule-specific weights
    # For treatment probabilities under this specific rule
    rule_data <- rule_data_frames[[rule]]
    
    # Better handling of dimension matching
    if(dim(g_preds_bounded)[1] > length(initial_model_for_Y_data$ID)){
      weights[[rule]] <- clever_covariates[[rule]] / 
        rowSums(obs.treatment[initial_model_for_Y_data$ID,] * 
                  boundProbs(g_preds_bounded[initial_model_for_Y_data$ID,] * 
                               C_preds_bounded[initial_model_for_Y_data$ID], bounds = gbound))
    } else {
      weights[[rule]] <- clever_covariates[[rule]] / 
        rowSums(obs.treatment[initial_model_for_Y_data$ID,] * 
                  boundProbs(g_preds_bounded * C_preds_bounded, bounds = gbound))
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