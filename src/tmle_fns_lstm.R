###################################################################
# treatment regime functions                                      #
###################################################################

static_mtp_lstm <- function(row){ 
  # Static: Everyone gets quetiap (if bipolar), halo (if schizophrenia), ari (if MDD) and stays on it
  if(row$schiz==1){
    shifted <- factor(2, levels=levels(row$A))
  }else if(row$bipolar==1){
    shifted <- factor(4, levels=levels(row$A))
  }else if(row$mdd==1){
    shifted <- factor(1, levels=levels(row$A))
  }else if(row$schiz==-1 & row$bipolar==-1 & row$mdd==-1){
    shifted <- row
  }
  return(shifted)
}

dynamic_mtp_lstm <- function(row){ 
  # Dynamic: Everyone starts with risp.
  # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to quetiap. (if bipolar), halo. (if schizophrenia), ari (if MDD); otherwise stay on risp.
  if(row$t==0){ 
    shifted <- factor(5, levels=levels(row$A))
  }else if(row$t>=1){
    if((row$L1 >0 | row$L2 >0 | row$L3 >0)){
      if(row$schiz==1){
        shifted <- factor(2, levels=levels(row$A)) # switch to halo
      }else if(row$bipolar==1){
        shifted <- factor(4, levels=levels(row$A)) # switch to quet.
      }else if(row$mdd==1){
        shifted <- factor(1, levels=levels(row$A)) # switch to arip
      }
    }else{
      shifted <- factor(5, levels=levels(row$A))  # otherwise stay on risp.
    }
  }
  return(shifted)
}

stochastic_mtp_lstm <- function(row){
  # Stochastic: at each t>0, 95% chance of staying with treatment at t-1, 5% chance of randomly switching according to Multinomial distibution
  if(row$t==0){ # do nothing first period
    shifted <- row[grep("A",colnames(row), value=TRUE)]  
  }else if(row$t>=1){
    random_treat <- Multinom(1, StochasticFun(row[grep("A",colnames(row), value=TRUE)], d=c(0,0,0,0,0,0))[as.numeric(levels(row$A))[row$A]])
    shifted <- random_treat
  }
  return(shifted)
}

###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end){
  
  C <- initial_model_for_Y$data$C # 1=Censored
  
  for(rule in names(tmle_rules)){
    # Determine batch size
    batch_size <- 1000  # Adjust based on your data size and memory constraints
    
    # Number of batches
    num_batches <- ceiling(nrow(initial_model_for_Y_data) / batch_size)
    
    # Initialize an empty list to store batch results
    batch_results <- vector("list", num_batches)
    
    # Process each batch
    for (i in 1:num_batches) {
      # Calculate row indices for the current batch
      batch_indices <- ((i - 1) * batch_size + 1):min(i * batch_size, nrow(initial_model_for_Y_data))
      
      # Extract the batch data
      batch_data <- initial_model_for_Y_data[batch_indices, ]
      
      # Apply the function to the batch and store the result
      batch_results[[i]] <- do.call(rbind, lapply(seq_len(nrow(batch_data)), function(j) {
        # Extract the j-th row as a data frame
        row_df <- batch_data[j, , drop = FALSE]
 
        # Apply the function
        return(tmle_rules[[rule]](row_df))
      }))
    }
    
    # Combine all batch results
    shifted <- do.call(rbind, batch_results)
    
    newdata <- cbind(data,shifted)
    assign(paste0("Q_",rule), lstm(data=newdata[c(grep("Y",colnames(newdata),value = TRUE),tmle_covars_Y)], outcome=grep("Y",colnames(newdata),value = TRUE), covariates=tmle_covars_Y, t_end=t.end, window_size=window.size, out_activation="sigmoid", loss_fn = "binary_crossentropy", output_dir, inference=TRUE))
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

TMLE_IC_lstm <- function(tmle_contrasts, initial_model_for_Y, time.censored, alpha=0.05, iptw=FALSE, gcomp=FALSE){
  
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