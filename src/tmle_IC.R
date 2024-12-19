###################################################################
# Influence Curve                                                 #
###################################################################

TMLE_IC <- function(tmle_contrasts, initial_model_for_Y, time.censored, alpha=0.05, iptw=FALSE, gcomp=FALSE, estimator="tmle"){
  
  t.end <- length(initial_model_for_Y)
  n.rules <- if(estimator=="tmle-lstm") {
    # For LSTM case, get n.rules from Qstar dimensions
    if(!is.null(tmle_contrasts[[t.end]]$Qstar)) {
      ncol(tmle_contrasts[[t.end]]$Qstar)
    } else {
      3  # Default to 3 rules if Qstar not available
    }
  } else {
    # Original case
    ncol(tmle_contrasts[[t.end]]$weights)
  }
  
  # calcuate final TMLE estimate
  # Calculate final TMLE estimate based on estimator type
  tmle_final <- if(estimator=="tmle-lstm") {
    # LSTM version - handle matrix format
    if(iptw) {
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) {
        ifelse(t<t.end, 
               mean(tmle_contrasts[[t]]$Qstar_iptw[,x], na.rm=TRUE),
               mean(tmle_contrasts[[t]]$Qstar_iptw[,x], na.rm=TRUE))
      }))
    } else if(gcomp) {
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) {
        ifelse(t<t.end, 
               mean(tmle_contrasts[[t]]$Qstar_gcomp[,x], na.rm=TRUE),
               mean(tmle_contrasts[[t]]$Qstar_gcomp[,x], na.rm=TRUE))
      }))
    } else {
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) {
        ifelse(t<t.end, 
               mean(tmle_contrasts[[t]]$Qstar[,x], na.rm=TRUE),
               mean(tmle_contrasts[[t]]$Qstar[,x], na.rm=TRUE))
      }))
    }
  } else {
    # Original version
    if(iptw){
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) 
        ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE), 
               mean(tmle_contrasts[[t]]$Qstar_iptw[[x]], na.rm=TRUE))))
    } else if(gcomp){
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) 
        ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]], na.rm=TRUE), 
               mean(tmle_contrasts[[t]]$Qstar_gcomp[[x]], na.rm=TRUE))))
    } else{
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) 
        ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar[[x]], na.rm=TRUE), 
               mean(tmle_contrasts[[t]]$Qstar[[x]], na.rm=TRUE))))
    }
  }
  
  # Add names for both cases
  for(t in 1:t.end){
    names(tmle_final[[t]]) <- if(estimator=="tmle-lstm") {
      c("static", "dynamic", "stochastic")
    } else {
      colnames(tmle_contrasts[[t.end]]$weights)
    }
  }
  
  # Calculate influence curve for both cases
  Y_uncensored <- lapply(1:(t.end-1), function(t) {
    if(estimator=="tmle-lstm") {
      initial_model_for_Y$data$Y[!initial_model_for_Y$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]
    } else {
      initial_model_for_Y[[t]][,1]$data$Y[!initial_model_for_Y[[t]][,1]$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]
    }
  })
  Y_uncensored[[t.end]] <- if(estimator=="tmle-lstm") {
    initial_model_for_Y$data$Y[!initial_model_for_Y$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]
  } else {
    initial_model_for_Y[[t.end]]$data$Y[!initial_model_for_Y[[t.end]]$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]
  }
  
  # Handle missing values
  for(t in 1:t.end){
    Y_uncensored[[t]][is.na(Y_uncensored[[t]])] <- 0 
  }
  
  # Calculate influence curves
  infcurv <- if(estimator=="tmle-lstm") {
    calc_ic_lstm(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, time.censored, iptw, gcomp)
  } else {
    calc_ic_original(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, time.censored, iptw, gcomp)
  }
  
  # Calculate CIs and variances
  CIs <- lapply(2:t.end, function(t) {
    sapply(1:n.rules, function(x) CI(est=tmle_final[[t]][x], infcurv=infcurv[[t]][,x], alpha=alpha))
  })
  names(CIs) <- paste0("t=", 2:t.end)
  
  vars <- lapply(2:t.end, function(t) {
    sapply(1:n.rules, function(x) var(infcurv[[t]][,x], na.rm=TRUE))
  })
  names(vars) <- paste0("t=", 2:t.end)
  
  # Add column names
  for(t in 2:(t.end-1)){
    colnames(CIs[[t]]) <- if(estimator=="tmle-lstm") {
      c("static", "dynamic", "stochastic")
    } else {
      colnames(tmle_contrasts[[t.end]]$weights)
    }
    names(vars[[t]]) <- if(estimator=="tmle-lstm") {
      c("static", "dynamic", "stochastic")
    } else {
      colnames(tmle_contrasts[[t.end]]$weights)
    }
  }
  
  return(list("infcurv"=infcurv, "CI"=CIs, "var"=vars, "est"=tmle_final))
}

TMLE_IC <- function(tmle_contrasts, initial_model_for_Y, time.censored, alpha=0.05, iptw=FALSE, gcomp=FALSE, estimator="tmle"){
  
  t.end <- length(initial_model_for_Y)
  n.rules <- if(estimator=="tmle-lstm") {
    # For LSTM case, get n.rules from Qstar dimensions
    if(!is.null(tmle_contrasts[[t.end]]$Qstar)) {
      ncol(tmle_contrasts[[t.end]]$Qstar)
    } else {
      3  # Default to 3 rules if Qstar not available
    }
  } else {
    # Original case
    ncol(tmle_contrasts[[t.end]]$weights)
  }
  
  # Calculate final TMLE estimate based on estimator type
  tmle_final <- if(estimator=="tmle-lstm") {
    # LSTM version - handle matrix format
    if(iptw) {
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) {
        ifelse(t<t.end, 
               mean(tmle_contrasts[[t]]$Qstar_iptw[,x], na.rm=TRUE),
               mean(tmle_contrasts[[t]]$Qstar_iptw[,x], na.rm=TRUE))
      }))
    } else if(gcomp) {
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) {
        ifelse(t<t.end, 
               mean(tmle_contrasts[[t]]$Qstar_gcomp[,x], na.rm=TRUE),
               mean(tmle_contrasts[[t]]$Qstar_gcomp[,x], na.rm=TRUE))
      }))
    } else {
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) {
        ifelse(t<t.end, 
               mean(tmle_contrasts[[t]]$Qstar[,x], na.rm=TRUE),
               mean(tmle_contrasts[[t]]$Qstar[,x], na.rm=TRUE))
      }))
    }
  } else {
    # Original version
    if(iptw){
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) 
        ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE), 
               mean(tmle_contrasts[[t]]$Qstar_iptw[[x]], na.rm=TRUE))))
    } else if(gcomp){
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) 
        ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]], na.rm=TRUE), 
               mean(tmle_contrasts[[t]]$Qstar_gcomp[[x]], na.rm=TRUE))))
    } else{
      lapply(1:t.end, function(t) sapply(1:n.rules, function(x) 
        ifelse(t<t.end, mean(tmle_contrasts[[t]][,x]$Qstar[[x]], na.rm=TRUE), 
               mean(tmle_contrasts[[t]]$Qstar[[x]], na.rm=TRUE))))
    }
  }
  
  # Add names for both cases
  for(t in 1:t.end){
    names(tmle_final[[t]]) <- if(estimator=="tmle-lstm") {
      c("static", "dynamic", "stochastic")
    } else {
      colnames(tmle_contrasts[[t.end]]$weights)
    }
  }
  
  # Calculate influence curve for both cases
  Y_uncensored <- lapply(1:(t.end-1), function(t) {
    if(estimator=="tmle-lstm") {
      initial_model_for_Y$data$Y[!initial_model_for_Y$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]
    } else {
      initial_model_for_Y[[t]][,1]$data$Y[!initial_model_for_Y[[t]][,1]$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]
    }
  })
  Y_uncensored[[t.end]] <- if(estimator=="tmle-lstm") {
    initial_model_for_Y$data$Y[!initial_model_for_Y$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]
  } else {
    initial_model_for_Y[[t.end]]$data$Y[!initial_model_for_Y[[t.end]]$data$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]
  }
  
  # Handle missing values
  for(t in 1:t.end){
    Y_uncensored[[t]][is.na(Y_uncensored[[t]])] <- 0 
  }
  
  # Calculate influence curves
  infcurv <- if(estimator=="tmle-lstm") {
    calc_ic_lstm(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, time.censored, iptw, gcomp)
  } else {
    calc_ic_original(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, time.censored, iptw, gcomp)
  }
  
  # Calculate CIs and variances
  CIs <- lapply(2:t.end, function(t) {
    sapply(1:n.rules, function(x) CI(est=tmle_final[[t]][x], infcurv=infcurv[[t]][,x], alpha=alpha))
  })
  names(CIs) <- paste0("t=", 2:t.end)
  
  vars <- lapply(2:t.end, function(t) {
    sapply(1:n.rules, function(x) var(infcurv[[t]][,x], na.rm=TRUE))
  })
  names(vars) <- paste0("t=", 2:t.end)
  
  # Add column names
  for(t in 2:(t.end-1)){
    colnames(CIs[[t]]) <- if(estimator=="tmle-lstm") {
      c("static", "dynamic", "stochastic")
    } else {
      colnames(tmle_contrasts[[t.end]]$weights)
    }
    names(vars[[t]]) <- if(estimator=="tmle-lstm") {
      c("static", "dynamic", "stochastic")
    } else {
      colnames(tmle_contrasts[[t.end]]$weights)
    }
  }
  
  return(list("infcurv"=infcurv, "CI"=CIs, "var"=vars, "est"=tmle_final))
}

# Helper functions for influence curve calculations
calc_ic_lstm <- function(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, time.censored, iptw, gcomp) {
  infcurv <- vector("list", t.end)
  
  # Calculate influence curves for each time point
  for(t in t.end:2) {
    infcurv[[t]] <- matrix(0, nrow=length(Y_uncensored[[t]]), ncol=n.rules)
    for(x in 1:n.rules) {
      # Get Qstar predictions based on estimator type
      if(gcomp) {
        qstar <- tmle_contrasts[[t]]$Qstar_gcomp[,x]
        qstar_prev <- if(t > 1) tmle_contrasts[[t-1]]$Qstar_gcomp[,x] else NULL
      } else if(iptw) {
        qstar <- tmle_contrasts[[t]]$Qstar_iptw[,x]
        qstar_prev <- if(t > 1) tmle_contrasts[[t-1]]$Qstar_iptw[,x] else NULL
      } else {
        qstar <- tmle_contrasts[[t]]$Qstar[,x]
        qstar_prev <- if(t > 1) tmle_contrasts[[t-1]]$Qstar[,x] else NULL
      }
      
      # Calculate influence curve components
      ic_components <- qstar - tmle_final[[t]][x]
      if(t < t.end) {
        ic_components <- ic_components + 
          tmle_contrasts[[t]]$weights[,x] * (Y_uncensored[[t]] - qstar_prev)
      }
      
      infcurv[[t]][,x] <- ic_components
    }
    colnames(infcurv[[t]]) <- c("static", "dynamic", "stochastic")
  }
  
  return(infcurv)
}

calc_ic_original <- function(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, time.censored, iptw, gcomp) {
  infcurv <- list()
  
  if(gcomp){
    # Last time period
    infcurv[[t.end]] <- sapply(1:n.rules, function(x) 
      tmle_contrasts[[t.end]]$Qstar_gcomp[[x]][!tmle_contrasts[[t.end]]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_final[[t.end]][[x]] +
        (Y_uncensored[[t.end]] - tmle_contrasts[[(t.end-1)]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) + 
        (tmle_contrasts[[2]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[1]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) +
        rowSums(sapply((t.end-1):2, function(t)
          (tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[(t-1)]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]])
        )))
    
    # Time periods 2 to t.end-1
    for(t in (t.end-1):2){
      infcurv[[t]] <- sapply(1:n.rules, function(x)
        tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_final[[t]][[x]] +
          (Y_uncensored[[t]] - tmle_contrasts[[(t-1)]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) +
          (tmle_contrasts[[2]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[1]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) +
          ifelse(t>2, rowSums(sapply((t-1):2, function(i)
            (tmle_contrasts[[i]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[i]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[(i-1)]][,x]$Qstar_gcomp[[x]][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]])
          )), 0))
    }
    
  }else if(iptw){
    # Last time period
    infcurv[[t.end]] <- sapply(1:n.rules, function(x) 
      tmle_contrasts[[t.end]]$Qstar_iptw[[x]][!tmle_contrasts[[t.end]]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_final[[t.end]][[x]] +
        tmle_contrasts[[(t.end-1)]][,x]$weights[,x][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] * (Y_uncensored[[t.end]] - tmle_contrasts[[(t.end-1)]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) +
        tmle_contrasts[[1]][,x]$weights[,x][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] * (tmle_contrasts[[2]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[1]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) +
        rowSums(sapply((t.end-1):2, function(t)
          tmle_contrasts[[(t-1)]][,x]$weights[,x][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] * (tmle_contrasts[[t]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[(t-1)]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]])
        )))
    
    # Time periods 2 to t.end-1
    for(t in (t.end-1):2){
      infcurv[[t]] <- sapply(1:n.rules, function(x)
        tmle_contrasts[[t]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_final[[t]][[x]] +
          tmle_contrasts[[(t-1)]][,x]$weights[,x][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] * (Y_uncensored[[t]] - tmle_contrasts[[(t-1)]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) +
          tmle_contrasts[[1]][,x]$weights[,x][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] * (tmle_contrasts[[2]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[1]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) +
          ifelse(t>2, rowSums(sapply((t-1):2, function(i)
            tmle_contrasts[[(i-1)]][,x]$weights[,x][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] * (tmle_contrasts[[i]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[i]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[(i-1)]][,x]$Qstar_iptw[[x]][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]])
          )), 0))
    }
    
  }else{
    # Last time period
    infcurv[[t.end]] <- sapply(1:n.rules, function(x) 
      tmle_contrasts[[t.end]]$Qstar[[x]][!tmle_contrasts[[t.end]]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_final[[t.end]][[x]] +
        tmle_contrasts[[(t.end-1)]][,x]$weights[,x][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] * (Y_uncensored[[t.end]] - tmle_contrasts[[(t.end-1)]][,x]$Qstar[[x]][!tmle_contrasts[[(t.end-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) +
        tmle_contrasts[[1]][,x]$weights[,x][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] * (tmle_contrasts[[2]][,x]$Qstar[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[1]][,x]$Qstar[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]]) +
        rowSums(sapply((t.end-1):2, function(t)
          tmle_contrasts[[(t-1)]][,x]$weights[,x][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] * (tmle_contrasts[[t]][,x]$Qstar[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]] - tmle_contrasts[[(t-1)]][,x]$Qstar[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t.end+1))]])
        )))
    
    # Time periods 2 to t.end-1
    for(t in (t.end-1):2){
      infcurv[[t]] <- sapply(1:n.rules, function(x)
        tmle_contrasts[[t]][,x]$Qstar[[x]][!tmle_contrasts[[t]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_final[[t]][[x]] +
          tmle_contrasts[[(t-1)]][,x]$weights[,x][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] * (Y_uncensored[[t]] - tmle_contrasts[[(t-1)]][,x]$Qstar[[x]][!tmle_contrasts[[(t-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) +
          tmle_contrasts[[1]][,x]$weights[,x][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] * (tmle_contrasts[[2]][,x]$Qstar[[x]][!tmle_contrasts[[2]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[1]][,x]$Qstar[[x]][!tmle_contrasts[[1]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]]) +
          ifelse(t>2, rowSums(sapply((t-1):2, function(i)
            tmle_contrasts[[(i-1)]][,x]$weights[,x][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] * (tmle_contrasts[[i]][,x]$Qstar[[x]][!tmle_contrasts[[i]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]] - tmle_contrasts[[(i-1)]][,x]$Qstar[[x]][!tmle_contrasts[[(i-1)]][,x]$ID %in% time.censored$ID[which(time.censored$time_censored<(t+1))]])
          )), 0))
    }
  }
  
  # Add column names
  for(t in 2:t.end){
    colnames(infcurv[[t]]) <- colnames(tmle_contrasts[[t.end]]$weights)
  }
  
  return(infcurv)
}