###################################################################
# Influence Curve                                                 #
###################################################################

TMLE_IC <- function(tmle_contrasts, initial_model_for_Y, time.censored, alpha=0.05, iptw=FALSE, gcomp=FALSE, estimator="tmle") {
  # Get end time point safely
  t.end <- min(length(tmle_contrasts), length(initial_model_for_Y))
  if(t.end < 1) {
    stop("No valid time points found in input")
  }
  
  # Get number of rules safely
  n.rules <- if(estimator=="tmle-lstm") {
    if(!is.null(tmle_contrasts[[t.end]]$Qstar)) {
      ncol(tmle_contrasts[[t.end]]$Qstar)
    } else if(!is.null(tmle_contrasts[[t.end]]$Qstar_gcomp)) {
      ncol(tmle_contrasts[[t.end]]$Qstar_gcomp)
    } else if(!is.null(tmle_contrasts[[t.end]]$Qstar_iptw)) {
      ncol(tmle_contrasts[[t.end]]$Qstar_iptw)
    } else {
      3  # Default
    }
  } else {
    ncol(tmle_contrasts[[t.end]]$weights)
  }
  
  # Calculate final TMLE estimate
  tmle_final <- if(estimator=="tmle-lstm") {
    lapply(1:t.end, function(t) {
      if(iptw && !is.null(tmle_contrasts[[t]]$Qstar_iptw)) {
        sapply(1:n.rules, function(x) {
          mean(tmle_contrasts[[t]]$Qstar_iptw[,x], na.rm=TRUE)
        })
      } else if(gcomp && !is.null(tmle_contrasts[[t]]$Qstar_gcomp)) {
        sapply(1:n.rules, function(x) {
          mean(tmle_contrasts[[t]]$Qstar_gcomp[,x], na.rm=TRUE)
        })
      } else if(!is.null(tmle_contrasts[[t]]$Qstar)) {
        sapply(1:n.rules, function(x) {
          mean(tmle_contrasts[[t]]$Qstar[,x], na.rm=TRUE)
        })
      } else {
        rep(NA, n.rules)
      }
    })
  } else {
    lapply(1:t.end, function(t) {
      if(iptw) {
        sapply(1:n.rules, function(x) {
          if(t < t.end) {
            mean(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE)
          } else {
            mean(tmle_contrasts[[t]]$Qstar_iptw[[x]], na.rm=TRUE)
          }
        })
      } else if(gcomp) {
        sapply(1:n.rules, function(x) {
          if(t < t.end) {
            mean(tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]], na.rm=TRUE)
          } else {
            mean(tmle_contrasts[[t]]$Qstar_gcomp[[x]], na.rm=TRUE)
          }
        })
      } else {
        sapply(1:n.rules, function(x) {
          if(t < t.end) {
            mean(tmle_contrasts[[t]][,x]$Qstar[[x]], na.rm=TRUE)
          } else {
            mean(tmle_contrasts[[t]]$Qstar[[x]], na.rm=TRUE)
          }
        })
      }
    })
  }
  
  # Add names to estimates
  for(t in seq_along(tmle_final)) {
    names(tmle_final[[t]]) <- if(estimator=="tmle-lstm") {
      c("static", "dynamic", "stochastic")
    } else {
      if(!is.null(tmle_contrasts[[t.end]]$weights)) {
        colnames(tmle_contrasts[[t.end]]$weights)
      } else {
        paste0("rule", 1:n.rules)
      }
    }
  }
  
  # Get uncensored outcomes
  Y_uncensored <- lapply(1:t.end, function(t) {
    if(estimator=="tmle-lstm") {
      if(!is.null(initial_model_for_Y$data$Y)) {
        y_vals <- initial_model_for_Y$data$Y
        if(!is.null(time.censored)) {
          censored_ids <- time.censored$ID[which(time.censored$time_censored < (t+1))]
          y_vals[initial_model_for_Y$data$ID %in% censored_ids] <- NA
        }
        y_vals
      } else {
        rep(NA, n.rules)
      }
    } else {
      # Original version logic
      if(t == t.end) {
        y_vals <- initial_model_for_Y[[t]]$data$Y
      } else {
        y_vals <- initial_model_for_Y[[t]][,1]$data$Y
      }
      if(!is.null(time.censored)) {
        censored_ids <- time.censored$ID[which(time.censored$time_censored < (t+1))]
        y_vals[initial_model_for_Y[[t]][,1]$data$ID %in% censored_ids] <- NA
      }
      y_vals
    }
  })
  
  # Handle missing values
  Y_uncensored <- lapply(Y_uncensored, function(y) {
    y[is.na(y)] <- 0
    y
  })
  
  # Calculate influence curves
  infcurv <- if(estimator=="tmle-lstm") {
    calc_ic_lstm(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, 
                 time.censored, iptw, gcomp)
  } else {
    calc_ic_original(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, 
                     time.censored, iptw, gcomp)
  }
  
  # Calculate CIs for time points 2:t.end
  CIs <- lapply(2:t.end, function(t) {
    if(is.null(infcurv[[t]]) || is.null(tmle_final[[t]])) {
      matrix(NA, nrow=2, ncol=n.rules)
    } else {
      sapply(1:n.rules, function(x) {
        if(ncol(infcurv[[t]]) < x || length(tmle_final[[t]]) < x) {
          c(NA, NA)
        } else {
          CI(est=tmle_final[[t]][x], infcurv=infcurv[[t]][,x], alpha=alpha)
        }
      })
    }
  })
  names(CIs) <- paste0("t=", 2:t.end)
  
  # Calculate variances
  vars <- lapply(2:t.end, function(t) {
    if(is.null(infcurv[[t]])) {
      rep(NA, n.rules)
    } else {
      sapply(1:n.rules, function(x) {
        if(ncol(infcurv[[t]]) < x) {
          NA
        } else {
          var(infcurv[[t]][,x], na.rm=TRUE)
        }
      })
    }
  })
  names(vars) <- paste0("t=", 2:t.end)
  
  # Add column names
  for(t in 2:(t.end-1)) {
    if(!is.null(CIs[[t-1]])) {
      colnames(CIs[[t-1]]) <- if(estimator=="tmle-lstm") {
        c("static", "dynamic", "stochastic")
      } else {
        colnames(tmle_contrasts[[t.end]]$weights)
      }
    }
    if(!is.null(vars[[t-1]])) {
      names(vars[[t-1]]) <- if(estimator=="tmle-lstm") {
        c("static", "dynamic", "stochastic")
      } else {
        colnames(tmle_contrasts[[t.end]]$weights)  
      }
    }
  }
  
  return(list(infcurv=infcurv, CI=CIs, var=vars, est=tmle_final))
}

calc_ic_lstm <- function(tmle_contrasts, tmle_final, Y_uncensored, t.end, n.rules, time.censored, iptw, gcomp) {
  # Initialize influence curves for ALL time points
  infcurv <- vector("list", t.end)
  
  # Helper function to ensure vectors match length
  ensure_length <- function(vec, target_length, default = 0) {
    if(length(vec) == 0) return(rep(default, target_length))
    if(length(vec) < target_length) {
      return(rep(vec, length.out = target_length))
    }
    if(length(vec) > target_length) {
      return(vec[1:target_length])
    }
    return(vec)
  }
  
  # Initialize all matrices first
  for(t in 1:t.end) {
    n_samples <- if(!is.null(Y_uncensored[[t]])) {
      length(Y_uncensored[[t]])
    } else {
      length(Y_uncensored[[1]])  # Fallback to first time point
    }
    infcurv[[t]] <- matrix(0, nrow=n_samples, ncol=n.rules)
    colnames(infcurv[[t]]) <- c("static", "dynamic", "stochastic")
  }
  
  # Calculate influence curves for t.end down to 2
  for(t in t.end:2) {
    curr_samples <- nrow(infcurv[[t]])
    
    for(x in 1:n.rules) {
      # Get predictions with dimension check
      qstar <- if(gcomp) {
        ensure_length(tmle_contrasts[[t]]$Qstar_gcomp[,x], curr_samples)
      } else if(iptw) {
        ensure_length(tmle_contrasts[[t]]$Qstar_iptw[,x], curr_samples)
      } else {
        ensure_length(tmle_contrasts[[t]]$Qstar[,x], curr_samples)
      }
      
      # Calculate centering term
      centering <- tmle_final[[t]][x]
      if(length(centering) != 1) centering <- mean(centering, na.rm=TRUE)
      
      # Calculate main influence curve component
      # Ensure lengths match
      infcurv[[t]][,x] <- ensure_length(qstar - centering, curr_samples)
      
      # Add score contribution if t < t.end
      if(t < t.end) {
        # Get weights
        weights <- if(!is.null(tmle_contrasts[[t]]$weights)) {
          ensure_length(tmle_contrasts[[t]]$weights[,x], curr_samples)
        } else {
          rep(1, curr_samples)
        }
        
        # Get previous predictions
        if(t > 1) {
          qstar_prev <- if(gcomp) {
            ensure_length(tmle_contrasts[[t-1]]$Qstar_gcomp[,x], curr_samples)
          } else if(iptw) {
            ensure_length(tmle_contrasts[[t-1]]$Qstar_iptw[,x], curr_samples)
          } else {
            ensure_length(tmle_contrasts[[t-1]]$Qstar[,x], curr_samples)
          }
          
          # Add score term
          y_curr <- ensure_length(Y_uncensored[[t]], curr_samples)
          score <- weights * (y_curr - qstar_prev)
          infcurv[[t]][,x] <- infcurv[[t]][,x] + ensure_length(score, curr_samples)
        }
      }
    }
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