# TMLE_IC function that computes estimates based on actual observed data only
TMLE_IC <- function(tmle_contrasts, initial_model_for_Y, time.censored=NULL, iptw=FALSE, gcomp=FALSE, 
                    estimator="tmle", basic_only=FALSE, variance_estimates=NULL, diagnostics=FALSE) {
  
  # Check if tmle_contrasts is valid
  if(is.null(tmle_contrasts) || length(tmle_contrasts) == 0) {
    message("tmle_contrasts is NULL or empty, cannot compute estimates without data")
    stop("No data available to compute estimates - provide valid tmle_contrasts")
  }
  
  # Extract estimates based on estimator type - with robust handling of NULL elements
  if(estimator=="tmle" || estimator=="tmle-lstm") {
    # Count how many time points we have with valid data
    valid_time_points <- 0
    for(t in 1:length(tmle_contrasts)) {
      if(!is.null(tmle_contrasts[[t]]) && 
         (!is.null(tmle_contrasts[[t]]$Qstar) || 
          !is.null(tmle_contrasts[[t]]$Qstar_iptw) || 
          !is.null(tmle_contrasts[[t]]$Qstar_gcomp))) {
        valid_time_points <- valid_time_points + 1
      }
    }
    
    message("Found ", valid_time_points, " valid time points for estimation")
    
    if(iptw) {
      # IPTW estimates
      est <- matrix(NA, nrow=length(tmle_contrasts), ncol=3)
      colnames(est) <- c("static", "dynamic", "stochastic")
      
      # Process each time point
      for(t in 1:length(tmle_contrasts)) {
        # Skip NULL elements
        if(is.null(tmle_contrasts[[t]])) {
          next
        }
        
        # Extract IPTW estimates
        if(!is.null(tmle_contrasts[[t]]$Qstar_iptw)) {
          if(is.vector(tmle_contrasts[[t]]$Qstar_iptw)) {
            for(i in 1:min(3, length(tmle_contrasts[[t]]$Qstar_iptw))) {
              # Values are event probabilities, convert to survival probabilities
              est[t,i] <- 1 - tmle_contrasts[[t]]$Qstar_iptw[i]
            }
          } else if(is.matrix(tmle_contrasts[[t]]$Qstar_iptw)) {
            for(i in 1:min(3, ncol(tmle_contrasts[[t]]$Qstar_iptw))) {
              # Values are event probabilities, convert to survival probabilities
              est[t,i] <- 1 - mean(tmle_contrasts[[t]]$Qstar_iptw[,i], na.rm=TRUE)
            }
          }
        }
      }
    } else if(gcomp) {
      # G-computation estimates
      est <- matrix(NA, nrow=length(tmle_contrasts), ncol=3)
      colnames(est) <- c("static", "dynamic", "stochastic")
      
      # Process each time point
      for(t in 1:length(tmle_contrasts)) {
        # Skip NULL elements
        if(is.null(tmle_contrasts[[t]])) {
          next
        }
        
        # Extract G-computation estimates
        if(!is.null(tmle_contrasts[[t]]$Qstar_gcomp)) {
          if(is.vector(tmle_contrasts[[t]]$Qstar_gcomp)) {
            for(i in 1:min(3, length(tmle_contrasts[[t]]$Qstar_gcomp))) {
              # Values are event probabilities, convert to survival probabilities
              est[t,i] <- 1 - tmle_contrasts[[t]]$Qstar_gcomp[i]
            }
          } else if(is.matrix(tmle_contrasts[[t]]$Qstar_gcomp)) {
            for(i in 1:min(3, ncol(tmle_contrasts[[t]]$Qstar_gcomp))) {
              # Values are event probabilities, convert to survival probabilities
              est[t,i] <- 1 - mean(tmle_contrasts[[t]]$Qstar_gcomp[,i], na.rm=TRUE)
            }
          }
        }
      }
    } else {
      # TMLE estimates
      est <- matrix(NA, nrow=length(tmle_contrasts), ncol=3)
      colnames(est) <- c("static", "dynamic", "stochastic")
      
      # Process each time point
      for(t in 1:length(tmle_contrasts)) {
        # Skip NULL elements
        if(is.null(tmle_contrasts[[t]])) {
          next
        }
        
        # Extract TMLE estimates
        # First try Qstar
        if(!is.null(tmle_contrasts[[t]]$Qstar)) {
          # Add diagnostic information about the Qstar values
          if(diagnostics) {
            message(paste0("Processing Qstar values for time point ", t))
            if(is.vector(tmle_contrasts[[t]]$Qstar)) {
              message(paste0("Qstar is a vector of length ", length(tmle_contrasts[[t]]$Qstar)))
              message(paste0("Sample Qstar values: ", paste(head(tmle_contrasts[[t]]$Qstar), collapse=", ")))
            } else if(is.matrix(tmle_contrasts[[t]]$Qstar)) {
              message(paste0("Qstar is a matrix: ", nrow(tmle_contrasts[[t]]$Qstar), "x", ncol(tmle_contrasts[[t]]$Qstar)))
              if(nrow(tmle_contrasts[[t]]$Qstar) > 0) {
                message(paste0("Mean Qstar values by column: ", paste(colMeans(tmle_contrasts[[t]]$Qstar, na.rm=TRUE), collapse=", ")))
              }
            }
          }
          
          if(is.vector(tmle_contrasts[[t]]$Qstar)) {
            for(i in 1:min(3, length(tmle_contrasts[[t]]$Qstar))) {
              # Values are event probabilities, convert to survival probabilities
              qstar_value <- tmle_contrasts[[t]]$Qstar[i]
              
              # Log the actual event probability value for diagnostic purposes
              if(qstar_value > 0.6) {
                message(paste0("WARNING: High Qstar (event probability) value at t=", t, ", rule=", i, ": ", qstar_value))
              } else if(qstar_value < 0.01) {
                message(paste0("WARNING: Very low Qstar (event probability) value at t=", t, ", rule=", i, ": ", qstar_value))
              } else {
                message(paste0("Normal Qstar (event probability) value at t=", t, ", rule=", i, ": ", qstar_value))
              }
              
              # Convert event probability to survival probability (1 - event_prob)
              est[t,i] <- 1 - qstar_value
            }
          } else if(is.matrix(tmle_contrasts[[t]]$Qstar)) {
            for(i in 1:min(3, ncol(tmle_contrasts[[t]]$Qstar))) {
              # Values are event probabilities, convert to survival probabilities
              mean_qstar <- mean(tmle_contrasts[[t]]$Qstar[,i], na.rm=TRUE)
              
              # Log the actual event probability value for diagnostic purposes
              if(mean_qstar > 0.6) {
                message(paste0("WARNING: High mean Qstar (event probability) value at t=", t, ", rule=", i, ": ", mean_qstar))
              } else if(mean_qstar < 0.01) {
                message(paste0("WARNING: Very low mean Qstar (event probability) value at t=", t, ", rule=", i, ": ", mean_qstar))
              } else {
                message(paste0("Normal mean Qstar (event probability) value at t=", t, ", rule=", i, ": ", mean_qstar))
              }
              
              # Convert event probability to survival probability (1 - event_prob)
              est[t,i] <- 1 - mean_qstar
            }
          }
        } 
        # If Qstar is not available, try using Qstar_gcomp as a fallback
        else if(!is.null(tmle_contrasts[[t]]$Qstar_gcomp)) {
          # Add diagnostic information
          if(diagnostics) {
            message(paste0("Using Qstar_gcomp fallback for time point ", t))
            if(is.vector(tmle_contrasts[[t]]$Qstar_gcomp)) {
              message(paste0("Qstar_gcomp values: ", paste(tmle_contrasts[[t]]$Qstar_gcomp, collapse=", ")))
            } else if(is.matrix(tmle_contrasts[[t]]$Qstar_gcomp)) {
              message(paste0("Qstar_gcomp summary - rows: ", nrow(tmle_contrasts[[t]]$Qstar_gcomp), 
                         ", cols: ", ncol(tmle_contrasts[[t]]$Qstar_gcomp)))
            }
          }
          
          if(is.vector(tmle_contrasts[[t]]$Qstar_gcomp)) {
            for(i in 1:min(3, length(tmle_contrasts[[t]]$Qstar_gcomp))) {
              # Values are event probabilities, convert to survival probabilities
              gcomp_value <- tmle_contrasts[[t]]$Qstar_gcomp[i]
              
              # Check for extreme values and just log them without modifying
              if(gcomp_value > 0.95) {
                message(paste0("WARNING: High G-comp value at t=", t, ", rule=", i, ": ", gcomp_value))
                # No longer adjusting extreme values - using original values
              }
              
              est[t,i] <- 1 - gcomp_value
            }
          } else if(is.matrix(tmle_contrasts[[t]]$Qstar_gcomp)) {
            for(i in 1:min(3, ncol(tmle_contrasts[[t]]$Qstar_gcomp))) {
              # Values are event probabilities, convert to survival probabilities
              mean_gcomp <- mean(tmle_contrasts[[t]]$Qstar_gcomp[,i], na.rm=TRUE)
              
              # Check for extreme values and just log them without modifying
              if(mean_gcomp > 0.95) {
                message(paste0("WARNING: High mean G-comp value at t=", t, ", rule=", i, ": ", mean_gcomp))
                # No longer adjusting extreme values - using original values
              }
              
              est[t,i] <- 1 - mean_gcomp
            }
          }
        }
      }
    }
  } else {
    # For other estimators, create matrix with genuine NA values
    est <- matrix(NA, nrow=length(tmle_contrasts), ncol=3)
    colnames(est) <- c("static", "dynamic", "stochastic")
  }
  
  # Check if we have any estimates at all
  if(is.null(est) || (is.matrix(est) && all(is.na(est)))) {
    message("No valid estimates found for ", estimator)
    
    # For TMLE estimator, try to use G-computation or IPTW as fallback
    if(estimator == "tmle" || estimator == "tmle-lstm") {
      message("Trying to use G-computation estimates as fallback")
      
      # Create G-computation estimates
      est_gcomp <- matrix(NA, nrow=length(tmle_contrasts), ncol=3)
      colnames(est_gcomp) <- c("static", "dynamic", "stochastic")
      
      # Process each time point
      for(t in 1:length(tmle_contrasts)) {
        # Skip NULL elements
        if(is.null(tmle_contrasts[[t]])) {
          next
        }
        
        # Extract G-computation estimates
        if(!is.null(tmle_contrasts[[t]]$Qstar_gcomp)) {
          if(is.vector(tmle_contrasts[[t]]$Qstar_gcomp)) {
            for(i in 1:min(3, length(tmle_contrasts[[t]]$Qstar_gcomp))) {
              # Values are event probabilities, convert to survival probabilities
              est_gcomp[t,i] <- 1 - tmle_contrasts[[t]]$Qstar_gcomp[i]
            }
          } else if(is.matrix(tmle_contrasts[[t]]$Qstar_gcomp)) {
            for(i in 1:min(3, ncol(tmle_contrasts[[t]]$Qstar_gcomp))) {
              # Values are event probabilities, convert to survival probabilities
              est_gcomp[t,i] <- 1 - mean(tmle_contrasts[[t]]$Qstar_gcomp[,i], na.rm=TRUE)
            }
          }
        }
      }
      
      # Check if G-computation has values
      if(!all(is.na(est_gcomp))) {
        message("Using G-computation estimates as fallback")
        est <- est_gcomp
      } else {
        message("No G-computation values either, keeping NAs")
      }
    }
  }
  
  # Do not replace NA values - maintain NA for missing data
  # This ensures we don't introduce artificial values where data is missing
  
  # Calculate standard errors only when data is available
  se_list <- lapply(1:nrow(est), function(t) {
    se_vals <- rep(NA, ncol(est))
    # For each rule/column
    for(i in 1:ncol(est)) {
      # Skip if we don't have data at this time point
      if(is.null(tmle_contrasts[[t]]) || 
         is.null(tmle_contrasts[[t]]$Qstar) || 
         !is.matrix(tmle_contrasts[[t]]$Qstar) ||
         ncol(tmle_contrasts[[t]]$Qstar) < i) {
        # Keep as NA to indicate no data
        next
      }
      
      # Get values for this rule
      values <- tmle_contrasts[[t]]$Qstar[, i]
      valid_values <- values[!is.na(values) & is.finite(values) & values != -1]
      
      if(length(valid_values) > 0) {
        # Calculate standard error from the data
        se_vals[i] <- sd(valid_values, na.rm=TRUE) / sqrt(length(valid_values))
        
        # Keep NA if calculation fails
        if(!is.finite(se_vals[i])) {
          se_vals[i] <- NA
        }
      }
      # If no valid values, keep it as NA
    }
    se_vals
  })
  
  # Compute confidence intervals only when both estimates and SEs are available
  CI <- lapply(1:length(se_list), function(t) {
    ci_mat <- matrix(NA, nrow=2, ncol=ncol(est))
    colnames(ci_mat) <- colnames(est)
    
    for(i in 1:ncol(est)) {
      # Only compute CI when both estimate and SE are valid
      if(!is.na(est[t,i]) && !is.na(se_list[[t]][i])) {
        ci_mat[1,i] <- est[t,i] - 1.96 * se_list[[t]][i]  # Lower bound
        ci_mat[2,i] <- est[t,i] + 1.96 * se_list[[t]][i]  # Upper bound
      }
      # Otherwise leave as NA
    }
    ci_mat
  })
  
  # Add diagnostics info if requested
  if(diagnostics) {
    # Extract raw Qstar values for inspection
    qstar_values <- list()
    qstar_iptw_values <- list()
    qstar_gcomp_values <- list()
    
    for(t in 1:length(tmle_contrasts)) {
      if(!is.null(tmle_contrasts[[t]])) {
        # Extract TMLE Qstar values
        if(!is.null(tmle_contrasts[[t]]$Qstar)) {
          qstar_values[[t]] <- tmle_contrasts[[t]]$Qstar
        }
        
        # Extract IPTW Qstar values
        if(!is.null(tmle_contrasts[[t]]$Qstar_iptw)) {
          qstar_iptw_values[[t]] <- tmle_contrasts[[t]]$Qstar_iptw
        }
        
        # Extract G-comp Qstar values
        if(!is.null(tmle_contrasts[[t]]$Qstar_gcomp)) {
          qstar_gcomp_values[[t]] <- tmle_contrasts[[t]]$Qstar_gcomp
        }
      }
    }
    
    # Return with diagnostic information
    return(list(
      "est" = est, 
      "CI" = CI, 
      "se" = se_list,
      "diagnostics" = list(
        "qstar_values" = qstar_values,
        "qstar_iptw_values" = qstar_iptw_values,
        "qstar_gcomp_values" = qstar_gcomp_values
      )
    ))
  } else {
    # Return standard output
    return(list("est"=est, "CI"=CI, "se"=se_list))
  }
}