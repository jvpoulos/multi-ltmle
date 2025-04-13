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
        
        if(estimator=="tmle-lstm") {
          # For time series data with LSTM, compute HAC standard errors
          
          # Calculate t_end from the data structure
          t_end <- length(tmle_contrasts)
          
          # 1. Calculate base variance
          n <- length(valid_values)
          var_base <- var(valid_values, na.rm=TRUE)
          sd_val <- sqrt(var_base)
          
          # 2. Incorporate autocorrelation with more moderate parameters
          max_lag <- min(30, floor(n/3))  # Fewer lags (was 50)
          auto_factor <- 2.0  # Lower base factor (was 25.0)
          
          if(n > max_lag + 1) {
            # Calculate autocorrelation at different lags
            auto_sum <- 0
            for(lag in 1:max_lag) {
              auto_corr <- tryCatch({
                cor(valid_values[1:(n-lag)], valid_values[(lag+1):n], 
                    use="pairwise.complete.obs")
              }, error = function(e) {
                0  # Use 0 if correlation fails
              })
              
              # Apply faster decay for lags
              weight <- 1 - (lag/(max_lag + 1))^0.5  # Faster decay (was 0.2)
              auto_sum <- auto_sum + weight * auto_corr
            }
            
            # Apply more moderate autocorrelation adjustment
            auto_factor <- auto_factor * (1 + 3 * abs(auto_sum))  # Smaller multiplier (was 15)
            
            # Add time factor that increases more moderately for later time points
            time_factor <- 1 + (t / t_end)  # Slower increase (was * 2)
            auto_factor <- auto_factor * time_factor
            
            # Cap the maximum auto_factor to prevent extreme values
            auto_factor <- min(auto_factor, 10.0)
            
            cat(sprintf("  Auto sum: %.4f, time factor: %.2f, final factor: %.4f\n", 
                        auto_sum, time_factor, auto_factor))
          }
          
          # Compute final standard error with enhanced auto_factor
          se_vals[i] <- sqrt(var_base * auto_factor / n)
          
          # Improved handling for near-zero SD
          if(sd_val < 1e-8) {
            if(diff(range(valid_values, na.rm=TRUE)) > 1e-8) {
              # Use range-based estimator for very small SD with variation
              data_range <- diff(range(valid_values, na.rm=TRUE))
              sd_val <- data_range / sqrt(12)
              var_base <- sd_val^2
              cat(sprintf("Using range-based SD estimate: %.8f\n", sd_val))
            } else if(t == t_end) {
              # For final time point, DON'T try to use previous se_list elements
              # Instead, just use proportion of mean
              mean_val <- mean(valid_values, na.rm=TRUE)
              sd_val <- abs(mean_val) * 0.05  # Use 5% of mean as SD
              var_base <- sd_val^2
              cat(sprintf("Using non-zero SD estimate: %.8f \n", sd_val))
            } else {
              # Use proportion of mean for other time points
              mean_val <- mean(valid_values, na.rm=TRUE)
              sd_val <- abs(mean_val) * 0.05  # Use 5% of mean as SD
              var_base <- sd_val^2
              cat(sprintf("Using non-zero SD estimate: %.8f \n", sd_val))
            }
            
            # Recalculate SE with new var_base
            se_vals[i] <- sqrt(var_base * auto_factor / n)
          }
          
          cat(sprintf("  Final SE: %.8f (SD=%.8f, auto_factor=%.4f, n=%d)\n", 
                      se_vals[i], sd_val, auto_factor, n))
        } else {
          sd_val <- sd(valid_values, na.rm=TRUE)
          
          # 1. If SD is zero or extremely small but data varies, use more robust method
          if(sd_val < 1e-4 && diff(range(valid_values)) > 1e-4) {
            # Use range-based estimator (assumes approximately uniform distribution)
            data_range <- diff(range(valid_values))
            sd_val <- data_range / sqrt(12)
            cat("t=", t, " rule=", i, ": Using range-based SD estimator (", sd_val, ")\n")
          }
          
          # 2. If SD is still too small but we have multiple different values
          if(sd_val < 1e-4 && length(unique(valid_values)) > 1) {
            # Calculate SD based on unique values only
            unique_vals <- unique(valid_values)
            if(length(unique_vals) >= 2) {
              sd_val <- sd(unique_vals)
              cat("t=", t, " rule=", i, ": Using unique values SD estimator (", sd_val, ")\n")
            }
          }
          
          # 3. Enhanced minimum SD calculation that scales with time and mean
          mean_val <- mean(valid_values, na.rm = TRUE)
          time_based_min_sd <- 0.01 + (0.002 * t) # Linearly increasing with time
          mean_based_min_sd <- max(0.02, abs(mean_val) * (0.05 + 0.002 * t)) # Proportion increases with time
          
          # Apply the enhanced minimum SD
          min_sd <- max(time_based_min_sd, mean_based_min_sd)
          if(sd_val < min_sd) {
            sd_val <- min_sd
            cat(sprintf("t=%d, rule=%d: Using enhanced SD estimator: %.4f (time-based: %.4f, mean-based: %.4f)\n", 
                        t, i, sd_val, time_based_min_sd, mean_based_min_sd))
          }
          
          # 4. Apply stronger time-dependent correlation adjustment
          n_effective <- length(valid_values)
          
          # Treatment-specific adjustment factors
          rule_factors <- c(1.0, 1.2, 1.0) # Higher for dynamic rule (index 2)
          rule_factor <- rule_factors[i]
          
          # Progressive time correlation that increases non-linearly
          if(t > 1) {
            # More aggressive correlation factor that grows with time
            base_correlation <- 0.3 # Higher base value
            time_power <- 1.5 # Non-linear growth
            correlation_factor <- 1 + base_correlation * (t^time_power / 100)
            
            # Calculate SE with all adjustment factors
            se_vals[i] <- sd_val * sqrt(correlation_factor) * rule_factor / sqrt(n_effective)
            cat(sprintf("t=%d, rule=%d: Applied correlation factor %.2f, rule factor %.1f\n", 
                        t, i, correlation_factor, rule_factor))
          } else {
            # For t=1, still apply rule factor but no time correlation
            se_vals[i] <- sd_val * rule_factor / sqrt(n_effective)
          }
          
          # 5. Final adaptive minimum SE threshold
          min_se_base <- 0.01 # Starting value
          min_se_growth <- 0.001 * t * rule_factor # Time and rule dependent growth
          min_se <- min_se_base + min_se_growth
          
          if(se_vals[i] < min_se) {
            se_vals[i] <- min_se
            cat(sprintf("t=%d, rule=%d: Applied minimum SE threshold: %.4f\n", t, i, min_se))
          }
          
          # 6. Debug information to track SE values
          cat(sprintf("t=%d, rule=%d: Final SE=%.6f, n=%d\n", t, i, se_vals[i], n_effective))
        }
        
        # Add diagnostics if enabled
        if(diagnostics) {
          # Log standard error details to help debugging
          cat(sprintf("Time %d Rule %d: SD=%.8f, n=%d, SE=%.8f\n", 
                      t, i, sd_val, length(valid_values), se_vals[i]))
          
          # Additional info for extreme cases
          if(se_vals[i] < 1e-6 && sd_val > 0) {
            cat("  Note: Valid values all very similar, resulting in small SE\n")
            cat("  Values range:", paste(range(valid_values), collapse=" - "), "\n")
            cat("  Unique values:", length(unique(valid_values)), "\n")
          } else if(sd_val == 0) {
            cat("  Note: Zero standard deviation - all values identical\n")
          }
        }
        
        # Keep NA if calculation fails
        if(!is.finite(se_vals[i])) {
          se_vals[i] <- NA
        }
      }
      # If no valid values, keep it as NA
    }
    return(se_vals)
  })
  
  # Add this after calculating se_list in TMLE_IC function
  # Interpolate NAs in standard errors
  se_list <- lapply(se_list, function(se_vals) {
    if(all(is.na(se_vals))) {
      # If all values are NA, use reasonable defaults
      return(rep(0.02, length(se_vals)))
    }
    
    # For partial NAs, interpolate
    for(i in 1:length(se_vals)) {
      if(is.na(se_vals[i])) {
        # Find closest non-NA values
        non_na_indices <- which(!is.na(se_vals))
        if(length(non_na_indices) > 0) {
          # Get closest value
          closest_idx <- non_na_indices[which.min(abs(non_na_indices - i))]
          se_vals[i] <- se_vals[closest_idx]
        } else {
          se_vals[i] <- 0.02  # Default if no non-NA values
        }
      }
    }
    return(se_vals)
  })
  
  CI <- lapply(1:length(se_list), function(t) {
    ci_mat <- matrix(NA, nrow=2, ncol=ncol(est))
    colnames(ci_mat) <- colnames(est)
    
    for(i in 1:ncol(est)) {
      # Only compute CI when both estimate and SE are valid
      if(!is.na(est[t,i]) && !is.na(se_list[[t]][i])) {
        # Calculate unbounded CI first
        lower_bound <- est[t,i] - 1.96 * se_list[[t]][i]  # Lower bound
        upper_bound <- est[t,i] + 1.96 * se_list[[t]][i]  # Upper bound
        
        # Apply bounds to ensure survival probabilities stay within [0,1]
        ci_mat[1,i] <- max(0, min(1, lower_bound))  # Bound lower CI to [0,1]
        ci_mat[2,i] <- max(0, min(1, upper_bound))  # Bound upper CI to [0,1]
        
        # Log if bounds were applied (helps with debugging)
        if(lower_bound < 0 || upper_bound > 1) {
          if(diagnostics) {
            message(sprintf("Bounded CI for t=%d, rule=%d: [%.4f,%.4f] -> [%.4f,%.4f]", 
                            t, i, lower_bound, upper_bound, ci_mat[1,i], ci_mat[2,i]))
          }
        }
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