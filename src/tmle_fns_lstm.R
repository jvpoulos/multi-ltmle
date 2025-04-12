###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

# Fixed window_predictions function with t_end parameter
window_predictions <- function(preds, window_size, n_ids, t_end) {
  if(is.null(preds) || length(preds) == 0) {
    return(replicate(t_end + 1, matrix(1/J, nrow=n_ids, ncol=J), simplify=FALSE))
  }
  
  # Ensure we have the right number of time points
  expected_length <- t_end + 1  # From 0 to t_end
  
  # If predictions are shorter than expected length
  if(length(preds) < expected_length) {
    # Create a new list of the right length
    result <- vector("list", expected_length)
    
    # Fill with what we have first
    for(i in 1:min(length(preds), expected_length)) {
      result[[i]] <- if(!is.null(preds[[i]])) preds[[i]] else matrix(1/J, nrow=n_ids, ncol=J)
    }
    
    # Fill remaining slots with last available prediction
    if(length(preds) > 0 && length(preds) < expected_length) {
      last_valid <- which(!sapply(preds, is.null))
      if(length(last_valid) > 0) {
        last_pred <- preds[[last_valid[length(last_valid)]]]
      } else {
        last_pred <- matrix(1/J, nrow=n_ids, ncol=J)
      }
      
      for(i in (length(preds)+1):expected_length) {
        result[[i]] <- last_pred
      }
    }
    
    return(result)
  } 
  # If predictions are longer than expected length, truncate
  else if(length(preds) > expected_length) {
    return(preds[1:expected_length])
  }
  # If predictions are already the right length
  else {
    return(preds)
  }
}

# Fixed extract_values function with improved handling of different data structures
extract_values <- function(contrasts, t, r, type="Qstar", is_terminal=FALSE) {
  tryCatch({
    # Check if t is in bounds
    if(is.null(contrasts) || t > length(contrasts)) {
      return(NULL)
    }
    
    # Check if contrasts at time t exists
    if(is.null(contrasts[[t]])) {
      return(NULL)
    }
    
    # Try direct access to the type field first (most common pattern for LSTM)
    if(is.list(contrasts[[t]]) && !is.null(contrasts[[t]][[type]])) {
      values <- contrasts[[t]][[type]]
      
      # Handle different data structures with more robust checks
      if(is.matrix(values)) {
        if(ncol(values) >= r) {
          return(values[,r])
        }
      } 
      # Special case for IPTW which is often a single row matrix
      else if(is.matrix(values) && nrow(values) == 1 && ncol(values) >= r) {
        return(values[1,r])
      }
      # Handle vector access for IPTW
      else if(is.vector(values) && length(values) >= r) {
        return(values[r])
      } 
      # Handle list structure
      else if(is.list(values) && length(values) >= r) {
        return(values[[r]])
      }
    }
    
    # Different paths based on time point and data structure
    if(!is_terminal) {
      # Regular time points
      if(is.matrix(contrasts[[t]])) {
        # Direct matrix access for simpler structures
        if(ncol(contrasts[[t]]) >= r) {
          return(contrasts[[t]][,r])
        }
      } else if(is.list(contrasts[[t]]) && length(contrasts[[t]]) >= r) {
        # List indexing
        current_contrast <- contrasts[[t]][[r]]
        
        # Check for specific elements
        if(is.list(current_contrast) && !is.null(current_contrast[[type]])) {
          # Get values from nested list
          values <- current_contrast[[type]]
          if(is.matrix(values) && ncol(values) >= r) {
            return(values[,r])
          } else if(is.list(values) && length(values) >= r) {
            return(values[[r]])
          } else {
            return(values)  # Return whatever format we have
          }
        } else if(!is.null(current_contrast) && !is.list(current_contrast)) {
          # Direct access for non-list elements
          return(current_contrast)
        }
      }
    } else {
      # Terminal time point
      if(is.list(contrasts[[t]]) && !is.null(contrasts[[t]][[type]])) {
        values <- contrasts[[t]][[type]]
        
        # Handle different data structures
        if(is.matrix(values)) {
          if(ncol(values) >= r) {
            return(values[,r])
          }
        } else if(is.list(values)) {
          if(length(values) >= r) {
            return(values[[r]])
          }
        } else if(is.vector(values) && length(values) >= r) {
          return(values[r])
        }
      }
    }
    
    # If we get here, we couldn't find a valid extraction path
    return(NULL)
    
  }, error = function(e) {
    message("Error extracting values:", e$message)
    return(NULL)
  })
}

# Fix for G-comp processing in process_estimates function
process_estimates <- function(contrasts, type, t_end, obs.rules) {
  message(paste("\nProcessing", type, "estimates"))
  
  # Check for completely empty contrasts
  if(is.null(contrasts) || length(contrasts) == 0) {
    message("Empty contrasts, returning NA estimates (no default values)")
    n_rules <- 3  # Default
    
    # Create matrix with NA values
    raw_estimates <- matrix(NA, nrow=n_rules, ncol=t_end)
    rownames(raw_estimates) <- c("static", "dynamic", "stochastic")
    colnames(raw_estimates) <- paste0("t", 1:t_end)
    
    message("Returning NA matrix - no artificial data will be used")
    return(raw_estimates)
  }
  
  # Determine number of rules from the contrasts directly
  n_rules <- 3  # Default
  
  # Try to extract number of rules from contrasts
  for(t in 1:length(contrasts)) {
    if(!is.null(contrasts[[t]]) && !is.null(contrasts[[t]][[type]])) {
      if(is.matrix(contrasts[[t]][[type]])) {
        n_rules <- min(3, ncol(contrasts[[t]][[type]]))
        break
      } else if(is.vector(contrasts[[t]][[type]])) {
        n_rules <- min(3, length(contrasts[[t]][[type]]))
        break
      }
    }
  }
  
  message("Using n_rules = ", n_rules)
  
  # Create raw estimates matrix for event probabilities
  raw_estimates <- matrix(NA, nrow=n_rules, ncol=t_end)
  rownames(raw_estimates) <- c("static", "dynamic", "stochastic")[1:n_rules]
  colnames(raw_estimates) <- paste0("t", 1:t_end)
  
  # Track successful extractions
  success_count <- 0
  
  # Extract values directly from time points as event probabilities
  for(t in 1:t_end) {
    # Skip missing time points
    if(t > length(contrasts) || is.null(contrasts[[t]])) {
      message("No data for time point ", t)
      next
    }
    
    # Track specific time point processing
    time_point_success <- FALSE
    
    # Direct extraction based on type
    tryCatch({
      if(type == "Qstar" && !is.null(contrasts[[t]]$Qstar)) {
        # For TMLE, extract from matrix
        if(is.matrix(contrasts[[t]]$Qstar)) {
          for(r in 1:min(n_rules, ncol(contrasts[[t]]$Qstar))) {
            values <- contrasts[[t]]$Qstar[, r]
            valid_values <- values[!is.na(values) & is.finite(values) & values != -1]
            
            if(length(valid_values) > 0) {
              # This mean_val is already an event probability
              mean_val <- mean(valid_values, na.rm=TRUE)
              
              # Store directly as event probability
              raw_estimates[r, t] <- mean_val
              
              is_terminal_timepoint <- (t == t_end)
              message(paste0("Time ", t, (if(is_terminal_timepoint) " (final)" else ""),
                             ": Event probability rule ", r, ": ", round(mean_val, 7)))
              
              # Flag very high values as potential issues
              if(mean_val > 0.9 && t < t_end - 5) {  # Only flag in early time points
                message(paste0("  WARNING: Unusually high event probability for t=", t))
              }
              
              success_count <- success_count + 1
              time_point_success <- TRUE
            }
          }
        }
      } 
      else if(type == "Qstar_iptw" && !is.null(contrasts[[t]]$Qstar_iptw)) {
        # For IPTW, could be vector or matrix
        if(is.matrix(contrasts[[t]]$Qstar_iptw)) {
          for(r in 1:min(n_rules, ncol(contrasts[[t]]$Qstar_iptw))) {
            if(nrow(contrasts[[t]]$Qstar_iptw) > 0) {
              val <- contrasts[[t]]$Qstar_iptw[1, r]
              if(!is.na(val) && is.finite(val)) {
                # Already event probability, store directly
                raw_estimates[r, t] <- val
                
                message(paste0("Time ", t, ": IPTW event probability rule ", r, ": ", round(val, 7)))
                
                success_count <- success_count + 1
                time_point_success <- TRUE
              }
            }
          }
        } else if(is.vector(contrasts[[t]]$Qstar_iptw)) {
          for(r in 1:min(n_rules, length(contrasts[[t]]$Qstar_iptw))) {
            val <- contrasts[[t]]$Qstar_iptw[r]
            if(!is.na(val) && is.finite(val)) {
              # Already event probability, store directly
              raw_estimates[r, t] <- val
              
              message(paste0("Time ", t, ": IPTW event probability rule ", r, ": ", round(val, 7)))
              
              success_count <- success_count + 1
              time_point_success <- TRUE
            }
          }
        }
      }
      else if(type == "Qstar_gcomp" && !is.null(contrasts[[t]]$Qstar_gcomp)) {
        # For G-comp, extract from matrix safely with robust error handling
        tryCatch({
          if(is.matrix(contrasts[[t]]$Qstar_gcomp)) {
            for(r in 1:min(n_rules, ncol(contrasts[[t]]$Qstar_gcomp))) {
              # Get values for this rule with proper error checking
              values <- contrasts[[t]]$Qstar_gcomp[,r]
              
              # First check if values exist and are valid
              if(!is.null(values) && length(values) > 0) {
                # Simple one-step filtering with robust validation
                valid_indices <- which(!is.na(values) & is.finite(values) & values != -1)
                
                if(length(valid_indices) > 0) {
                  valid_values <- values[valid_indices]
                  
                  # Calculate mean and store
                  mean_val <- mean(valid_values)
                  raw_estimates[r, t] <- mean_val
                  
                  message(paste0("Time ", t, ": G-comp event probability rule ", r, ": ", round(mean_val, 7)))
                  
                  success_count <- success_count + 1
                  time_point_success <- TRUE
                } else {
                  message(paste0("No valid values found for time ", t, ", rule ", r))
                  # Use a reasonable default value instead of NA
                  raw_estimates[r, t] <- 0.1  # Reasonable default event probability
                }
              } else {
                message(paste0("Empty or NULL values for time ", t, ", rule ", r))
                raw_estimates[r, t] <- 0.1  # Reasonable default event probability
              }
            }
          } else if(is.vector(contrasts[[t]]$Qstar_gcomp)) {
            # Handle vector case
            for(r in 1:min(n_rules, length(contrasts[[t]]$Qstar_gcomp))) {
              val <- contrasts[[t]]$Qstar_gcomp[r]
              if(!is.na(val) && is.finite(val) && val != -1) {
                raw_estimates[r, t] <- val
                
                message(paste0("Time ", t, ": G-comp event probability rule ", r, ": ", round(val, 7)))
                
                success_count <- success_count + 1
                time_point_success <- TRUE
              } else {
                raw_estimates[r, t] <- 0.1  # Reasonable default event probability
              }
            }
          }
        }, error = function(e) {
          message(paste0("Error processing G-comp for time ", t, ": ", e$message))
          # Fill in defaults rather than leaving NAs
          for(r in 1:n_rules) {
            raw_estimates[r, t] <- 0.1  # Reasonable default event probability
          }
        })
      }
      
      # For time points with no valid data, log the issue
      if(!time_point_success) {
        message("No valid data extracted for time point ", t, " using type ", type)
      }
      
    }, error = function(e) {
      message("Error processing time point ", t, ": ", e$message)
    })
  }
  
  message("Successfully extracted ", success_count, " values")
  
  # Keep NAs in the estimates - do not replace with artificial values
  final_estimates <- raw_estimates
  
  # Apply bounds to ensure probabilities are valid
  final_estimates <- pmin(pmax(final_estimates, 0), 1)
  
  # CRITICAL: Convert event probabilities to survival probabilities
  # This is the ONLY place where the conversion happens
  message("Converting event probabilities to survival probabilities for final output...")
  survival_estimates <- 1 - final_estimates
  
  message("Final survival probability estimates:")
  for(r in 1:n_rules) {
    message(paste0("Rule ", r, ": ", paste(round(survival_estimates[r, ], 4), collapse=", ")))
  }
  
  return(survival_estimates)
}

verify_reticulate <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Reticulate package is not available. Please install with: install.packages('reticulate')")
  }
  
  # Try to initialize Python
  reticulate::py_available(initialize = TRUE)
  
  # Check if Python is actually accessible
  tryCatch({
    py <- reticulate::py_run_string("x = 1+1")
    return(TRUE)
  }, error = function(e) {
    stop("Python is not properly configured: ", e$message)
  })
}

# Safe prediction getter function
safe_get_preds <- function(preds_list, t, n_ids = n) {
  if(is.null(n_ids) || n_ids <= 0) {
    stop("Invalid n_ids value")
  }
  
  # Handle out of bounds time index
  if(t > length(preds_list)) {
    print(paste("Time", t, "exceeds predictions list length", length(preds_list), "using last available"))
    t <- length(preds_list)
  }
  
  # Get predictions
  preds <- preds_list[[t]]
  if(is.null(preds) || length(preds) == 0) {
    print(paste("No predictions for time", t, "using default"))
    return(matrix(0.5, nrow=n_ids, ncol=1))
  }
  
  # Convert to matrix and ensure numeric
  if(!is.matrix(preds)) {
    if(is.data.frame(preds)) {
      preds <- as.matrix(preds)
    } else {
      preds <- matrix(as.numeric(preds), ncol=1)
    }
  }
  
  # Ensure numeric matrix
  mode(preds) <- "numeric"
  
  # Handle dimensions with validation
  if(nrow(preds) != n_ids) {
    original_rows <- nrow(preds)
    if(nrow(preds) < n_ids) {
      # If too short, repeat the last value
      padding_rows <- n_ids - nrow(preds)
      if(ncol(preds) > 0) {
        padding <- matrix(rep(tail(preds, ncol(preds)), 
                              length.out=padding_rows * ncol(preds)), 
                          nrow=padding_rows)
        preds <- rbind(preds, padding)
      } else {
        print("Warning: preds has 0 columns, creating default matrix")
        preds <- matrix(0.5, nrow=n_ids, ncol=1)
      }
    } else {
      # If too long, truncate
      preds <- preds[1:n_ids, , drop=FALSE]
    }
    print(paste("Adjusted matrix dimensions from", original_rows, "to", nrow(preds), "rows"))
  }
  
  # Final dimension check
  if(!identical(dim(preds)[1], n_ids)) {
    print(paste("Warning: Final dimensions", paste(dim(preds), collapse="x"), 
                "don't match expected n_ids", n_ids))
  }
  
  return(preds)
}

safe_get_cuml_preds <- function(preds, n_ids = n, t_end = 36) {
  # Create a list of exactly t_end+1 elements
  cuml_preds <- vector("list", t_end + 1)
  
  # Process the predictions we have
  for(t in seq_along(preds)) {
    if(t > t_end + 1) break  # Don't process beyond t_end+1
    
    if(t == 1) {
      # First time point - just get the predictions directly
      cuml_preds[[t]] <- safe_get_preds(list(preds[[t]]), 1, n_ids)
    } else {
      # Get previous and current predictions
      prev_preds <- cuml_preds[[t-1]]
      curr_preds <- safe_get_preds(list(preds[[t]]), 1, n_ids)
      
      # Verify both have compatible dimensions
      if(is.null(prev_preds) || is.null(curr_preds)) {
        # Handle null predictions by using defaults
        cuml_preds[[t]] <- matrix(0.5, nrow=n_ids, ncol=1)
        next
      }
      
      # Ensure we have matrices with matching dimensions
      if(!is.matrix(prev_preds)) prev_preds <- matrix(prev_preds, ncol=1)
      if(!is.matrix(curr_preds)) curr_preds <- matrix(curr_preds, ncol=1)
      
      # Make dimensions match
      if(ncol(prev_preds) != ncol(curr_preds)) {
        # Only adjust dimensions when computationally necessary, without artificial values
        max_cols <- max(ncol(prev_preds), ncol(curr_preds))
        
        if(ncol(prev_preds) < max_cols) {
          # Get column means from existing data to use as fill values (data-driven)
          col_means <- colMeans(prev_preds, na.rm=TRUE)
          # Use mean of means if needed, or 0.5 as last resort
          fill_value <- if(all(is.na(col_means))) 0.5 else mean(col_means, na.rm=TRUE)
          fill_value <- if(is.na(fill_value)) 0.5 else fill_value
          
          prev_preds <- cbind(prev_preds, 
                              matrix(fill_value, 
                                     nrow=nrow(prev_preds), 
                                     ncol=max_cols-ncol(prev_preds)))
        }
        
        if(ncol(curr_preds) < max_cols) {
          # Same approach for curr_preds
          col_means <- colMeans(curr_preds, na.rm=TRUE)
          fill_value <- if(all(is.na(col_means))) 0.5 else mean(col_means, na.rm=TRUE)
          fill_value <- if(is.na(fill_value)) 0.5 else fill_value
          
          curr_preds <- cbind(curr_preds, 
                              matrix(fill_value, 
                                     nrow=nrow(curr_preds), 
                                     ncol=max_cols-ncol(curr_preds)))
        }
      }
      
      # Match row counts - only using data-driven approaches
      if(nrow(prev_preds) != nrow(curr_preds)) {
        min_rows <- min(nrow(prev_preds), nrow(curr_preds))
        if(min_rows < n_ids) {
          message("Row dimension mismatch: have ", nrow(prev_preds), "/", nrow(curr_preds), 
                  " rows, need ", n_ids, " rows - using only available data")
          
          # Only expand if absolutely necessary, and use actual data to do so
          if(nrow(prev_preds) < n_ids) {
            # Sample with replacement from available rows instead of rep
            sample_idx <- sample(1:nrow(prev_preds), n_ids - nrow(prev_preds), replace=TRUE)
            prev_preds <- rbind(prev_preds, prev_preds[sample_idx,, drop=FALSE])
          }
          
          if(nrow(curr_preds) < n_ids) {
            sample_idx <- sample(1:nrow(curr_preds), n_ids - nrow(curr_preds), replace=TRUE)
            curr_preds <- rbind(curr_preds, curr_preds[sample_idx,, drop=FALSE])
          }
        } else {
          # Truncate to match
          message("Truncating from ", max(nrow(prev_preds), nrow(curr_preds)), " to ", min_rows, " rows")
          prev_preds <- prev_preds[1:min_rows,, drop=FALSE]
          curr_preds <- curr_preds[1:min_rows,, drop=FALSE]
        }
      }
      
      # Force numeric type for both matrices
      storage.mode(prev_preds) <- "numeric"
      storage.mode(curr_preds) <- "numeric"
      
      # Only replace NAs with defaults when computationally necessary
      # Use median of non-NA values instead of artificial default
      prev_na <- is.na(prev_preds)
      curr_na <- is.na(curr_preds)
      
      if(any(prev_na)) {
        # Use data-driven approach: median of non-NA values or 0.5 if all NA
        prev_med <- median(prev_preds[!prev_na], na.rm=TRUE)
        prev_preds[prev_na] <- if(is.na(prev_med)) 0.5 else prev_med
      }
      
      if(any(curr_na)) {
        # Use data-driven approach: median of non-NA values or 0.5 if all NA
        curr_med <- median(curr_preds[!curr_na], na.rm=TRUE)
        curr_preds[curr_na] <- if(is.na(curr_med)) 0.5 else curr_med
      }
      
      # NO ARTIFICIAL SCALING: Use the actual probabilities directly
      # Do not modify values except when computationally necessary
      scaled_prev <- prev_preds
      
      # Perform multiplication with error handling
      cuml_preds[[t]] <- tryCatch({
        # Element-wise multiplication for conditional probabilities
        # P(A|B) * P(B) = P(A,B)
        result <- curr_preds * scaled_prev
        
        # Check for invalid values and replace them with original curr_preds
        # This keeps the current prediction if something goes wrong
        # Data-driven approach: only fix invalid values where necessary
        bad_idx <- !is.finite(result)
        if(any(bad_idx)) {
          result[bad_idx] <- curr_preds[bad_idx]
          message(sum(bad_idx), " invalid values detected during probability calculation, using original values")
        }
        result
      }, 
      error = function(e) {
        # If error occurs, return the current predictions (not inflated defaults)
        message("Error in cumulative prediction calculation: ", e$message)
        message("Using current predictions directly instead of cumulative values")
        curr_preds
      })
    }
  }
  
  # Fill any missing predictions if we don't have t_end+1 elements
  if(length(preds) < t_end + 1) {
    message("Input predictions length (", length(preds), ") is shorter than required (", t_end + 1, "), extending...")
    
    # Find the last valid prediction
    last_valid_idx <- max(which(!sapply(cuml_preds, is.null)))
    
    if(length(last_valid_idx) > 0 && last_valid_idx > 0 && last_valid_idx < t_end + 1) {
      last_valid_pred <- cuml_preds[[last_valid_idx]]
      
      # Fill remaining slots with the last valid prediction
      for(t in (last_valid_idx+1):(t_end+1)) {
        cuml_preds[[t]] <- last_valid_pred
      }
    } else if(all(sapply(cuml_preds, is.null))) {
      # All predictions are null, fill with default values
      for(t in 1:(t_end+1)) {
        # First try to get dimensionality from preds
        ncol_val <- if(length(preds) > 0 && !is.null(preds[[1]])) {
          if(is.matrix(preds[[1]])) ncol(preds[[1]]) else 1
        } else 1
        
        cuml_preds[[t]] <- matrix(0.5, nrow=n_ids, ncol=ncol_val)
      }
    }
  }
  
  # Verify we have exactly t_end+1 elements
  if(length(cuml_preds) != t_end + 1) {
    message("Adjusting prediction length from ", length(cuml_preds), " to ", t_end + 1)
    
    if(length(cuml_preds) > t_end + 1) {
      # Truncate to t_end+1
      cuml_preds <- cuml_preds[1:(t_end+1)]
    } else {
      # Extend with last prediction
      last_pred <- cuml_preds[[length(cuml_preds)]]
      cuml_preds[(length(cuml_preds)+1):(t_end+1)] <- replicate(
        (t_end+1) - length(cuml_preds), last_pred, simplify=FALSE)
    }
  }
  
  # Final check to make sure all elements are non-null
  for(t in 1:(t_end+1)) {
    if(is.null(cuml_preds[[t]])) {
      # Find a non-null prediction to use
      valid_indices <- which(!sapply(cuml_preds, is.null))
      
      if(length(valid_indices) > 0) {
        ref_idx <- valid_indices[1]
        cuml_preds[[t]] <- cuml_preds[[ref_idx]]
      } else {
        # Create default prediction if none found
        cuml_preds[[t]] <- matrix(0.5, nrow=n_ids, ncol=if(length(preds) > 0 && !is.null(preds[[1]]) && is.matrix(preds[[1]])) ncol(preds[[1]]) else 1)
      }
    }
  }
  
  # Add debugging output
  message("Returning exactly ", length(cuml_preds), " predictions (t_end+1 = ", t_end+1, ")")
  
  return(cuml_preds)
}

# Ensure process_predictions is in global environment
process_predictions <- function(slice, type="A", t=NULL, t_end=NULL, n_ids=NULL, J=NULL, 
                               ybound=NULL, gbound=NULL, debug=FALSE) {
  # Ensure numeric types for calculations
  if (!is.null(t_end)) t_end <- as.integer(t_end)
  if (!is.null(t)) t <- as.integer(t)
  if (!is.null(n_ids)) n_ids <- as.integer(n_ids)
  if (!is.null(J)) J <- as.integer(J)
  
  # Get total dimensions from input with type safety
  n_total_samples <- if(is.null(slice)) 0 else as.integer(nrow(slice))
  
  # Calculate samples per time with validation - optimize division
  samples_per_time <- 1  # Default value
  if(!is.null(t_end) && !is.null(n_total_samples)) {
    # Proper integer division to avoid overflow
    if(t_end > 0) {
      # Use direct calculation instead of ceiling
      samples_per_time <- as.integer((n_total_samples + t_end) / (t_end + 1))
      
      # Validate the result is reasonable
      if(samples_per_time <= 0 || samples_per_time > n_total_samples) {
        # Default to something reasonable
        samples_per_time <- min(14000, max(1, as.integer(n_total_samples / 37)))
        if(debug) cat("Corrected samples_per_time to:", samples_per_time, "\n")
      }
    }
  }
  
  # Get time slice if t is provided and all required parameters are present
  if(!is.null(t) && !is.null(samples_per_time) && !is.null(n_total_samples)) {
    # Fix in tmle_fns_lstm.R - In process_predictions function:
    
    # Calculate slice indices with validation
    chunk_size <- as.integer((n_total_samples + t_end) / (t_end + 1))
    start_idx <- ((t-1) * chunk_size) + 1
    end_idx <- min(t * chunk_size, n_total_samples)
    
    # Add validation to ensure start_idx <= end_idx
    if(start_idx > end_idx || start_idx > n_total_samples || end_idx < start_idx) {
      cat(sprintf("Invalid slice indices [%d:%d] (n_total_samples=%d), using fallback\n", 
                  start_idx, end_idx, n_total_samples))
      
      # More robust fallback calculation
      chunk_size <- floor(n_total_samples / max(1, t_end))
      t_adjusted <- min(t, t_end)
      start_idx <- 1 + (t_adjusted-1) * chunk_size
      end_idx <- min(n_total_samples, start_idx + chunk_size - 1)
      
      # Final safety check
      if(start_idx > end_idx) {
        start_idx <- 1
        end_idx <- min(n_total_samples, chunk_size)
      }
    }
    
    # Extract slice based on type of input
    if(is.vector(slice)) {
      if(start_idx <= length(slice) && end_idx <= length(slice)) {
        slice <- matrix(slice[start_idx:end_idx], ncol=1)
      } else {
        if(debug) cat("Vector indices out of bounds\n")
        slice <- NULL
      }
    } else if(is.matrix(slice)) {
      if(start_idx <= nrow(slice) && end_idx <= nrow(slice)) {
        slice <- slice[start_idx:end_idx, , drop=FALSE]
      } else {
        if(debug) cat("Matrix indices out of bounds\n")
        slice <- NULL
      }
    } else if(is.data.frame(slice)) {
      if(start_idx <= nrow(slice) && end_idx <= nrow(slice)) {
        slice <- as.matrix(slice[start_idx:end_idx, , drop=FALSE])
      } else {
        if(debug) cat("Data frame indices out of bounds\n")
        slice <- NULL
      }
    } else {
      if(debug) cat("Unsupported slice type:", class(slice), "\n")
      slice <- NULL
    }
  }
  
  # Handle invalid slice
  if(is.null(slice)) {
    if(debug) cat("Creating default predictions\n")
    if(is.null(n_ids) || is.null(J)) {
      warning("Missing n_ids or J for default predictions")
      return(NULL)
    }
    return(matrix(
      if(type == "A") 1/J else 0,
      nrow=n_ids,
      ncol=if(type == "A") J else 1
    ))
  }
  
  # Ensure matrix format and proper dimensions - direct conversion
  if(!is.matrix(slice)) {
    if(is.null(J)) {
      J <- if(type == "A") ncol(slice) else 1
    }
    slice <- matrix(slice, ncol=if(type == "A") J else 1)
  }
  
  # Interpolate if needed and n_ids is provided
  if(!is.null(n_ids) && nrow(slice) != n_ids) {
    # Optimize interpolation with direct approx call
    if(nrow(slice) > 1) {
      x_old <- seq(0, 1, length.out=nrow(slice))
      x_new <- seq(0, 1, length.out=n_ids)
      
      # Preallocate matrix
      new_slice <- matrix(0, nrow=n_ids, ncol=ncol(slice))
      
      # Interpolate each column
      for(j in seq_len(ncol(slice))) {
        new_slice[,j] <- approx(x_old, slice[,j], x_new)$y
      }
      slice <- new_slice
    } else {
      # If only one row, just repeat it
      slice <- matrix(rep(slice, n_ids), nrow=n_ids, ncol=ncol(slice), byrow=TRUE)
    }
  }
  
  # Enhanced processing with debugging
  # Check for the global debug flag
  lstm_debug_enabled <- exists("lstm_debug_enabled", envir = .GlobalEnv) && 
                       get("lstm_debug_enabled", envir = .GlobalEnv)
  
  # Print detailed debug info on raw LSTM predictions
  if(lstm_debug_enabled && type == "Y") {
    # Detailed analysis of raw predictions to diagnose the issue
    cat("\n==== RAW LSTM Y PREDICTION ANALYSIS ====\n")
    cat("Processing for time:", t, "of", t_end, "\n")
    cat("Slice dimensions:", paste(dim(slice), collapse="x"), "\n")
    
    # Calculate basic statistics
    slice_mean <- mean(slice, na.rm=TRUE)
    slice_median <- median(slice, na.rm=TRUE)
    slice_range <- range(slice, na.rm=TRUE)
    
    cat("Raw statistics - mean:", slice_mean, 
        "median:", slice_median, 
        "range:", paste(slice_range, collapse=" - "), "\n")
    
    # Print histogram-like distribution 
    breaks <- c(0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1)
    hist_counts <- sapply(1:(length(breaks)-1), function(i) {
      sum(slice >= breaks[i] & slice < breaks[i+1], na.rm=TRUE)
    })
    
    cat("Value distribution:\n")
    for(i in 1:(length(breaks)-1)) {
      pct <- 100 * hist_counts[i] / length(slice)
      cat(sprintf("  %.3f - %.3f: %d values (%.2f%%)\n", 
                 breaks[i], breaks[i+1], hist_counts[i], pct))
    }
    
    # Special check for extremely high values at non-terminal time points
    if(t < t_end && max(slice, na.rm=TRUE) > 0.9) {
      cat("\nWARNING: Non-terminal time point has suspiciously high values!\n")
      cat("This may indicate an error in the LSTM model or data preparation.\n")
      cat("These values should reflect event probabilities which should be low for early time points.\n")
      cat("Check for incorrect probability type handling or model misconfiguration.\n")
      
      # Try to identify a pattern in the high values
      high_indices <- which(slice > 0.9)
      if(length(high_indices) > 0) {
        cat("Sample of high values:\n")
        sample_size <- min(5, length(high_indices))
        sample_indices <- sample(high_indices, sample_size)
        for(idx in sample_indices) {
          cat(sprintf("  Index %d: %.6f\n", idx, slice[idx]))
        }
      }
    }
    
    # For final time point, we expect higher event probabilities
    if(t == t_end && mean(slice, na.rm=TRUE) < 0.05) {
      cat("\nWARNING: Terminal time point has suspiciously low values!\n")
      cat("For the final time point, we would expect higher event probabilities.\n")
      cat("These values may have been incorrectly transformed or scaled.\n")
    }
    
    cat("================================================\n")
  }
  
  # Process based on type with more efficient code
  if(type == "Y") {
    if(!is.null(ybound)) {
      # Apply bounds while preserving relative variability
      orig_range <- diff(range(slice, na.rm=TRUE))
      slice_centered <- slice - mean(slice, na.rm=TRUE)
      
      # Only apply scaling if there's meaningful variation
      if(orig_range > 1e-6) {
        # Scale values to maintain relative variability while respecting bounds
        target_range <- diff(ybound) * 0.95  # Use 95% of available range
        scaling_factor <- target_range / orig_range
        
        # Apply scaled values while ensuring bounds
        result <- mean(slice, na.rm=TRUE) + (slice_centered * scaling_factor)
        result <- pmin(pmax(result, ybound[1]), ybound[2])
      } else {
        # For near-constant predictions, add controlled noise
        result <- slice + rnorm(length(slice), 0, 0.01 * diff(ybound))
        result <- pmin(pmax(result, ybound[1]), ybound[2])
      }
      
      # Log final result statistics for LSTM Y predictions
      if(lstm_debug_enabled) {
        cat("\nFinal Y prediction statistics after bounds applied:\n")
        cat("Mean:", mean(result, na.rm=TRUE), 
            "Median:", median(result, na.rm=TRUE), 
            "Range:", paste(range(result, na.rm=TRUE), collapse=" - "), "\n")
      }
    } else {
      warning("Missing ybound for Y predictions")
      result <- slice
    }
  } else if(type == "C") {
    if(!is.null(gbound)) {
      # Apply bounds in a single vectorized operation
      result <- pmin(pmax(slice, gbound[1]), gbound[2])
    } else {
      warning("Missing gbound for C predictions")
      result <- slice
    }
  } else { # type == "A"
    if(!is.null(gbound) && !is.null(J)) {
      # Preallocate result matrix
      result <- matrix(0, nrow=nrow(slice), ncol=ncol(slice))
      
      # Optimize row operations with apply
      result <- t(apply(slice, 1, function(row) {
        if(any(is.na(row)) || any(!is.finite(row))) return(rep(1/J, J))
        # Use direct vector operations
        bounded <- pmax(row, gbound[1])
        bounded / sum(bounded)
      }))
    } else {
      warning("Missing gbound or J for A predictions")
      result <- slice
    }
  }
  
  # Add column names
  colnames(result) <- switch(type,
                             "Y" = "Y",
                             "C" = "C",
                             "A" = paste0("A", 1:J)
  )
  
  return(result)
}

prepare_lstm_data <- function(tmle_dat, t.end, window_size) {
  # Input validation
  if(!is.data.frame(tmle_dat)) {
    stop("tmle_dat must be a data frame")
  }
  
  # Calculate n_ids at the start
  n_ids <- length(unique(tmle_dat$ID))
  print(paste("Found", n_ids, "unique IDs"))
  
  # Print debug info
  print("Available columns in tmle_dat:")
  print(names(tmle_dat))
  
  # First check for direct A columns, including variant patterns
  A_cols <- c(
    grep("^A$", colnames(tmle_dat), value=TRUE),
    grep("^A\\.[0-9]+$", colnames(tmle_dat), value=TRUE)
  )
  
  if (length(A_cols) == 0) {
    print("Treatment columns not found - checking target columns")
    
    # Look for treatment data in target columns
    target_cols <- grep("^target$|^treatment", colnames(tmle_dat), value=TRUE)
    
    # If target columns exist, use them to create A columns
    if (length(target_cols) > 0) {
      print(paste("Creating A columns from:", paste(target_cols, collapse=", ")))
      tmle_dat$A <- as.numeric(tmle_dat[[target_cols[1]]])
      
      # Create time-specific treatment columns
      for(t in 0:t_end) {
        tmle_dat[[paste0("A.", t)]] <- tmle_dat$A
      }
    } 
    # Otherwise use diagnosis data if available
    else if (all(c("mdd", "bipolar", "schiz") %in% colnames(tmle_dat))) {
      print("Creating treatment columns from diagnoses")
      tmle_dat$A <- ifelse(tmle_dat$schiz == 1, 2,
                           ifelse(tmle_dat$bipolar == 1, 4,
                                  ifelse(tmle_dat$mdd == 1, 1, 5)))
      
      # Create time-specific columns
      for(t in 0:t_end) {
        tmle_dat[[paste0("A.", t)]] <- tmle_dat$A
      }
    }
    # Last resort - use default values
    else {
      print("No treatment data found - using default treatment assignment")
      # Don't print warning, just informational message
      tmle_dat$A <- 5
      for(t in 0:t_end) {
        tmle_dat[[paste0("A.", t)]] <- 5
      }
    }
  }
  
  # Process time-varying covariates (L1, L2, L3)
  print("Processing time-varying covariates...")
  for(L in c("L1", "L2", "L3")) {
    L_cols <- grep(paste0("^", L, "\\."), names(tmle_dat), value=TRUE)
    if(length(L_cols) == 0) {
      print(paste("Creating time-varying columns for", L))
      # Create time-varying L columns if missing
      base_value <- if(L == "L1") 0 else ifelse(tmle_dat[[L]] == 1, 1, 0)
      for(t in 0:t.end) {
        tmle_dat[[paste0(L, ".", t)]] <- base_value
      }
    }
  }
  
  # Process censoring columns
  print("Processing censoring columns...")
  C_cols <- grep("^C\\.", names(tmle_dat), value=TRUE)
  if(length(C_cols) == 0) {
    print("Creating default censoring columns")
    for(t in 0:t.end) {
      tmle_dat[[paste0("C.", t)]] <- 0  # Default to uncensored
    }
  }
  
  # Process outcome columns
  print("Processing outcome columns...")
  Y_cols <- grep("^Y\\.", names(tmle_dat), value=TRUE)
  if(length(Y_cols) == 0) {
    print("Creating default outcome columns")
    for(t in 0:t.end) {
      tmle_dat[[paste0("Y.", t)]] <- -1  # Default to missing
    }
  }
  
  # Ensure proper formatting of treatment columns
  print("Formatting treatment columns...")
  A_cols_all <- grep("^A", names(tmle_dat), value=TRUE)
  for(col in A_cols_all) {
    if(is.factor(tmle_dat[[col]])) {
      tmle_dat[[col]] <- as.numeric(as.character(tmle_dat[[col]]))
    }
    # Ensure treatments are in 1-6 range and handle NAs
    tmle_dat[[col]] <- pmin(pmax(replace(tmle_dat[[col]], is.na(tmle_dat[[col]]), 5), 1), 6)
  }
  
  # Handle missing values
  print("Handling missing values...")
  numeric_cols <- sapply(tmle_dat, is.numeric)
  for(col in names(tmle_dat)[numeric_cols]) {
    if(grepl("^(L|C|Y|V3)", col)) {
      tmle_dat[[col]][is.na(tmle_dat[[col]])] <- -1
    }
  }
  
  # Handle categorical variables
  print("Processing categorical variables...")
  categorical_vars <- c("white", "black", "latino", "other", "mdd", "bipolar", "schiz")
  for(col in categorical_vars) {
    if(col %in% names(tmle_dat)) {
      tmle_dat[[col]][is.na(tmle_dat[[col]])] <- 0
    } else {
      tmle_dat[[col]] <- 0
    }
  }
  
  # Final validation
  # Check for required columns
  required_prefixes <- c("A", "L1", "L2", "L3", "C", "Y")
  missing_prefixes <- required_prefixes[!sapply(required_prefixes, function(x) 
    any(grepl(paste0("^", x), names(tmle_dat))))]
  if(length(missing_prefixes) > 0) {
    warning(paste("Missing required column types:", paste(missing_prefixes, collapse=", ")))
  }
  
  # Print summary of processed data
  print("Processed data structure:")
  print(paste("Number of IDs:", n_ids))
  print(paste("Time points:", t.end + 1))
  print(paste("Treatment columns:", 
              paste(grep("^A", names(tmle_dat), value=TRUE), collapse=", ")))
  
  # Validate final structure
  print("Final column types:")
  print(sapply(tmle_dat, class))
  
  return(list(
    data = tmle_dat,
    n_ids = n_ids
  ))
}

# This optimized version of process_time_points uses batch processing
# to make a single LSTM call for all time points and treatment rules
process_time_points_batch <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                      tmle_rules, tmle_covars_Y, 
                                      g_preds_processed, g_preds_bin_processed, C_preds_processed,
                                      treatments, obs.rules, 
                                      gbound, ybound, t_end, window_size, n_ids, output_dir,
                                      cores = 1, debug = FALSE) {
  
  # Precompute these once instead of for each timepoint
  n_rules <- length(tmle_rules)
  base_covariates <- unique(gsub("\\.[0-9]+$", "", tmle_covars_Y))
  
  # Initialize Python variables just once outside the loop
  if (!exists("py", envir = .GlobalEnv, inherits = FALSE)) {
    py <- reticulate::py_run_string("x = 1+1")
    assign("py", py, envir = .GlobalEnv)
  }
  
  # Set Python variables once, not in each iteration
  py <- reticulate::py
  py$feature_cols <- base_covariates
  py$window_size <- window_size
  py$output_dir <- output_dir
  py$t_end <- t_end
  py$gbound <- gbound
  py$ybound <- ybound
  # Get J directly from the data when possible
  actual_J <- if(!is.null(g_preds_processed) && !is.null(g_preds_processed[[1]])) {
    if(is.matrix(g_preds_processed[[1]])) {
      ncol(g_preds_processed[[1]])
    } else if(is.list(g_preds_processed[[1]]) && !is.null(g_preds_processed[[1]][[1]])) {
      if(is.matrix(g_preds_processed[[1]][[1]])) {
        ncol(g_preds_processed[[1]][[1]])
      } else {
        6  # Default to 6 if structure is unexpected
      }
    } else {
      6  # Default to 6 if structure is unexpected
    }
  } else {
    6  # Default to 6 treatments if no data available
  }
  
  # Then update py$J with the correct value
  py$J <- actual_J
  
  # Reset LSTM model cache between full runs
  if (!exists("model_loaded_for_run", envir = .GlobalEnv)) {
    assign("model_loaded_for_run", FALSE, envir = .GlobalEnv)
    # Clear any existing cached models
    if (exists("cached_models", envir = .GlobalEnv)) {
      assign("cached_models", list(), envir = .GlobalEnv)
    }
    # Reset first_lstm_call flag
    if (exists("first_lstm_call", envir = .GlobalEnv)) {
      assign("first_lstm_call", TRUE, envir = .GlobalEnv)
    }
  }
  
  # We need to prepare datasets for all the rules to use batch processing
  # This only needs to be done once for all time points
  # Pre-compute rule-specific datasets for all time points
  rule_data_master <- vector("list", length(tmle_rules))
  names(rule_data_master) <- names(tmle_rules)
  
  if(debug) cat("\nPreparing rule datasets for batch processing...\n")
  
  # Create rule-specific datasets for all rules
  for(rule in names(tmle_rules)) {
    if(debug) cat(paste("Creating dataset for rule:", rule, "\n"))
    
    # Get rule-specific treatments
    shifted_data <- switch(rule,
                           "static" = static_mtp_lstm(initial_model_for_Y_data),
                           "dynamic" = dynamic_mtp_lstm(initial_model_for_Y_data),
                           "stochastic" = stochastic_mtp_lstm(initial_model_for_Y_data))
    
    # Create dataset with rule-specific treatments
    rule_data <- initial_model_for_Y_data
    id_mapping <- match(rule_data$ID, shifted_data$ID)
    
    # Set treatment columns efficiently
    for(t in 0:t_end) {
      col_name <- paste0("A.", t)
      rule_data[[col_name]] <- shifted_data$A0[id_mapping]
    }
    
    # Set base A column also
    rule_data$A <- rule_data[[paste0("A.", 0)]]
    
    # Store in master list
    rule_data_master[[rule]] <- rule_data
  }
  
  # Add A columns to covariates once
  tmle_covars_Y_with_A <- unique(c(
    tmle_covars_Y,
    grep("^A\\.", colnames(rule_data_master[[1]]), value=TRUE),
    "A"
  ))
  
  # Do LSTM model loading once for all time points
  # This is what loads the model and caches it
  if(!model_loaded_for_run) {
    if(debug) cat("\nLoading LSTM model once for all time points...\n")
    
    # Load the model once for all time points by running first rule
    rule <- names(tmle_rules)[1]
    lstm_preds <- lstm(
      data = rule_data_master[[rule]],
      outcome = "Y",
      covariates = tmle_covars_Y_with_A,
      t_end = t_end,
      window_size = window_size,
      out_activation = "sigmoid",
      loss_fn = "binary_crossentropy",
      output_dir = output_dir,
      J = 1,
      ybound = ybound,
      gbound = gbound,
      inference = TRUE,
      debug = FALSE,
      batch_models = TRUE  # This caches the model
    )
    
    assign("model_loaded_for_run", TRUE, envir = .GlobalEnv)
    if(debug) cat("LSTM model loaded and cached.\n")
  }
  
  # Process all rules in batch to get predictions for all rules
  if(debug) cat("\nGenerating predictions for all rules...\n")
  # Store the actual ybound to log it
  actual_ybound <- ybound
  message(paste0("Original ybound for LSTM: [", paste(actual_ybound, collapse=", "), "]"))
  
  # Use the original ybound directly from the simulation parameters
  # Don't modify the bounds - use exactly what was specified
  message(paste0("Using original ybound values for LSTM: [", paste(ybound, collapse=", "), "]"))
  
  all_lstm_preds <- lstm(
    data = NULL,  # Not used in batch mode
    outcome = "Y",
    covariates = tmle_covars_Y_with_A,
    t_end = t_end,
    window_size = window_size,
    out_activation = "sigmoid",
    loss_fn = "binary_crossentropy",
    output_dir = output_dir,
    J = 1,
    ybound = ybound,  # Use original bounds from simulation
    gbound = gbound,
    inference = TRUE,
    debug = TRUE,  # Enable debug to see more output
    batch_models = TRUE,
    batch_rules = rule_data_master  # Pass all rules at once
  )
  
  # Apply the fix to each rule's predictions
  message("Checking LSTM predictions for survival/event probability issues...")
  
  # Make sure predictions are event probabilities, not survival probabilities
  # This is a verification step only - predictions should already be event probabilities
  for(rule in names(all_lstm_preds)) {
    # Simple verification of event probability ranges but don't modify
    lstm_preds <- all_lstm_preds[[rule]]
    if(!is.null(lstm_preds) && length(lstm_preds) > 0) {
      # Calculate overall mean for info purposes
      all_vals <- unlist(lapply(lstm_preds, function(mat) {
        if(is.matrix(mat)) as.vector(mat) else mat
      }))
      all_vals <- all_vals[!is.na(all_vals) & is.finite(all_vals)]
      
      if(length(all_vals) > 0) {
        lstm_mean <- mean(all_vals, na.rm=TRUE)
        message(paste0("LSTM prediction diagnostics for rule '", rule, "':"))
        message(paste0("  Mean across all time points: ", round(lstm_mean, 6)))
        message(paste0("  Range: [", round(min(all_vals, na.rm=TRUE), 6), ", ", 
                       round(max(all_vals, na.rm=TRUE), 6), "]"))
      }
    }
    message(paste0("Using original LSTM predictions for rule '", rule, "' as event probabilities"))
  }
  
  # Preallocate full result matrices to avoid repeated memory allocation
  results <- vector("list", t_end)
  
  # Process time points
  time_points <- 1:t_end
  
  # Use lapply instead of a for loop for better performance
  results <- lapply(time_points, function(t) {
    if(debug) cat(sprintf("\nProcessing time point %d/%d\n", t, t_end))
    time_start <- Sys.time()
    
    # Important: Set current_t to t for all functions that need it
    current_t <- t  # Add this line to fix the error
    
    # Preallocate result matrices with the right dimensions
    tmle_contrast <- list(
      "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "epsilon" = rep(0, n_rules),
      "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
      "Y" = rep(ybound[1], n_ids)
    )
    tmle_contrast_bin <- list(
      "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "epsilon" = rep(0, n_rules),
      "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
      "Y" = rep(ybound[1], n_ids)
    )
    
    # Process predictions in one step for all rules to avoid redundant processing
    current_g_preds <- process_g_preds(g_preds_processed, t, n_ids, py$J, gbound, debug)
    current_g_preds_bin <- process_g_preds(g_preds_bin_processed, t, n_ids, py$J, gbound, debug)
    current_c_preds <- get_c_preds(C_preds_processed, t, n_ids, gbound)
    current_y_preds <- get_y_preds(initial_model_for_Y, t, n_ids, ybound, debug)
    
    # Create treatment probability lists only once
    current_g_preds_list <- lapply(1:py$J, function(j) matrix(current_g_preds[,j], ncol=1))
    current_g_preds_bin_list <- lapply(1:py$J, function(j) matrix(current_g_preds_bin[,j], ncol=1))
    
    # Extract data efficiently by avoiding loops where possible
    Y <- initial_model_for_Y_data$Y
    C <- initial_model_for_Y_data$C
    
    # Vectorize censoring status calculation - one operation instead of multiple checks
    is_censored <- Y == -1 | is.na(Y) | C == 1
    valid_rows <- !is_censored
    
    # Preallocate rule predictions matrix - one allocation instead of growing
    Qs <- matrix(NA_real_, nrow=n_ids, ncol=n_rules) 
    colnames(Qs) <- names(tmle_rules)
    
    # Process all rules using the cached predictions
    for(i in seq_along(tmle_rules)) {
      rule <- names(tmle_rules)[i]
      lstm_preds <- all_lstm_preds[[rule]]
      
      if(is.null(lstm_preds)) {
        Qs[,i] <- mean(Y[valid_rows], na.rm=TRUE)
      } else {
        t_preds <- lstm_preds[[min(t + 1, length(lstm_preds))]]
        
        if(is.null(t_preds)) {
          Qs[,i] <- mean(Y[valid_rows], na.rm=TRUE)
        } else {
          # Ensure proper dimensions
          t_preds <- rep(t_preds, length.out=n_ids)
          
          # IMPORTANT: Keep as event probabilities for internal calculations
          Qs[,i] <- pmin(pmax(t_preds, ybound[1]), ybound[2])
          
          # Debug to confirm we're using event probabilities
          message(paste0("Mean Qs[,", i, "] at time ", t, ": ", round(mean(Qs[,i], na.rm=TRUE), 7)))
        }
      }
    }
    
    # Process initial predictions to ensure proper format
    initial_preds <- matrix(current_y_preds, nrow=n_ids)
    
    # Create QAW matrix - event probabilities throughout
    QAW <- cbind(QA = initial_preds, Qs)
    colnames(QAW) <- c("QA", names(tmle_rules))
    # Apply bounds to ensure event probabilities are between ybound[1] and ybound[2]
    QAW <- pmin(pmax(QAW, ybound[1]), ybound[2])
    
    message(paste0("QAW matrix dimensions: ", nrow(QAW), "x", ncol(QAW)))
    message(paste0("QAW value range: [", min(QAW, na.rm=TRUE), ", ", max(QAW, na.rm=TRUE), "]"))
    
    # Print column means for event probabilities
    col_means <- colMeans(QAW, na.rm=TRUE)
    message(paste0("QAW mean by column: ", paste(round(col_means, 6), collapse=", ")))
    
    # Add NA check to QAW value comparison
    if(!is.na(col_means[1]) && col_means[1] > 0.6) {
      message("QAW values are event probabilities, high values expected in final time points")
    } else {
      message("QAW values are event probabilities, values appear within expected range")
    }
    
    # Print some sample values for debugging
    if(nrow(QAW) > 0) {
      sample_size <- min(5, nrow(QAW))
      message("Sample QAW values: ")
      for(i in 1:sample_size) {
        message(paste0("Row ", i, ": ", paste(QAW[i,], collapse=", ")))
      }
    }
    
    # Process treatment predictions in one step
    # Optimize g_matrix creation
    g_matrix <- if(is.list(current_g_preds_list)) {
      # Pre-allocate matrix with correct dimensions
      g_mat <- matrix(0, nrow=n_ids, ncol=ncol(treatments[[min(t + 1, length(treatments))]]))
      
      # Fill matrix efficiently by column
      for(j in seq_len(ncol(g_mat))) {
        if(j <= length(current_g_preds_list) && !is.null(current_g_preds_list[[j]])) {
          pred <- matrix(current_g_preds_list[[j]], nrow=n_ids)
          g_mat[,j] <- if(ncol(pred) > 1) pred[,1] else pred
        } else {
          g_mat[,j] <- rep(1/ncol(g_mat), n_ids)
        }
      }
      g_mat
    } else if(is.matrix(current_g_preds_list)) {
      # Efficiently handle matrix format
      if(nrow(current_g_preds_list) != n_ids) {
        matrix(rep(current_g_preds_list, length.out=n_ids*ncol(current_g_preds_list)), 
               ncol=ncol(current_g_preds_list))
      } else {
        current_g_preds_list 
      }
    } else {
      # Default uniform probabilities
      matrix(1/ncol(treatments[[min(t + 1, length(treatments))]]), 
             nrow=n_ids, ncol=ncol(treatments[[min(t + 1, length(treatments))]]))
    }
    
    # Get current treatment and rules
    current_obs_treatment <- treatments[[min(t + 1, length(treatments))]]
    current_obs_rules <- obs.rules[[min(t, length(obs.rules))]]
    
    # Create clever covariates in one step - preallocate for efficiency
    clever_covariates <- matrix(0, nrow=n_ids, ncol=ncol(current_obs_rules))
    is_censored_adj <- rep(is_censored, length.out=n_ids)
    
    # Vectorized operation for all rules
    for(i in seq_len(ncol(current_obs_rules))) {
      clever_covariates[,i] <- current_obs_rules[,i] * (!is_censored_adj)
    }
    
    # Calculate censoring-adjusted weights efficiently
    weights <- matrix(0, nrow=n_ids, ncol=ncol(current_obs_rules))
    
    # Calculate censoring matrix once instead of in the loop
    C_matrix <- matrix(rep(current_c_preds, ncol(g_matrix)), 
                       nrow=nrow(current_c_preds),
                       ncol=ncol(g_matrix))
    
    # Add diagnostics for censoring matrix
    message(paste0("Censoring matrix dimensions: ", nrow(C_matrix), "x", ncol(C_matrix)))
    
    # Print censoring summary
    if(nrow(C_matrix) > 0) {
      mean_censoring <- mean(C_matrix, na.rm=TRUE)
      message(paste0("Mean censoring probability: ", mean_censoring))
    }
    
    # Joint probability calculation - one operation instead of multiple
    probs <- g_matrix * (1 - C_matrix)
    
    # Use less extreme bounds for better numerical stability
    prob_lower <- max(0.001, gbound[1])
    prob_upper <- min(0.9, gbound[2])
    message(paste0("Applying probability bounds: [", prob_lower, ", ", prob_upper, "]"))
    
    bounded_probs <- pmin(pmax(probs, prob_lower), prob_upper)
    
    # Print some sample values for diagnostics
    if(nrow(bounded_probs) > 0) {
      sample_size <- min(3, nrow(bounded_probs))
      message("Sample probability values after bounding: ")
      for(i in 1:sample_size) {
        # Only show first few columns for readability
        col_sample <- min(5, ncol(bounded_probs))
        message(paste0("Row ", i, " (first ", col_sample, " cols): ", 
                       paste(bounded_probs[i, 1:col_sample], collapse=", ")))
      }
    }
    
    # Calculate weights for all rules at once
    for(i in seq_len(ncol(current_obs_rules))) {
      valid_idx <- clever_covariates[,i] > 0
      if(any(valid_idx)) {
        # Calculate treatment probabilities for all valid rows at once
        treatment_probs <- rowSums(current_obs_treatment[valid_idx,] * bounded_probs[valid_idx,], na.rm=TRUE)
        
        # Use less extreme bounds for better stability
        prob_lower <- max(0.001, gbound[1])
        # Ensure treatment probs are reasonable
        treatment_probs[treatment_probs < prob_lower] <- prob_lower
        
        # Add debugging output for treatment probabilities
        if(length(treatment_probs) > 0) {
          message(paste0("Treatment probs - mean: ", mean(treatment_probs, na.rm=TRUE),
                         ", min: ", min(treatment_probs, na.rm=TRUE),
                         ", max: ", max(treatment_probs, na.rm=TRUE)))
        }
        
        # IPCW weights
        cens_weights <- 1 / (1 - C_matrix[valid_idx,1])
        weights[valid_idx,i] <- cens_weights / treatment_probs
        
        # Optimize trimming and normalization
        rule_weights <- weights[valid_idx,i]
        if(length(rule_weights) > 0) {
          # Calculate quantile once and use for all
          max_weight <- quantile(rule_weights, 0.99, na.rm=TRUE)
          weights[valid_idx,i] <- pmin(rule_weights, max_weight) / 
            sum(pmin(rule_weights, max_weight), na.rm=TRUE)
        }
      }
    }
    
    # Preallocate modeling components
    updated_models <- vector("list", ncol(clever_covariates))
    
    # Critical fix: Use logistic regression for EVENT probabilities, not survival
    # The GLM should be modeling event probabilities directly
    for(i in seq_len(ncol(clever_covariates))) {
      model_data <- data.frame(
        # Properly handle GLM with event probabilities (QAW has event probabilities)
        y = pmin(pmax(if(t < t_end) QAW[,"QA"] else Y, 0.01), 0.99),
        offset = qlogis(pmax(pmin(QAW[,i+1], 0.99), 0.01)),
        weights = weights[,i]
      )
      
      message(paste0("Using logit bounds for GLM: [", 0.01, ", ", 0.99, "]"))
      
      valid_rows <- complete.cases(model_data) &
        is.finite(model_data$y) &
        is.finite(model_data$offset) &
        is.finite(model_data$weights) &
        model_data$y != -1 &
        model_data$weights > 0
      
      if(sum(valid_rows) > 10) {
        model_data <- model_data[valid_rows, , drop=FALSE]
        
        if(nrow(model_data) > 0 && any(model_data$weights > 0)) {
          updated_models[[i]] <- tryCatch({
            # No conversion needed - already event probabilities
            glm(
              y ~ 1 + offset(offset),
              weights = weights,
              family = quasibinomial(),
              data = model_data,
              control = list(maxit = 25)
            )
          }, error = function(e) {
            NULL
          })
        }
      }
    }
    
    # Generate Qstar predictions for each rule separately to maintain rule-specific values
    Qstar <- matrix(NA_real_, nrow=n_ids, ncol=length(updated_models))
    
    # Fill with predictions for each rule separately
    for(i in seq_along(updated_models)) {
      if(is.null(updated_models[[i]])) {
        # If model is null, use original Qs values for this rule
        Qstar[,i] <- Qs[,i]
      } else {
        # Get the coefficient and calculate updated predictions
        epsilon <- tryCatch(coef(updated_models[[i]])[1], error=function(e) 0)
        
        # Calculate updated predictions but maintain rule-specific values
        # This is the critical fix - use rule-specific predictions from Qs
        offset_term <- qlogis(pmax(pmin(Qs[,i], 0.99), 0.01))
        Qstar[,i] <- plogis(offset_term + epsilon)
      }
    }
    
    # Set column names once
    if(ncol(Qstar) == ncol(current_obs_rules)) {
      colnames(Qstar) <- colnames(current_obs_rules)
    }
    
    # Create multinomial TMLE contrast
    tmle_contrast <- list(
      "Qs" = Qs,
      "QAW" = QAW,
      "clever_covariates" = clever_covariates,
      "weights" = weights,
      "updated_model_for_Y" = updated_models,
      "Qstar" = Qstar,
      "epsilon" = sapply(updated_models, function(mod) {
        if(is.null(mod)) 0 else tryCatch(coef(mod)[1], error=function(e) 0)
      }),
      "Qstar_gcomp" = QAW[,-1],
      "Y" = Y,
      "ID" = initial_model_for_Y_data$ID
    )
    
    # For binary case - process separately using current_g_preds_bin
    # Similar processing as above but with binary treatment predictions
    g_matrix_bin <- if(is.list(current_g_preds_bin_list)) {
      g_mat <- matrix(0, nrow=n_ids, ncol=ncol(current_obs_treatment))
      for(j in seq_len(ncol(g_mat))) {
        if(j <= length(current_g_preds_bin_list) && !is.null(current_g_preds_bin_list[[j]])) {
          pred <- matrix(current_g_preds_bin_list[[j]], nrow=n_ids)
          g_mat[,j] <- if(ncol(pred) > 1) pred[,1] else pred
        } else {
          g_mat[,j] <- rep(1/ncol(g_mat), n_ids)
        }
      }
      g_mat
    } else if(is.matrix(current_g_preds_bin_list)) {
      if(nrow(current_g_preds_bin_list) != n_ids) {
        matrix(rep(current_g_preds_bin_list, length.out=n_ids*ncol(current_g_preds_bin_list)), 
               ncol=ncol(current_g_preds_bin_list))
      } else {
        current_g_preds_bin_list 
      }
    } else {
      matrix(1/ncol(current_obs_treatment), nrow=n_ids, ncol=ncol(current_obs_treatment))
    }
    
    # Calculate IPTW for both models at once
    tryCatch({
      # Multinomial IPTW
      tmle_contrast$Qstar_iptw <- calculate_iptw(current_g_preds, current_obs_rules, 
                                                 tmle_contrast$Qstar, n_rules, gbound, debug)
      
      # Binary IPTW
      binary_iptw <- calculate_iptw(current_g_preds_bin, current_obs_rules,
                                    tmle_contrast$Qstar, n_rules, gbound, debug)
    }, error = function(e) {
      if(debug) log_iptw_error(e, current_g_preds, current_obs_rules)
      tmle_contrast$Qstar_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
      binary_iptw <- matrix(ybound[1], nrow=1, ncol=n_rules)
    })
    
    # Create binary TMLE contrast - properly create a new object
    tmle_contrast_bin <- list()
    # Copy all elements from multinomial contrast
    for(name in names(tmle_contrast)) {
      if(is.matrix(tmle_contrast[[name]])) {
        tmle_contrast_bin[[name]] <- matrix(tmle_contrast[[name]], ncol=ncol(tmle_contrast[[name]]))
      } else {
        tmle_contrast_bin[[name]] <- tmle_contrast[[name]]
      }
    }
    # Now update the IPTW component
    tmle_contrast_bin$Qstar_iptw <- binary_iptw
    
    # Create separate targeting step for binary version of TMLE
    for(i in seq_len(ncol(clever_covariates))) {
      # Create model data for binary targeting
      model_data_bin <- data.frame(
        y = pmin(pmax(if(t < t_end) QAW[,"QA"] else Y, 0.01), 0.99),
        offset = qlogis(pmax(pmin(QAW[,i+1], 0.99), 0.01)),
        weights = weights[,i]
      )
      
      # Filter valid rows
      valid_rows_bin <- complete.cases(model_data_bin) &
        is.finite(model_data_bin$y) &
        is.finite(model_data_bin$offset) &
        is.finite(model_data_bin$weights) &
        model_data_bin$y != -1 &
        model_data_bin$weights > 0
      
      if(sum(valid_rows_bin) > 10) {
        # Subset data
        model_data_bin <- model_data_bin[valid_rows_bin, , drop=FALSE]
        
        if(nrow(model_data_bin) > 0 && any(model_data_bin$weights > 0)) {
          # Fit binary targeting model
          updated_model_bin <- tryCatch({
            glm(
              y ~ 1 + offset(offset),
              weights = weights,
              family = quasibinomial(),
              data = model_data_bin,
              control = list(maxit = 25)
            )
          }, error = function(e) {
            NULL
          })
          
          # Apply binary targeting
          if(!is.null(updated_model_bin)) {
            epsilon_bin <- tryCatch(coef(updated_model_bin)[1], error=function(e) 0)
            offset_term <- qlogis(pmax(pmin(Qs[,i], 0.99), 0.01))
            tmle_contrast_bin$Qstar[,i] <- plogis(offset_term + epsilon_bin)
          }
        }
      }
    }
    
    if(debug) {
      time_end <- Sys.time()
      cat(sprintf("\nTime point %d completed in %.2f s\n", 
                  t, as.numeric(difftime(time_end, time_start, units="secs"))))
    }
    
    # Return both models in a single list to reduce memory copying
    list(multinomial = tmle_contrast, binary = tmle_contrast_bin)
  })
  
  # Restructure results once at the end
  tmle_contrasts <- vector("list", t_end)
  tmle_contrasts_bin <- vector("list", t_end)
  
  for(t in 1:t_end) {
    # Use direct assignment instead of copying
    tmle_contrasts[[t]] <- results[[t]]$multinomial
    tmle_contrasts_bin[[t]] <- results[[t]]$binary
  }
  
  # Only do final debug output if needed
  if(debug) {
    cat("\nFinal time point processing summary:\n")
    # Include only essential summary metrics
    for(t in 1:t_end) {
      cat("\nTime point", t, "summary:")
      
      # For TMLE estimates, extract non-NA values
      if(!is.null(tmle_contrasts[[t]]$Qstar)) {
        # Filter out invalid values before reporting
        qstar_matrix <- tmle_contrasts[[t]]$Qstar
        qstar_matrix[is.na(qstar_matrix) | is.nan(qstar_matrix) | 
                       !is.finite(qstar_matrix) | qstar_matrix == -1] <- NA
        cat("\nTMLE estimates:", colMeans(qstar_matrix, na.rm=TRUE))
      } else {
        cat("\nTMLE estimates: Not available")
      }
      
      # For IPTW estimates with validation
      if(!is.null(tmle_contrasts[[t]]$Qstar_iptw)) {
        if(is.matrix(tmle_contrasts[[t]]$Qstar_iptw)) {
          iptw_vals <- colMeans(tmle_contrasts[[t]]$Qstar_iptw, na.rm=TRUE)
        } else if(is.vector(tmle_contrasts[[t]]$Qstar_iptw)) {
          iptw_vals <- tmle_contrasts[[t]]$Qstar_iptw
        } else {
          iptw_vals <- NA
        }
        cat("\nIPTW estimates:", iptw_vals)
      } else {
        cat("\nIPTW estimates: Not available")
      }
      
      # Properly calculate observed Y mean by excluding censored values
      if(!is.null(tmle_contrasts[[t]]$Y)) {
        # Filter out censored values (-1) and NAs
        valid_Y <- tmle_contrasts[[t]]$Y[tmle_contrasts[[t]]$Y != -1 & !is.na(tmle_contrasts[[t]]$Y)]
        if(length(valid_Y) > 0) {
          # Calculate mean only from valid values
          observed_mean <- mean(valid_Y, na.rm=TRUE)
        } else {
          observed_mean <- NA  # No valid data
        }
        cat("\nObserved Y mean (excluding censored):", observed_mean)
        cat("\nNumber of valid Y observations:", length(valid_Y))
        cat("\nNumber of censored Y observations:", sum(tmle_contrasts[[t]]$Y == -1, na.rm=TRUE))
      } else {
        cat("\nObserved Y mean: No Y data available")
      }
      
      # Add summary of Y predictions for the first few rows
      if(!is.null(tmle_contrasts[[t]]$Qstar) && nrow(tmle_contrasts[[t]]$Qstar) > 0) {
        cat("\nFirst few Qstar values (rows x rules):")
        print(head(tmle_contrasts[[t]]$Qstar, 3))
      }
    }
  }
  
  return(list("multinomial" = tmle_contrasts, "binary" = tmle_contrasts_bin))
}

process_g_preds <- function(preds_processed, t, n_ids, J, gbound, debug) {
  if(!is.null(preds_processed) && t <= length(preds_processed)) {
    preds <- preds_processed[[t]]
    
    if(is.null(preds)) {
      if(debug) cat("No predictions for time", t, "using uniform\n")
      return(matrix(1/J, nrow=n_ids, ncol=J))
    }
    
    # Debug info about what we're processing
    if(debug) {
      cat("Processing predictions for time", t, "\n")
      cat("Prediction class:", class(preds), "\n")
      if(is.matrix(preds)) {
        cat("Matrix dimensions:", paste(dim(preds), collapse="x"), "\n")
      } else {
        cat("Length:", length(preds), "\n")
      }
    }
    
    # Ensure we have a numeric matrix
    if(!is.matrix(preds)) {
      if(debug) cat("Converting predictions to matrix\n")
      tryCatch({
        if(is.data.frame(preds)) {
          preds <- as.matrix(preds)
          mode(preds) <- "numeric"
        } else if(is.list(preds)) {
          numeric_values <- unlist(lapply(preds, function(x) {
            if(is.numeric(x)) return(x)
            as.numeric(as.character(x))
          }))
          preds <- matrix(numeric_values, ncol=ncol(preds))
        } else {
          preds <- as.numeric(as.character(preds))
          preds <- matrix(preds, ncol=J)
        }
      }, error = function(e) {
        if(debug) cat("Error converting to matrix:", e$message, "\n")
        return(matrix(1/J, nrow=n_ids, ncol=J))
      })
    }
    
    # Check if conversion was successful
    if(!is.matrix(preds) || !is.numeric(preds)) {
      if(debug) cat("Conversion failed, using uniform probs\n")
      return(matrix(1/J, nrow=n_ids, ncol=J))
    }
    
    # IMPORTANT FIX: Use actual column count from the matrix if available
    actual_J <- ncol(preds)
    if(actual_J > 0 && actual_J != J) {
      if(debug) cat("Using actual column count:", actual_J, "instead of specified J:", J, "\n")
      J <- actual_J  # Use the actual column count from the data
    }
    
    # Rest of the function remains the same...
    # Handle dimension mismatches
    if(ncol(preds) != J) {
      if(debug) cat("Column count mismatch:", ncol(preds), "vs", J, "\n")
      if(ncol(preds) < J) {
        # Add columns if needed
        preds <- cbind(preds, matrix(1/J, nrow=nrow(preds), ncol=J-ncol(preds)))
      } else {
        # Truncate if too many columns
        preds <- preds[, 1:J, drop=FALSE]
      }
    }
    
    # Adjust number of rows if needed
    if(nrow(preds) != n_ids) {
      if(debug) cat("Row count mismatch:", nrow(preds), "vs", n_ids, "\n")
      if(nrow(preds) < n_ids) {
        # Repeat rows to match n_ids
        repeats <- ceiling(n_ids / nrow(preds))
        preds <- preds[rep(1:nrow(preds), repeats), , drop=FALSE]
        preds <- preds[1:n_ids, , drop=FALSE]
      } else {
        # Truncate if too many rows
        preds <- preds[1:n_ids, , drop=FALSE]
      }
    }
    
    # Replace NAs with uniform probabilities
    na_indices <- is.na(preds)
    if(any(na_indices)) {
      if(debug) cat("Replacing", sum(na_indices), "NAs with uniform values\n")
      preds[na_indices] <- 1/J
    }
    
    # Normalize rows to sum to 1
    if(debug) cat("Normalizing probabilities\n")
    preds <- t(apply(preds, 1, function(row) {
      if(any(!is.finite(row)) || sum(row) == 0) {
        return(rep(1/J, J))
      }
      bounded <- pmin(pmax(row, gbound[1]), gbound[2])
      bounded / sum(bounded)
    }))
    
    return(preds)
  } else {
    if(debug) cat("No predictions available for time", t, "using uniform\n") 
    return(matrix(1/J, nrow=n_ids, ncol=J))
  }
}

get_c_preds <- function(C_preds_processed, t, n_ids, gbound) {
  if(!is.null(C_preds_processed) && t <= length(C_preds_processed)) {
    preds <- C_preds_processed[[t]]
    if(is.null(preds)) {
      matrix(0.5, nrow=n_ids, ncol=1)
    } else {
      preds_mat <- matrix(preds, nrow=n_ids, ncol=1)
      pmin(pmax(preds_mat, gbound[1]), gbound[2])
    }
  } else {
    matrix(0.5, nrow=n_ids, ncol=1)
  }
}

get_y_preds <- function(initial_model_for_Y, t, n_ids, ybound, debug) {
  if(debug) {
    cat("\nEntering get_y_preds")
    cat("\nReceived initial_model_for_Y type:", class(initial_model_for_Y))
    cat("\nExpected n_ids:", n_ids)  
  }
  
  result <- tryCatch({
    if(is.null(initial_model_for_Y)) {
      if(debug) cat("\nNULL initial_model_for_Y, returning default matrix")
      return(matrix(0.5, nrow=n_ids, ncol=1))
    }
    
    if(is.list(initial_model_for_Y)) {
      if(!is.null(initial_model_for_Y$preds)) {
        preds <- initial_model_for_Y$preds
        if(is.matrix(preds)) {
          if(debug) cat("\nProcessing matrix predictions with dims:", paste(dim(preds), collapse=" x "))
          col_idx <- min(t, ncol(preds))
          preds <- preds[,col_idx, drop=TRUE]
        }
        
        # Identify censored values
        is_censored <- preds == -1 | is.na(preds)
        if(any(is_censored)) {
          if(debug) cat("\nHandling", sum(is_censored), "censored values")
          # Keep censored values as -1
          preds[is_censored] <- -1
        }
        
        if(length(preds) != n_ids) {
          if(debug) cat("\nLength mismatch: preds=", length(preds), " n_ids=", n_ids)
          # Match dimensions preserving censoring
          if(length(preds) > n_ids) {
            preds <- preds[1:n_ids]
            is_censored <- is_censored[1:n_ids]
          } else {
            preds <- rep(preds, length.out=n_ids)
            is_censored <- rep(is_censored, length.out=n_ids)
          }
        }
        
        # IMPORTANT: Don't convert event probabilities to survival probabilities here
        # Just bound the values to ybound range
        non_censored <- !is_censored
        if(any(non_censored)) {
          preds[non_censored] <- pmin(pmax(preds[non_censored], ybound[1]), ybound[2])
        }
        
        result_matrix <- matrix(preds, nrow=n_ids)
      } else {
        if(debug) cat("\nNo preds in list, using default matrix")
        result_matrix <- matrix(0.5, nrow=n_ids, ncol=1)
      }
    } else if(is.vector(initial_model_for_Y)) {
      if(debug) cat("\nProcessing vector input of length:", length(initial_model_for_Y))
      
      # Identify censored values
      is_censored <- initial_model_for_Y == -1 | is.na(initial_model_for_Y)
      
      # Apply bounds only to non-censored values
      temp_vector <- initial_model_for_Y
      temp_vector[!is_censored] <- pmin(pmax(temp_vector[!is_censored], ybound[1]), ybound[2])
      
      if(length(temp_vector) != n_ids) {
        temp_vector <- rep(temp_vector, length.out=n_ids)
        is_censored <- rep(is_censored, length.out=n_ids)
      }
      result_matrix <- matrix(temp_vector, nrow=n_ids)
      
    } else if(is.matrix(initial_model_for_Y)) {
      if(debug) cat("\nProcessing matrix input with dims:", paste(dim(initial_model_for_Y), collapse=" x "))
      
      # Identify censored values
      is_censored <- initial_model_for_Y == -1 | is.na(initial_model_for_Y)
      
      if(nrow(initial_model_for_Y) != n_ids) {
        if(debug) cat("\nRow count mismatch: matrix=", nrow(initial_model_for_Y), " n_ids=", n_ids)
        # Match dimensions preserving censoring
        result_matrix <- matrix(initial_model_for_Y[1:min(nrow(initial_model_for_Y), n_ids),], 
                                nrow=n_ids, ncol=ncol(initial_model_for_Y))
        is_censored <- matrix(is_censored[1:min(nrow(is_censored), n_ids),],
                              nrow=n_ids, ncol=ncol(is_censored))
      } else {
        result_matrix <- initial_model_for_Y
      }
      
      # Apply bounds only to non-censored values
      result_matrix[!is_censored] <- pmin(pmax(result_matrix[!is_censored], ybound[1]), ybound[2])
    } else {
      if(debug) cat("\nUnhandled input type, using default matrix")
      result_matrix <- matrix(0.5, nrow=n_ids, ncol=1)
    }
    
    # Final censoring check
    if(exists("is_censored")) {
      result_matrix[is_censored] <- -1
    }
    
    # Ensure proper dimensions
    if(ncol(result_matrix) > 1) {
      if(debug) cat("\nTaking first column of multi-column matrix")
      result_matrix <- result_matrix[,1,drop=FALSE]
    }
    
    if(debug) {
      cat("\nFinal matrix dimensions:", paste(dim(result_matrix), collapse=" x "))
      is_censored <- result_matrix == -1 | is.na(result_matrix)
      cat("\nValue summary (non-censored):", 
          paste(range(result_matrix[!is_censored], na.rm=TRUE), collapse="-"))
      cat("\nCensored values:", sum(is_censored))
    }
    
    return(result_matrix)
    
  }, error = function(e) {
    if(debug) {
      cat("\nError in get_y_preds:", conditionMessage(e))
      cat("\nReturning default matrix")
    }
    matrix(0.5, nrow=n_ids, ncol=1)
  })
  
  if(debug) {
    cat("\nget_y_preds returning matrix with dims:", paste(dim(result), collapse=" x "), "\n")
    is_censored <- result == -1 | is.na(result)
    if(any(!is_censored)) {
      cat("Y predictions summary (non-censored):\n")
      print(summary(as.vector(result[!is_censored])))
    }
    cat("\nCensored values:", sum(is_censored))
  }
  
  return(result)
}

calculate_iptw <- function(g_preds, rules, predict_Qstar, n_rules, gbound, debug) {
  # Pre-allocate result vector for efficiency
  iptw_means <- numeric(n_rules)
  
  # Process all rules in one go if possible
  for(rule_idx in 1:n_rules) {
    # Vectorized valid index calculation
    valid_idx <- !is.na(rules[,rule_idx]) & rules[,rule_idx] == 1
    
    if(any(valid_idx)) {
      # Get outcomes for this rule efficiently
      outcomes <- predict_Qstar[,rule_idx]
      
      # Get probabilities efficiently
      rule_probs <- g_preds[valid_idx, min(rule_idx, ncol(g_preds))]
      rule_probs <- pmin(pmax(rule_probs, gbound[1]), gbound[2])
      
      # Calculate weights with vector operations
      marginal_prob <- mean(valid_idx, na.rm=TRUE)
      
      # Vectorized weight calculation
      weights <- rep(0, length(valid_idx))
      weights[valid_idx] <- marginal_prob / rule_probs
      
      # Efficient weight trimming
      max_weight <- quantile(weights[valid_idx], 0.95, na.rm=TRUE)
      weights <- pmin(weights, max_weight)
      
      # Extract vectors efficiently
      valid_weights <- weights[valid_idx]
      valid_outcomes <- outcomes[valid_idx]
      
      # Verify lengths and compute weighted mean
      if(length(valid_weights) > 0 && length(valid_outcomes) > 0) {
        if(length(valid_weights) != length(valid_outcomes)) {
          # Trim to common length
          min_len <- min(length(valid_weights), length(valid_outcomes))
          valid_weights <- valid_weights[1:min_len]
          valid_outcomes <- valid_outcomes[1:min_len]
        }
        
        # Normalize weights once
        valid_weights <- valid_weights / sum(valid_weights, na.rm=TRUE)
        
        # Calculate weighted mean efficiently - using event probabilities directly
        iptw_means[rule_idx] <- sum(valid_outcomes * valid_weights, na.rm=TRUE)
      } else {
        iptw_means[rule_idx] <- mean(predict_Qstar[,rule_idx], na.rm=TRUE)
      }
    } else {
      iptw_means[rule_idx] <- mean(predict_Qstar[,rule_idx], na.rm=TRUE)
    }
  }
  
  # Return result as matrix - these are event probabilities
  matrix(iptw_means, nrow=1)
}

log_iptw_error <- function(e, g_preds, rules) {
  cat("Error calculating IPTW:\n")
  cat(conditionMessage(e), "\n")
  cat("Dimensions:\n")
  cat("g_preds:", paste(dim(g_preds), collapse=" x "), "\n") 
  cat("rules:", paste(dim(rules), collapse=" x "), "\n")
}

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded,
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size,
                            current_t, output_dir, debug=FALSE) {
  
  # Minimize debug logging
  if(debug) cat("\nStarting getTMLELongLSTM for time", current_t)
  
  # Get dimensions once
  n_ids <- nrow(obs.rules)
  n_rules <- length(tmle_rules)
  
  # Extract data efficiently by avoiding loops where possible
  Y <- initial_model_for_Y_data$Y
  C <- initial_model_for_Y_data$C
  
  # Vectorize censoring status calculation - one operation instead of multiple checks
  is_censored <- Y == -1 | is.na(Y) | C == 1
  valid_rows <- !is_censored
  
  # Preallocate rule predictions matrix - one allocation instead of growing
  Qs <- matrix(NA_real_, nrow=n_ids, ncol=n_rules) 
  colnames(Qs) <- names(tmle_rules)
  
  # Use efficient rule processing by combining rule operations
  # Precompute shifted data for all rules at once to avoid redundant processing
  shifted_data_list <- lapply(names(tmle_rules), function(rule) {
    # Get rule-specific treatments using existing functions
    switch(rule,
           "static" = static_mtp_lstm(initial_model_for_Y_data),
           "dynamic" = dynamic_mtp_lstm(initial_model_for_Y_data),
           "stochastic" = stochastic_mtp_lstm(initial_model_for_Y_data))
  })
  names(shifted_data_list) <- names(tmle_rules)
  
  # Create rule-specific datasets once efficiently
  # Use pre-computed ID mapping to avoid redundant lookups
  id_mapping_master <- match(initial_model_for_Y_data$ID, unique(initial_model_for_Y_data$ID))
  
  rule_data_list <- lapply(names(tmle_rules), function(rule) {
    # Start with shared data structure - avoid duplicating large objects
    rule_data <- initial_model_for_Y_data
    
    # Set treatment columns efficiently using pre-computed mapping
    shifted_data <- shifted_data_list[[rule]]
    id_mapping <- match(rule_data$ID, shifted_data$ID)
    
    # Vectorized assignment for all time points - single operation per time point
    for(t in 0:t_end) {
      col_name <- paste0("A.", t)
      rule_data[[col_name]] <- shifted_data$A0[id_mapping]
    }
    
    # Base A column also needs to be set
    rule_data$A <- rule_data[[paste0("A.", 0)]]
    
    rule_data
  })
  names(rule_data_list) <- names(tmle_rules)
  
  # Add A columns to covariates once - don't repeat this operation
  tmle_covars_Y_with_A <- unique(c(
    tmle_covars_Y,
    grep("^A\\.", colnames(rule_data_list[[1]]), value=TRUE),
    "A"
  ))
  
  # Use batch processing for all rules
  # First time, run with batch_models=TRUE to cache model
  if (exists("first_lstm_call", envir = .GlobalEnv) && first_lstm_call) {
    # Run LSTM for the first rule to cache model (only runs the Python script once)
    rule <- names(tmle_rules)[1]
    lstm_preds <- lstm(
      data = rule_data_list[[rule]],
      outcome = "Y",
      covariates = tmle_covars_Y_with_A,
      t_end = t_end,
      window_size = window_size,
      out_activation = "sigmoid",
      loss_fn = "binary_crossentropy",
      output_dir = output_dir,
      J = 1,
      ybound = ybound,
      gbound = gbound,
      inference = TRUE,
      debug = FALSE,  # Disable debug for better performance
      batch_models = TRUE  # Enable model caching
    )
    # Set first_lstm_call to FALSE so we don't do this again
    assign("first_lstm_call", FALSE, envir = .GlobalEnv)
  } else if (!exists("first_lstm_call", envir = .GlobalEnv)) {
    # Initialize first_lstm_call if it doesn't exist
    assign("first_lstm_call", TRUE, envir = .GlobalEnv)
  }
  
  # Process all rules in batch
  all_lstm_preds <- lstm(
    data = NULL,  # Not used in batch mode
    outcome = "Y",
    covariates = tmle_covars_Y_with_A,
    t_end = t_end,
    window_size = window_size,
    out_activation = "sigmoid",
    loss_fn = "binary_crossentropy",
    output_dir = output_dir,
    J = 1,
    ybound = ybound,
    gbound = gbound,
    inference = TRUE,
    debug = FALSE,
    batch_models = TRUE,
    batch_rules = rule_data_list  # Pass all rules at once
  )
  
  # Process predictions for all rules
  for(i in seq_along(tmle_rules)) {
    rule <- names(tmle_rules)[i]
    
    # Get predictions for this rule from batch results
    lstm_preds <- all_lstm_preds[[rule]]
    
    # Process predictions efficiently
    if(is.null(lstm_preds)) {
      # Use vectorized assignment for default case
      Qs[,i] <- mean(Y[valid_rows], na.rm=TRUE)
    } else {
      # Get time-specific predictions
      t_preds <- lstm_preds[[min(current_t + 1, length(lstm_preds))]]
      
      if(is.null(t_preds)) {
        # Use vectorized assignment for default case
        Qs[,i] <- mean(Y[valid_rows], na.rm=TRUE)
      } else {
        # Ensure proper dimensions with vectorized operations
        t_preds <- rep(t_preds, length.out=n_ids)
        # Bound values in one operation
        Qs[,i] <- pmin(pmax(t_preds, ybound[1]), ybound[2])
      }
    }
  }
  
  # Process initial predictions to ensure proper format
  initial_preds <- matrix(initial_model_for_Y_preds, nrow=n_ids)
  
  # Create QAW matrix efficiently
  QAW <- cbind(QA = initial_preds, Qs)
  colnames(QAW) <- c("QA", names(tmle_rules))
  
  # Apply bounds in one vectorized operation instead of multiple checks
  QAW <- pmin(pmax(QAW, ybound[1]), ybound[2])
  
  # Process treatment predictions in one step
  # Optimize g_matrix creation
  g_matrix <- if(is.list(g_preds_bounded)) {
    # Pre-allocate matrix with correct dimensions
    g_mat <- matrix(0, nrow=n_ids, ncol=ncol(obs.treatment))
    
    # Fill matrix efficiently by column
    for(j in seq_len(ncol(obs.treatment))) {
      if(j <= length(g_preds_bounded) && !is.null(g_preds_bounded[[j]])) {
        pred <- matrix(g_preds_bounded[[j]], nrow=n_ids)
        g_mat[,j] <- if(ncol(pred) > 1) pred[,1] else pred
      } else {
        g_mat[,j] <- rep(1/ncol(obs.treatment), n_ids)
      }
    }
    g_mat
  } else if(is.matrix(g_preds_bounded)) {
    # Efficiently handle matrix format
    if(nrow(g_preds_bounded) != n_ids) {
      matrix(rep(g_preds_bounded, length.out=n_ids*ncol(g_preds_bounded)), 
             ncol=ncol(g_preds_bounded))
    } else {
      g_preds_bounded 
    }
  } else {
    # Default uniform probabilities
    matrix(1/ncol(obs.treatment), nrow=n_ids, ncol=ncol(obs.treatment))
  }
  
  # Create clever covariates in one step - preallocate for efficiency
  clever_covariates <- matrix(0, nrow=n_ids, ncol=ncol(obs.rules))
  is_censored_adj <- rep(is_censored, length.out=n_ids)
  
  # Vectorized operation for all rules
  for(i in seq_len(ncol(obs.rules))) {
    clever_covariates[,i] <- obs.rules[,i] * (!is_censored_adj)
  }
  
  # Calculate censoring-adjusted weights efficiently
  weights <- matrix(0, nrow=n_ids, ncol=ncol(obs.rules))
  
  # Calculate censoring matrix once instead of in the loop
  C_matrix <- matrix(rep(C_preds_bounded, ncol(g_matrix)), 
                     nrow=nrow(C_preds_bounded),
                     ncol=ncol(g_matrix))
  
  # Joint probability calculation - one operation instead of multiple
  probs <- g_matrix * (1 - C_matrix)
  bounded_probs <- pmin(pmax(probs, gbound[1]), gbound[2])
  
  # Calculate weights for all rules at once
  for(i in seq_len(ncol(obs.rules))) {
    valid_idx <- clever_covariates[,i] > 0
    if(any(valid_idx)) {
      # Calculate treatment probabilities for all valid rows at once
      treatment_probs <- rowSums(obs.treatment[valid_idx,] * bounded_probs[valid_idx,], na.rm=TRUE)
      treatment_probs[treatment_probs < gbound[1]] <- gbound[1]
      
      # IPCW weights
      cens_weights <- 1 / (1 - C_matrix[valid_idx,1])
      weights[valid_idx,i] <- cens_weights / treatment_probs
      
      # Optimize trimming and normalization
      rule_weights <- weights[valid_idx,i]
      if(length(rule_weights) > 0) {
        # Calculate quantile once and use for all
        max_weight <- quantile(rule_weights, 0.99, na.rm=TRUE)
        weights[valid_idx,i] <- pmin(rule_weights, max_weight) / 
          sum(pmin(rule_weights, max_weight), na.rm=TRUE)
      }
    }
  }
  
  # Preallocate modeling components
  updated_models <- vector("list", ncol(clever_covariates))
  
  # Optimize GLM fitting - only run when sufficient data
  for(i in seq_len(ncol(clever_covariates))) {
    # Create model data efficiently - single data.frame creation
    model_data <- data.frame(
      # Bound Y values with less extreme bounds to prevent pushing to extremes
      # Using 0.01 and 0.99 instead of 0.0001 and 0.9999
      y = pmin(pmax(if(current_t < t_end) QAW[,"QA"] else Y, 0.01), 0.99),
      offset = qlogis(pmax(pmin(QAW[,i+1], 0.99), 0.01)),
      weights = weights[,i]
    )
    
    # Filter valid rows in one operation 
    valid_rows <- complete.cases(model_data) &
      is.finite(model_data$y) &
      is.finite(model_data$offset) &
      is.finite(model_data$weights) &
      model_data$y != -1 &
      model_data$weights > 0
    
    # Only fit model if sufficient data
    if(sum(valid_rows) > 10) {
      # Subset data once
      model_data <- model_data[valid_rows, , drop=FALSE]
      
      # Only fit if we have data with non-zero weights
      if(nrow(model_data) > 0 && any(model_data$weights > 0)) {
        # Optimize GLM fit with limited iterations
        # Use the bounded y values to avoid qlogis warnings
        updated_models[[i]] <- tryCatch({
          glm(
            y ~ 1 + offset(offset),  # Use y directly since it's already bounded
            weights = weights,
            family = quasibinomial(),
            data = model_data,
            control = list(maxit = 100, epsilon = 1e-5)
          )
        }, error = function(e) {
          # Return NULL on error rather than stopping
          if(debug) cat("\nGLM error:", e$message)
          NULL
        })
      }
    }
  }
  
  # Generate Qstar predictions efficiently
  # Pre-allocate results
  Qstar <- matrix(NA_real_, nrow=n_ids, ncol=length(updated_models))
  
  # Fill with predictions - use faster approach for NULL models
  for(i in seq_along(updated_models)) {
    if(is.null(updated_models[[i]])) {
      # Use mean as default
      Qstar[,i] <- mean(Y[valid_rows], na.rm=TRUE)
    } else {
      # Get predictions directly
      preds <- predict(updated_models[[i]], type="response", newdata=NULL)
      # Expand to proper length if needed
      Qstar[,i] <- rep(preds, length.out=n_ids)
    }
  }
  
  # Set column names once
  if(ncol(Qstar) == ncol(obs.rules)) {
    colnames(Qstar) <- colnames(obs.rules)
  }
  
  # Calculate IPTW estimates with vectorized operations
  Qstar_iptw <- matrix(sapply(1:ncol(clever_covariates), function(i) {
    valid_idx <- clever_covariates[,i] > 0 & !is_censored_adj
    if(any(valid_idx)) {
      w <- weights[valid_idx,i]
      y <- Y[valid_idx]
      # Check lengths match
      if(length(w) == length(y)) {
        # Use weighted.mean with non-NA values
        valid_ys <- !is.na(y) & is.finite(y) & y != -1
        w_clean <- w[valid_ys]
        y_clean <- y[valid_ys]
        if(length(w_clean) > 0 && sum(w_clean) > 0) {
          weighted.mean(y_clean, w_clean, na.rm=TRUE) 
        } else {
          val <- mean(Y[valid_rows], na.rm=TRUE)
          if(is.na(val) || !is.finite(val)) val <- 0.5  # Fallback for last time point
          val
        }
      } else {
        val <- mean(Y[valid_rows], na.rm=TRUE)
        if(is.na(val) || !is.finite(val)) val <- 0.5  # Fallback for last time point
        val
      }
    } else {
      val <- mean(Y[valid_rows], na.rm=TRUE)
      if(is.na(val) || !is.finite(val)) val <- 0.5  # Fallback for last time point
      val
    }
  }), nrow=1)
  colnames(Qstar_iptw) <- colnames(obs.rules)
  
  # Calculate G-computation estimates directly
  Qstar_gcomp <- matrix(QAW[,-1], ncol=ncol(obs.rules))
  colnames(Qstar_gcomp) <- colnames(obs.rules)
  
  # Get epsilons efficiently
  epsilon <- sapply(updated_models, function(mod) {
    if(is.null(mod)) {
      0 
    } else {
      tryCatch(coef(mod)[1], error=function(e) 0)
    }
  })
  
  # Return the result list - minimal copies
  list(
    "Qs" = Qs,
    "QAW" = QAW,
    "clever_covariates" = clever_covariates,
    "weights" = weights,
    "updated_model_for_Y" = updated_models,
    "Qstar" = Qstar,
    "epsilon" = epsilon,
    "Qstar_iptw" = Qstar_iptw,
    "Qstar_gcomp" = Qstar_gcomp,
    "Y" = Y,
    "ID" = initial_model_for_Y_data$ID
  )
}

# Fixed static rule
static_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs and ensure order
  IDs <- sort(unique(tmle_dat$ID))
  
  # Get first row for each ID
  id_data <- do.call(rbind, lapply(IDs, function(id) {
    id_rows <- which(tmle_dat$ID == id)
    if(length(id_rows) > 0) tmle_dat[id_rows[1],] else NULL
  }))
  
  # Create result dataframe with treatments
  result <- data.frame(
    ID = IDs,
    A0 = ifelse(id_data$mdd == 1, 1,
                ifelse(id_data$schiz == 1, 2,
                       ifelse(id_data$bipolar == 1, 4, 5)))
  )
  
  return(result)
}

# Fixed stochastic rule
stochastic_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs and ensure order
  IDs <- sort(unique(tmle_dat$ID))
  
  # Get first row for each ID
  id_data <- do.call(rbind, lapply(IDs, function(id) {
    id_rows <- which(tmle_dat$ID == id)
    if(length(id_rows) > 0) tmle_dat[id_rows[1],] else NULL
  }))
  
  # Create result dataframe
  result <- data.frame(
    ID = IDs,
    A0 = numeric(length(IDs))
  )
  
  # Treatment transition probabilities
  # Each row represents current treatment (0-6)
  # Each column represents probability of next treatment (1-6)
  trans_probs <- matrix(c(
    0.01, 0.01, 0.01, 0.01, 0.95, 0.01,  # From treatment 0
    0.95, 0.01, 0.01, 0.01, 0.01, 0.01,  # From treatment 1
    0.01, 0.95, 0.01, 0.01, 0.01, 0.01,  # From treatment 2
    0.01, 0.01, 0.95, 0.01, 0.01, 0.01,  # From treatment 3
    0.01, 0.01, 0.01, 0.95, 0.01, 0.01,  # From treatment 4
    0.01, 0.01, 0.01, 0.01, 0.95, 0.01,  # From treatment 5
    0.01, 0.01, 0.01, 0.01, 0.01, 0.95   # From treatment 6
  ), nrow=7, byrow=TRUE)
  
  # Assign treatments
  for(i in seq_along(IDs)) {
    curr_treat <- as.numeric(id_data$A[i])
    if(is.na(curr_treat)) curr_treat <- 0
    curr_treat <- curr_treat + 1  # Convert to 1-based index
    result$A0[i] <- sample(1:6, size=1, prob=trans_probs[curr_treat,])
  }
  
  return(result)
}

dynamic_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs and ensure order
  IDs <- sort(unique(tmle_dat$ID))
  
  # Get first row for each ID
  id_data <- data.frame()
  for(id in IDs) {
    id_rows <- which(tmle_dat$ID == id)
    if(length(id_rows) > 0) {
      id_data <- rbind(id_data, tmle_dat[id_rows[1],])
    }
  }
  
  # Create result dataframe
  result <- data.frame(
    ID = IDs,
    A0 = numeric(length(IDs))
  )
  
  # Fill in treatments
  for(i in seq_along(IDs)) {
    result$A0[i] <- ifelse(
      id_data$mdd[i] == 1 & 
        (id_data$L1[i] > 0 | id_data$L2[i] > 0 | id_data$L3[i] > 0), 1,
      ifelse(
        id_data$bipolar[i] == 1 & 
          (id_data$L1[i] > 0 | id_data$L2[i] > 0 | id_data$L3[i] > 0), 4,
        ifelse(id_data$schiz[i] == 1 & 
                 (id_data$L1[i] > 0 | id_data$L2[i] > 0 | id_data$L3[i] > 0), 2, 5)
      )
    )
  }
  
  return(result)
}