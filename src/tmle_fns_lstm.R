###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

process_time_points <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size,
                                cores = 1, debug = FALSE, chunk_size = 1000) {
  
  # Initialize results
  n_ids <- length(unique(initial_model_for_Y_data$ID))
  n_rules <- length(tmle_rules)
  tmle_contrasts <- vector("list", t_end)
  
  if(debug) {
    cat(sprintf("\nProcessing %d IDs with %d rules\n", n_ids, n_rules))
    cat("Initial model for Y structure:\n")
    str(initial_model_for_Y)
  }
  
  # Pre-allocate results
  for(t in 1:t_end) {
    tmle_contrasts[[t]] <- list(
      "Qstar" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "epsilon" = rep(0, n_rules),
      "Qstar_gcomp" = matrix(ybound[1], nrow = n_ids, ncol = n_rules),
      "Qstar_iptw" = matrix(ybound[1], nrow = 1, ncol = n_rules),
      "Y" = rep(ybound[1], n_ids)
    )
  }
  
  # Helper function to ensure matrix dimensions
  ensure_matrix_dims <- function(mat, n_rows, n_cols = 1, default = 0.5) {
    if(is.null(mat)) return(matrix(default, nrow = n_rows, ncol = n_cols))
    
    if(!is.matrix(mat)) mat <- matrix(mat, ncol = n_cols)
    
    if(nrow(mat) != n_rows) {
      if(nrow(mat) < n_rows) {
        # Pad with last value
        mat <- matrix(rep(mat[1,], length.out = n_rows), 
                      nrow = n_rows, ncol = ncol(mat), byrow = TRUE)
      } else {
        # Truncate
        mat <- mat[1:n_rows,, drop = FALSE]
      }
    }
    
    if(ncol(mat) != n_cols) {
      if(ncol(mat) < n_cols) {
        # Repeat columns
        mat <- matrix(rep(mat, n_cols), nrow = n_rows, ncol = n_cols)
      } else {
        # Take first n_cols columns
        mat <- mat[,1:n_cols, drop = FALSE]
      }
    }
    
    return(mat)
  }
  
  # Create lookup table once
  id_map <- data.frame(
    orig_id = sort(unique(initial_model_for_Y_data$ID)),
    new_id = seq_len(n_ids)
  )
  id_lookup <- match(initial_model_for_Y_data$ID, id_map$orig_id)
  
  # Process sequentially by time point
  for(t in 1:t_end) {
    if(debug) cat(sprintf("\nProcessing time point %d/%d\n", t, t_end))
    time_start <- Sys.time()
    
    # Get predictions for current time point
    current_g_preds <- if(!is.null(g_preds_processed) && t <= length(g_preds_processed)) {
      ensure_matrix_dims(g_preds_processed[[t]], n_ids)
    } else {
      matrix(0.5, nrow = n_ids, ncol = 1)
    }
    
    current_c_preds <- if(!is.null(C_preds_processed) && t <= length(C_preds_processed)) {
      ensure_matrix_dims(C_preds_processed[[t]], n_ids)
    } else {
      matrix(0.5, nrow = n_ids, ncol = 1)
    }
    
    # Handle Y predictions carefully
    current_y_preds <- tryCatch({
      if(is.list(initial_model_for_Y) && !is.null(initial_model_for_Y$preds)) {
        if(is.matrix(initial_model_for_Y$preds)) {
          ensure_matrix_dims(initial_model_for_Y$preds[, min(t, ncol(initial_model_for_Y$preds))], n_ids)
        } else {
          ensure_matrix_dims(initial_model_for_Y$preds, n_ids)
        }
      } else if(is.matrix(initial_model_for_Y)) {
        ensure_matrix_dims(initial_model_for_Y[, min(t, ncol(initial_model_for_Y))], n_ids)
      } else {
        ensure_matrix_dims(initial_model_for_Y, n_ids)
      }
    }, error = function(e) {
      if(debug) cat("Error getting Y predictions:", conditionMessage(e), "\n")
      matrix(0.5, nrow = n_ids, ncol = 1)
    })
    
    if(debug) {
      cat("Prediction dimensions:\n")
      cat("G preds:", paste(dim(current_g_preds), collapse="x"), "\n")
      cat("C preds:", paste(dim(current_c_preds), collapse="x"), "\n")
      cat("Y preds:", paste(dim(current_y_preds), collapse="x"), "\n")
    }
    
    # Process in chunks
    chunk_size <- min(chunk_size, ceiling(n_ids/10))
    chunks <- split(1:n_ids, ceiling(seq_along(1:n_ids)/chunk_size))
    
    # Process each chunk
    for(chunk_idx in seq_along(chunks)) {
      current_ids <- chunks[[chunk_idx]]
      
      if(debug && chunk_idx %% 5 == 0) {
        cat(sprintf("Processing chunk %d/%d\n", chunk_idx, length(chunks)))
      }
      
      # Get chunk data safely
      tryCatch({
        chunk_mask <- id_lookup %in% current_ids
        chunk_data <- initial_model_for_Y_data[chunk_mask, ]
        
        chunk_g_preds <- ensure_matrix_dims(current_g_preds[current_ids,], length(current_ids))
        chunk_c_preds <- ensure_matrix_dims(current_c_preds[current_ids,], length(current_ids))
        chunk_y_preds <- ensure_matrix_dims(current_y_preds[current_ids,], length(current_ids))
        
        # Process chunk
        result <- getTMLELongLSTM(
          initial_model_for_Y_preds = chunk_y_preds,
          initial_model_for_Y_data = chunk_data,
          tmle_rules = tmle_rules,
          tmle_covars_Y = tmle_covars_Y,
          g_preds_bounded = chunk_g_preds,
          C_preds_bounded = chunk_c_preds,
          obs.treatment = treatments[[min(t + 1, length(treatments))]],
          obs.rules = obs.rules[[min(t, length(obs.rules))]],
          gbound = gbound,
          ybound = ybound,
          t_end = t_end,
          window_size = window_size,
          debug = debug
        )
        
        # Update results
        tmle_contrasts[[t]]$Qstar[current_ids,] <- result$Qstar
        tmle_contrasts[[t]]$Qstar_gcomp[current_ids,] <- result$Qstar_gcomp
        tmle_contrasts[[t]]$Y[current_ids] <- result$Y
        
        if(any(result$epsilon != 0)) {
          tmle_contrasts[[t]]$epsilon <- result$epsilon
        }
        
      }, error = function(e) {
        if(debug) {
          cat(sprintf("Error in chunk %d: %s\n", chunk_idx, conditionMessage(e)))
          cat("Using default values for this chunk\n")
        }
        # Use default values for this chunk
        tmle_contrasts[[t]]$Qstar[current_ids,] <- ybound[1]
        tmle_contrasts[[t]]$Qstar_gcomp[current_ids,] <- ybound[1]
        tmle_contrasts[[t]]$Y[current_ids] <- ybound[1]
      })
      
      # Clean up
      rm(chunk_data, chunk_g_preds, chunk_c_preds, chunk_y_preds)
      gc()
    }
    
    # Calculate final IPTW means
    tmle_contrasts[[t]]$Qstar_iptw <- matrix(
      colMeans(tmle_contrasts[[t]]$Qstar_gcomp, na.rm = TRUE),
      nrow = 1,
      ncol = n_rules
    )
    
    # Clean up
    rm(current_g_preds, current_c_preds, current_y_preds)
    gc()
    
    if(debug) {
      time_end <- Sys.time()
      cat(sprintf("Time point %d completed in %.2f s\n", 
                  t, as.numeric(difftime(time_end, time_start, units="secs"))))
    }
  }
  
  return(tmle_contrasts)
}

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size = 7,
                            debug = FALSE) {
  
  final_result <- tryCatch({
    # Input validation & preprocessing
    n_ids <- length(unique(initial_model_for_Y_data$ID))
    n_rules <- length(tmle_rules)
    
    if(debug) {
      cat(sprintf("\nProcessing %d IDs with %d rules\n", n_ids, n_rules))
      if(!is.null(initial_model_for_Y_preds)) {
        cat("Initial predictions range:", 
            paste(range(initial_model_for_Y_preds, na.rm=TRUE), collapse="-"), "\n")
      }
      cat("G predictions range:", paste(range(g_preds_bounded, na.rm=TRUE), collapse="-"), "\n")
      cat("C predictions range:", paste(range(C_preds_bounded, na.rm=TRUE), collapse="-"), "\n")
    }
    
    # Process initial predictions
    initial_preds <- if(is.matrix(initial_model_for_Y_preds)) {
      if(ncol(initial_model_for_Y_preds) == 1) {
        rep(initial_model_for_Y_preds[,1], n_rules)
      } else {
        as.vector(initial_model_for_Y_preds)
      }
    } else {
      rep(initial_model_for_Y_preds[1], n_ids * n_rules)
    }
    initial_preds <- matrix(pmin(pmax(initial_preds, ybound[1]), ybound[2]), 
                            nrow = n_ids, ncol = n_rules)
    
    # Initialize with initial predictions
    predicted_Y <- predict_Qstar <- initial_preds
    observed_Y <- rep(NA, n_ids)  # Start with NAs
    epsilon <- rep(0, n_rules)
    
    # Get observed outcomes
    if("Y" %in% colnames(initial_model_for_Y_data)) {
      y_data <- initial_model_for_Y_data
      y_data$Y[is.na(y_data$Y)] <- ybound[1]  # Set NA Y values to lower bound
      y_data$Y <- pmin(pmax(y_data$Y, ybound[1]), ybound[2])
      
      # Match to unique IDs
      unique_ids <- unique(y_data$ID)
      for(id in unique_ids) {
        idx <- which(id == y_data$ID)[1]  # Take first non-NA value for each ID
        if(!is.na(idx)) {
          observed_Y[which(unique_ids == id)] <- y_data$Y[idx]
        }
      }
      
      if(debug) {
        cat("Y values summary:\n")
        print(summary(observed_Y))
        cat("Number of non-NA Y:", sum(!is.na(observed_Y)), "\n")
      }
    }
    
    # Process rules
    for(i in seq_len(n_rules)) {
      if(debug) cat(sprintf("\nProcessing rule %d/%d\n", i, n_rules))
      
      # Get rule results
      rule_result <- tryCatch({
        result <- tmle_rules[[i]](initial_model_for_Y_data)
        if(!is.null(result)) {
          # Validate rule output
          if(debug) {
            cat("Rule result summary:\n")
            print(summary(result$A0))
            cat("Number of valid treatments:", sum(!is.na(result$A0)), "\n")
          }
          result[!is.na(result$ID) & !is.na(result$A0), , drop=FALSE]
        } else NULL
      }, error = function(e) {
        if(debug) cat(sprintf("Error in rule %d: %s\n", i, conditionMessage(e)))
        NULL
      })
      
      if(!is.null(rule_result) && nrow(rule_result) > 0) {
        # Get matched indices
        matched_ids <- match(rule_result$ID, unique(initial_model_for_Y_data$ID))
        valid_ids <- !is.na(matched_ids)
        
        if(any(valid_ids)) {
          valid_indices <- matched_ids[valid_ids]
          
          # Calculate H vector
          H <- numeric(n_ids)
          denom <- g_preds_bounded[valid_indices] * C_preds_bounded[valid_indices]
          denom <- pmax(denom, gbound[1])  # Ensure minimum bound
          
          # Get treatment values
          treat_values <- as.numeric(rule_result$A0[valid_ids])
          if(debug) {
            cat("Treatment values summary:\n")
            print(summary(treat_values))
          }
          
          H[valid_indices] <- (1 / denom) * treat_values
          
          if(debug) {
            cat("\nH vector summary:\n")
            print(summary(H[valid_indices]))
            cat("Number of non-zero H:", sum(H != 0), "\n")
          }
          
          # Prepare GLM data
          valid_y <- !is.na(observed_Y[valid_indices])
          if(any(valid_y)) {
            glm_data <- data.frame(
              y = observed_Y[valid_indices][valid_y],
              h = H[valid_indices][valid_y],
              offset = qlogis(initial_preds[valid_indices, i][valid_y])
            )
            
            # Ensure finite offset
            glm_data$offset[!is.finite(glm_data$offset)] <- 0
            
            if(nrow(glm_data) > 0 && var(glm_data$h) > 0) {
              if(debug) {
                cat("\nFitting GLM with data:\n")
                print(summary(glm_data))
              }
              
              glm_result <- tryCatch({
                fit <- glm(y ~ h + offset(offset), family = binomial(), data = glm_data)
                eps <- coef(fit)["h"]
                if(is.finite(eps) && abs(eps) < 10) eps else 0
              }, error = function(e) {
                if(debug) cat("GLM error:", conditionMessage(e), "\n")
                0
              })
              
              epsilon[i] <- glm_result
              
              if(debug) cat(sprintf("Fitted epsilon[%d]: %f\n", i, epsilon[i]))
              
              # Update predictions
              if(epsilon[i] != 0) {
                logit_pred <- qlogis(initial_preds[,i]) + epsilon[i] * H
                predict_Qstar[,i] <- pmin(pmax(plogis(logit_pred), ybound[1]), ybound[2])
                predicted_Y[,i] <- pmin(pmax(initial_preds[,i], ybound[1]), ybound[2])
                
                if(debug) {
                  cat("\nUpdated predictions summary:\n")
                  print(summary(predict_Qstar[,i]))
                }
              }
            }
          }
        }
      }
    }
    
    # Calculate final IPTW means
    iptw_means <- sapply(1:n_rules, function(r) {
      vals <- predicted_Y[,r]
      mean_val <- mean(vals[!is.na(vals)])
      if(!is.finite(mean_val)) mean_val <- ybound[1]
      mean_val
    })
    
    if(debug) {
      cat("\nFinal summary:\n")
      cat("Qstar range:", paste(range(predict_Qstar), collapse="-"), "\n")
      cat("Qstar_gcomp range:", paste(range(predicted_Y), collapse="-"), "\n")
      cat("Epsilon values:", paste(epsilon, collapse=", "), "\n")
      cat("IPTW means:", paste(iptw_means, collapse=", "), "\n")
    }
    
    # Return results
    list(
      "Qstar" = predict_Qstar,
      "epsilon" = epsilon,
      "Qstar_gcomp" = predicted_Y,
      "Qstar_iptw" = matrix(iptw_means, nrow=1),
      "Y" = observed_Y
    )
    
  }, error = function(e) {
    if(debug) cat(sprintf("\nError in getTMLELongLSTM: %s\n", conditionMessage(e)))
    # Return safe defaults
    list(
      "Qstar" = matrix(ybound[1], nrow=n_ids, ncol=n_rules),
      "epsilon" = rep(0, n_rules),
      "Qstar_gcomp" = matrix(ybound[1], nrow=n_ids, ncol=n_rules),
      "Qstar_iptw" = matrix(ybound[1], nrow=1, ncol=n_rules),
      "Y" = rep(ybound[1], n_ids)
    )
  })
  
  return(final_result)
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