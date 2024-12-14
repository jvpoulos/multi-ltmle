###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

process_time_points <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size,
                                cores = 1, debug = FALSE) {
  
  # Initialize results list
  tmle_contrasts <- vector("list", t_end)
  
  # Validate time indices
  if(length(obs.rules) < t_end) {
    if(debug) print("Warning: obs.rules shorter than t_end, padding with last value")
    last_rule <- obs.rules[[length(obs.rules)]]
    obs.rules <- c(obs.rules, replicate(t_end - length(obs.rules), last_rule, simplify = FALSE))
  }
  
  if(length(treatments) < t_end + 1) {
    if(debug) print("Warning: treatments shorter than t_end + 1, padding with last value")
    last_treatment <- treatments[[length(treatments)]] 
    treatments <- c(treatments, replicate(t_end + 1 - length(treatments), last_treatment, simplify = FALSE))
  }
  
  # Pre-process predictions to ensure correct dimensions
  preprocess_preds <- function(preds, t, n_ids) {
    if(is.null(preds) || length(preds) == 0) {
      return(matrix(0.5, nrow = n_ids, ncol = 1))
    }
    
    if(!is.matrix(preds)) {
      preds <- matrix(preds, ncol = 1)
    }
    
    if(nrow(preds) != n_ids) {
      if(nrow(preds) < n_ids) {
        # Pad with last value
        padding <- matrix(tail(preds, 1), nrow = n_ids - nrow(preds), ncol = ncol(preds))
        preds <- rbind(preds, padding)
      } else {
        # Truncate
        preds <- preds[1:n_ids, , drop = FALSE]
      }
    }
    
    return(preds)
  }
  
  # Get dimensions
  n_ids <- length(unique(initial_model_for_Y_data$ID))
  
  # Process each time point
  for(t in 1:t_end) {
    if(debug) {
      print(paste("Processing time point", t))
    }
    
    # Get current predictions safely
    current_g_preds <- if(!is.null(g_preds_processed) && length(g_preds_processed) >= t) {
      preprocess_preds(g_preds_processed[[t]], t, n_ids)
    } else {
      if(debug) print(paste("Warning: No g predictions for time", t))
      matrix(0.5, nrow = n_ids, ncol = 1)
    }
    
    current_C_preds <- if(!is.null(C_preds_processed) && length(C_preds_processed) >= t) {
      preprocess_preds(C_preds_processed[[t]], t, n_ids)
    } else {
      if(debug) print(paste("Warning: No C predictions for time", t))
      matrix(0.5, nrow = n_ids, ncol = 1)
    }
    
    # Get Y predictions for current time
    current_Y_preds <- if(is.matrix(initial_model_for_Y)) {
      initial_model_for_Y[, min(t, ncol(initial_model_for_Y)), drop = FALSE]
    } else {
      matrix(initial_model_for_Y[min(t, length(initial_model_for_Y))], ncol = 1)
    }
    
    current_Y_preds <- preprocess_preds(current_Y_preds, t, n_ids)
    
    # Process single timepoint with error handling
    tmle_contrasts[[t]] <- tryCatch({
      getTMLELongLSTM(
        initial_model_for_Y_preds = current_Y_preds,
        initial_model_for_Y_data = initial_model_for_Y_data,
        tmle_rules = tmle_rules,
        tmle_covars_Y = tmle_covars_Y,
        g_preds_bounded = current_g_preds,
        C_preds_bounded = current_C_preds,
        obs.treatment = treatments[[min(t + 1, length(treatments))]],
        obs.rules = obs.rules[[min(t, length(obs.rules))]],
        gbound = gbound,
        ybound = ybound,
        t_end = t_end,
        window_size = window_size
      )
    }, error = function(e) {
      if(debug) {
        print(paste("Error processing time point", t, ":", e$message))
        print("Returning default values")
      }
      # Return default values on error
      list(
        "Qstar" = matrix(0.5, nrow = n_ids, ncol = length(tmle_rules)),
        "epsilon" = rep(0, n_ids),
        "Qstar_gcomp" = matrix(0.5, nrow = n_ids, ncol = length(tmle_rules)),
        "Qstar_iptw" = rep(0.5, length(tmle_rules)),
        "Y" = rep(NA, n_ids)
      )
    })
  }
  
  return(tmle_contrasts)
}


# Update getTMLELongLSTM with proper Y value bounding
getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size = 7) {
  tryCatch({
    # Input validation & preprocessing
    n_ids <- length(unique(initial_model_for_Y_data$ID))
    
    # Validate and prep input arrays
    prep_array <- function(x, n, default = 0.5) {
      if (is.null(x)) return(rep(default, n))
      if (is.matrix(x)) x <- as.vector(x)
      if (length(x) != n) x <- rep(x[1], n)
      x
    }
    
    initial_model_for_Y_preds <- prep_array(initial_model_for_Y_preds, n_ids)
    g_preds_bounded <- prep_array(g_preds_bounded, n_ids)
    C_preds_bounded <- prep_array(C_preds_bounded, n_ids)
    
    # Initialize matrices with safe dimensions
    n_rules <- length(tmle_rules)
    predicted_Y <- predict_Qstar <- matrix(0.5, nrow = n_ids, ncol = n_rules)
    observed_Y <- rep(NA, n_ids)
    epsilon <- rep(0, n_rules)
    
    # Get observed outcomes from data and bound them
    if ("Y" %in% colnames(initial_model_for_Y_data)) {
      y_values <- initial_model_for_Y_data$Y
      id_values <- initial_model_for_Y_data$ID
      
      # Ensure Y values are bounded between 0 and 1
      y_values <- pmin(pmax(y_values, ybound[1]), ybound[2])
      
      # Match IDs and assign Y values
      for (i in seq_along(id_values)) {
        if (!is.na(y_values[i])) {
          idx <- which(unique(initial_model_for_Y_data$ID) == id_values[i])
          if (length(idx) > 0) {
            observed_Y[idx] <- y_values[i]
          }
        }
      }
    }
    
    # Process each rule
    for (i in seq_len(n_rules)) {
      print(paste("Processing rule", i))
      
      rule_result <- tryCatch({
        result <- tmle_rules[[i]](initial_model_for_Y_data)
        if(!is.null(result)) {
          result[!is.na(result[,1]), , drop=FALSE]
        } else NULL
      }, error = function(e) {
        print(paste("Error in rule", i, ":", e$message))
        NULL
      })
      
      if (!is.null(rule_result) && nrow(rule_result) > 0) {
        print(paste("Rule result dimensions:", paste(dim(rule_result), collapse = " x ")))
        print("Rule data summary:")
        print(summary(rule_result))
        
        # Create ID mapping
        id_map <- data.frame(
          orig_id = unique(initial_model_for_Y_data$ID),
          new_id = seq_len(n_ids)
        )
        
        # Get valid indices
        valid_indices <- match(rule_result$ID, id_map$orig_id)
        valid_indices <- valid_indices[!is.na(valid_indices)]
        
        if (length(valid_indices) > 0) {
          # Initialize H vector
          H <- numeric(n_ids)
          
          # Calculate denominator for valid indices
          denom <- boundProbs(
            g_preds_bounded[valid_indices] * C_preds_bounded[valid_indices],
            gbound
          )
          
          # Get treatment values for valid indices
          treat_values <- as.numeric(rule_result$A0[match(id_map$orig_id[valid_indices], rule_result$ID)])
          
          # Calculate H
          H[valid_indices] <- (1 / denom) * treat_values
          
          print("Denominator summary:")
          print(summary(denom))
          print("H summary:")
          print(summary(H[valid_indices]))
          
          # Prepare data for GLM with bounded Y values
          valid_obs <- !is.na(observed_Y[valid_indices])
          if (any(valid_obs)) {
            y_bounded <- pmin(pmax(observed_Y[valid_indices][valid_obs], ybound[1]), ybound[2])
            
            # Create GLM data with bounded Y values
            glm_data <- data.frame(
              y = y_bounded,
              h = H[valid_indices][valid_obs],
              offset = qlogis(initial_model_for_Y_preds[valid_indices][valid_obs])
            )
            
            # Ensure offset is finite
            glm_data$offset[!is.finite(glm_data$offset)] <- 0
            
            # Fit epsilon model
            if (nrow(glm_data) > 0) {
              glm_model <- tryCatch({
                glm(y ~ h + offset(offset), family = binomial(), data = glm_data)
              }, error = function(e) {
                print(paste("Error fitting epsilon model:", e$message))
                NULL
              })
              
              if (!is.null(glm_model)) {
                # Extract and bound epsilon
                epsilon[i] <- coef(glm_model)["h"]
                if (!is.finite(epsilon[i])) epsilon[i] <- 0
                
                print(paste("Epsilon:", epsilon[i]))
                
                # Update predictions
                logit_pred <- qlogis(initial_model_for_Y_preds) + epsilon[i] * H
                predict_Qstar[,i] <- plogis(logit_pred)
                predicted_Y[,i] <- initial_model_for_Y_preds
              }
            }
          }
        }
      }
    }
    
    # Return results with bounds checking
    list(
      "Qstar" = boundProbs(predict_Qstar, ybound),
      "epsilon" = epsilon,
      "Qstar_gcomp" = boundProbs(predicted_Y, ybound),
      "Qstar_iptw" = sapply(seq_len(ncol(predict_Qstar)), function(i) {
        boundProbs(mean(predicted_Y[,i], na.rm=TRUE), ybound)
      }),
      "Y" = observed_Y
    )
    
  }, error = function(e) {
    print(paste("Error in getTMLELongLSTM:", e$message))
    # Return safe default values
    list(
      "Qstar" = matrix(0.5, nrow=n_ids, ncol=length(tmle_rules)),
      "epsilon" = rep(0, n_rules),
      "Qstar_gcomp" = matrix(0.5, nrow=n_ids, ncol=length(tmle_rules)),
      "Qstar_iptw" = rep(0.5, length(tmle_rules)),
      "Y" = rep(NA, n_ids)
    )
  })
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