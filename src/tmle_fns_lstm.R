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
  
  # Process each time point
  for(t in 1:t_end) {
    if(debug) {
      print(paste("Processing time point", t))
      print(paste("obs.rules length:", length(obs.rules)))
      print(paste("treatments length:", length(treatments)))
    }
    
    # Get current predictions safely
    current_g_preds <- if(!is.null(g_preds_processed) && length(g_preds_processed) >= t) {
      g_preds_processed[[t]]
    } else {
      if(debug) print(paste("Warning: No g predictions for time", t))
      NULL
    }
    
    current_C_preds <- if(!is.null(C_preds_processed) && length(C_preds_processed) >= t) {
      C_preds_processed[[t]]
    } else {
      if(debug) print(paste("Warning: No C predictions for time", t))
      NULL
    }
    
    # Get current treatments and rules safely
    current_treatment <- treatments[[min(t + 1, length(treatments))]]
    current_rules <- obs.rules[[min(t, length(obs.rules))]]
    
    # Process single timepoint with error handling
    tmle_contrasts[[t]] <- tryCatch({
      getTMLELongLSTM(
        initial_model_for_Y_preds = initial_model_for_Y,
        initial_model_for_Y_data = initial_model_for_Y_data,
        tmle_rules = tmle_rules,
        tmle_covars_Y = tmle_covars_Y,
        g_preds_bounded = current_g_preds,
        C_preds_bounded = current_C_preds,
        obs.treatment = current_treatment,
        obs.rules = current_rules,
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
      n_ids <- length(unique(initial_model_for_Y_data$ID))
      list(
        "Qstar" = matrix(0.5, nrow=n_ids, ncol=length(tmle_rules)),
        "epsilon" = rep(0, n_ids),
        "Qstar_gcomp" = matrix(0.5, nrow=n_ids, ncol=length(tmle_rules)),
        "Qstar_iptw" = rep(0.5, length(tmle_rules)),
        "Y" = rep(NA, n_ids)
      )
    })
  }
  
  return(tmle_contrasts)
}


# Add safer array handling in getTMLELongLSTM
getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size = 7) {
  
  tryCatch({
    # Input validation & preprocessing
    n_ids <- length(unique(initial_model_for_Y_data$ID))
    
    # Validate and prep input arrays
    prep_array <- function(x, n, default = 0.5) {
      if(is.null(x)) return(rep(default, n))
      if(is.matrix(x)) x <- as.vector(x)
      if(length(x) != n) x <- rep(x[1], n)
      x
    }
    
    initial_model_for_Y_preds <- prep_array(initial_model_for_Y_preds, n_ids)
    g_preds_bounded <- prep_array(g_preds_bounded, n_ids)
    C_preds_bounded <- prep_array(C_preds_bounded, n_ids)
    
    # Initialize matrices with safe dimensions
    n_rules <- length(tmle_rules)
    predicted_Y <- predict_Qstar <- matrix(0.5, nrow=n_ids, ncol=n_rules)
    observed_Y <- rep(NA, n_ids)
    epsilon <- rep(0, n_ids)
    
    # Process each rule
    for(i in seq_len(n_rules)) {
      print(paste("Processing rule", i))
      
      # Get rule predictions
      HAW <- tryCatch({
        rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
        print(paste("Rule result dimensions:", paste(dim(rule_result), collapse=" x ")))
        print("Rule data summary:")
        print(summary(rule_result))
        
        if(!is.null(rule_result)) {
          rule_result[!is.na(rule_result[,1]), , drop=FALSE]
        } else NULL
      }, error = function(e) {
        print(paste("Error in rule", i, ":", e$message))
        NULL
      })
      
      if(!is.null(HAW) && nrow(HAW) > 0) {
        # Create ID mapping
        id_map <- data.frame(
          orig_id = unique(initial_model_for_Y_data$ID),
          new_id = seq_len(n_ids)
        )
        
        # Update predictions
        pred_Q <- initial_model_for_Y_preds
        
        # Get Y values
        y_idx <- match(HAW$ID, initial_model_for_Y_data$ID)
        valid_y_idx <- y_idx[!is.na(y_idx)]
        if(length(valid_y_idx) > 0) {
          valid_ids <- match(HAW[,"ID"], id_map$orig_id)
          valid_ids <- valid_ids[!is.na(valid_ids)]
          
          observed_Y[valid_ids] <- initial_model_for_Y_data$Y[valid_y_idx]
          predicted_Y[,i] <- predict_Qstar[,i] <- pred_Q
        }
        
        # Safe array operations for obs.rules
        if(!is.null(obs.rules) && ncol(obs.rules) >= i) {
          rule_indices <- which(obs.rules[,i] == 1)
          
          if(length(rule_indices) > 0) {
            # Get treatment index safely
            treat_index <- rep(0, n_ids)
            valid_haw_idx <- match(seq_len(n_ids), id_map$new_id)
            valid_haw_idx <- valid_haw_idx[!is.na(valid_haw_idx)]
            
            if(length(valid_haw_idx) > 0) {
              treat_values <- as.numeric(HAW$A0[valid_haw_idx])
              treat_values <- treat_values[!is.na(treat_values)]
              
              if(length(treat_values) > 0) {
                treat_index[valid_haw_idx] <- treat_values
              }
            }
            
            print("Treatment index summary:")
            print(table(treat_index[rule_indices], useNA="ifany"))
            
            # Calculate denominator
            denom <- boundProbs(
              g_preds_bounded[rule_indices] * C_preds_bounded[rule_indices],
              gbound
            )
            
            print("Denominator summary:")
            print(summary(denom))
            
            # Calculate H
            H <- numeric(n_ids)
            H[rule_indices] <- (1 / denom) * treat_index[rule_indices]
            
            print("H summary:")
            print(summary(H))
            
            # Update epsilon
            if(!all(H == 0) && !all(is.na(H))) {
              epsilon_model <- tryCatch({
                glm(
                  observed_Y ~ -1 + offset(qlogis(pred_Q)) + H,
                  family = binomial()
                )
              }, error = function(e) {
                print(paste("Error in epsilon calculation:", e$message))
                NULL
              })
              
              if(!is.null(epsilon_model)) {
                epsilon <- coef(epsilon_model)[1]
                print(paste("Epsilon:", epsilon))
                predict_Qstar[,i] <- plogis(qlogis(pred_Q) + epsilon * H)
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
    print("Error in getTMLELongLSTM:")
    print(e$message)
    # Return safe default values
    list(
      "Qstar" = matrix(0.5, nrow=n_ids, ncol=length(tmle_rules)),
      "epsilon" = rep(0, n_ids),
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