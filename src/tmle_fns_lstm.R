###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t.end, window.size) {
  
  print("Debug: Starting getTMLELongLSTM function")
  
  # Input validation and conversion to proper format
  if(is.matrix(initial_model_for_Y_preds)) {
    initial_model_for_Y_preds <- as.vector(initial_model_for_Y_preds)
  }
  if(is.matrix(g_preds_bounded)) {
    g_preds_bounded <- as.vector(g_preds_bounded)
  }
  if(is.matrix(C_preds_bounded)) {
    C_preds_bounded <- as.vector(C_preds_bounded)
  }
  
  print("Checking input parameters and dimensions:")
  print(paste("initial_model_for_Y_preds length:", length(initial_model_for_Y_preds)))
  print(paste("g_preds_bounded length:", length(g_preds_bounded)))
  print(paste("C_preds_bounded length:", length(C_preds_bounded)))
  print(paste("obs.rules dimensions:", paste(dim(obs.rules), collapse=" x ")))
  
  # Ensure consistent length for predictions
  n_ids <- length(unique(initial_model_for_Y_data$ID))
  
  # Pad or truncate predictions if needed
  pad_or_truncate <- function(x, n) {
    if(length(x) < n) {
      c(x, rep(mean(x, na.rm=TRUE), n - length(x)))
    } else if(length(x) > n) {
      x[1:n]
    } else {
      x
    }
  }
  
  initial_model_for_Y_preds <- pad_or_truncate(initial_model_for_Y_preds, n_ids)
  g_preds_bounded <- pad_or_truncate(g_preds_bounded, n_ids)
  C_preds_bounded <- pad_or_truncate(C_preds_bounded, n_ids)
  
  # Initialize output matrices
  predicted_Y <- predict_Qstar <- matrix(NA, nrow = n_ids, ncol = length(tmle_rules))
  observed_Y <- rep(NA, n_ids)
  epsilon <- rep(0, n_ids)
  
  print("Processing treatment rules...")
  for (i in seq_along(tmle_rules)) {
    print(paste("\nProcessing rule", i))
    
    HAW <- tryCatch({
      rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
      print(paste("Rule result dimensions:", paste(dim(rule_result), collapse=" x ")))
      print("Rule data summary:")
      print(summary(rule_result))
      rule_result
    }, error = function(e) {
      print(paste("Error in rule", i, ":", e$message))
      return(NULL)
    })
    
    if(is.null(HAW) || nrow(HAW) == 0) {
      print(paste("No valid data for rule", i))
      next
    }
    
    # Process NA rows
    HAW <- HAW[!is.na(HAW[,1]), , drop = FALSE]
    print(paste("Processing rule", i, "data"))
    print(paste("Valid rows:", nrow(HAW)))
    
    if(nrow(HAW) == 0) next
    
    # Update predictions
    pred_Q <- initial_model_for_Y_preds
    valid_ids <- match(HAW[,"ID"], unique(initial_model_for_Y_data$ID))
    valid_ids <- valid_ids[!is.na(valid_ids)]
    
    if(length(valid_ids) > 0) {
      observed_Y[valid_ids] <- initial_model_for_Y_data$Y[match(HAW$ID, initial_model_for_Y_data$ID)]
      predicted_Y[,i] <- predict_Qstar[,i] <- pred_Q
    }
    
    # Calculate H and update epsilon
    tryCatch({
      rule_indices <- which(obs.rules[,i] == 1)
      print(paste("Number of rule indices:", length(rule_indices)))
      if(length(rule_indices) == 0) next
      
      treat_index <- as.numeric(HAW$A0[match(seq_len(n_ids), HAW$ID)])
      treat_index <- treat_index[rule_indices]
      print("Treatment index summary:")
      print(table(treat_index, useNA="ifany"))
      
      # Calculate denominator
      denom <- g_preds_bounded[rule_indices] * C_preds_bounded[rule_indices]
      denom <- boundProbs(denom, gbound)
      print("Denominator summary:")
      print(summary(denom))
      
      # Calculate H
      H <- numeric(n_ids)
      H[rule_indices] <- (1 / denom) * treat_index
      print("H summary:")
      print(summary(H))
      
      if(!all(H == 0) && !all(is.na(H))) {
        print("Fitting epsilon model")
        epsilon_model <- glm(observed_Y ~ -1 + offset(qlogis(pred_Q)) + H, 
                             family = binomial)
        epsilon <- coef(epsilon_model)[1]
        predict_Qstar[,i] <- plogis(qlogis(pred_Q) + epsilon * H)
        print(paste("Epsilon:", epsilon))
      }
    }, error = function(e) {
      print("Error in calculations:")
      print(e$message)
    })
  }
  
  # Return results with proper dimensions
  results <- list(
    "Qstar" = boundProbs(predict_Qstar, ybound),
    "epsilon" = epsilon,
    "Qstar_gcomp" = boundProbs(predicted_Y, ybound),
    "Qstar_iptw" = sapply(seq_len(ncol(predict_Qstar)), function(i) {
      boundProbs(weighted.mean(
        predicted_Y[,i],
        w = rep(1, length(predicted_Y[,i])),
        na.rm = TRUE
      ), ybound)
    }),
    "Y" = observed_Y
  )
  
  return(results)
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
  
  # Debug info
  print("Static rule summary:")
  print(paste("Number of IDs:", length(IDs)))
  print(paste("MDD cases:", sum(id_data$mdd == 1)))
  print(paste("Schiz cases:", sum(id_data$schiz == 1)))
  print(paste("Bipolar cases:", sum(id_data$bipolar == 1)))
  
  # Create result dataframe with treatments
  result <- data.frame(
    ID = IDs,
    A0 = ifelse(id_data$mdd == 1, 1,
                ifelse(id_data$schiz == 1, 2,
                       ifelse(id_data$bipolar == 1, 4, 5)))
  )
  
  # Debug treatment assignments
  print("Treatment assignments:")
  print(table(result$A0))
  
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
  
  print("Stochastic rule summary:")
  print(paste("Number of IDs:", length(IDs)))
  print("Treatment assignments:")
  print(table(result$A0))
  
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