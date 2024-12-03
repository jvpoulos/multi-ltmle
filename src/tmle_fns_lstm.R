###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t.end, window.size) {
  
  print("Debug: Starting getTMLELongLSTM function")
  
  # Ensure initial_model_for_Y_data is a data frame
  initial_model_for_Y_data <- as.data.frame(initial_model_for_Y_data)
  
  # Get unique IDs
  ids <- unique(initial_model_for_Y_data$ID)
  n_ids <- length(ids)
  print(paste("Number of unique IDs:", n_ids))
  
  # Create time sequence
  n_time_points <- t.end + 1
  print(paste("Number of time points:", n_time_points))
  print(paste("Length of predictions:", length(initial_model_for_Y_preds)))
  
  # Process predictions to match number of IDs
  # Create mapping of IDs to prediction indices
  id_map <- match(initial_model_for_Y_data$ID, ids)
  pred_by_id <- numeric(n_ids)
  
  # Calculate mean prediction for each ID
  for(i in 1:n_ids) {
    id_preds <- initial_model_for_Y_preds[id_map == i]
    if(length(id_preds) > 0) {
      pred_by_id[i] <- mean(id_preds, na.rm = TRUE)
    } else {
      pred_by_id[i] <- mean(initial_model_for_Y_preds, na.rm = TRUE)  # Use overall mean if no predictions
    }
  }
  
  # Replace original predictions with ID-averaged predictions
  initial_model_for_Y_preds <- pred_by_id
  
  print(paste("Final prediction length:", length(initial_model_for_Y_preds)))
  
  # Initialize output matrices
  predicted_Y <- predict_Qstar <- matrix(NA, nrow = n_ids, ncol = length(tmle_rules))
  observed_Y <- rep(NA, n_ids)
  epsilon <- rep(0, n_ids)
  
  print("Processing treatment rules...")
  for (i in seq_along(tmle_rules)) {
    print(paste("Processing rule", i))
    
    # Get rule-specific predictions
    HAW <- tryCatch({
      tmle_rules[[i]](initial_model_for_Y_data)
    }, error = function(e) {
      warning(paste("Error in rule", i, ":", e$message))
      return(NULL)
    })
    
    if(is.null(HAW) || nrow(HAW) == 0) next
    
    # Remove NA rows
    HAW <- HAW[!is.na(HAW[,1]), , drop = FALSE]
    if(nrow(HAW) == 0) next
    
    # Update predictions
    pred_Q <- initial_model_for_Y_preds # Now matches n_ids
    valid_ids <- as.character(HAW[,"ID"]) %in% as.character(ids)
    
    if(any(valid_ids)) {
      observed_Y[match(HAW$ID[valid_ids], ids)] <- 
        initial_model_for_Y_data$Y[match(HAW$ID[valid_ids], initial_model_for_Y_data$ID)]
      predicted_Y[,i] <- predict_Qstar[,i] <- pred_Q
    }
    
    # Calculate H and update epsilon
    H <- (1 / boundProbs(g_preds_bounded * C_preds_bounded, gbound)) * 
      HAW[, obs.treatment[obs.rules == 1]]
    
    tryCatch({
      epsilon_model <- glm(observed_Y ~ -1 + offset(qlogis(pred_Q)) + H, 
                           family = binomial)
      epsilon <- coef(epsilon_model)[1]
      predict_Qstar[,i] <- plogis(qlogis(pred_Q) + epsilon * H)
    }, error = function(e) {
      warning(paste("Error in epsilon calculation for rule", i, ":", e$message))
    })
  }
  
  # Return results
  results <- list(
    "Qstar" = boundProbs(predict_Qstar, ybound),
    "epsilon" = epsilon,
    "Qstar_gcomp" = boundProbs(predicted_Y, ybound),
    "Qstar_iptw" = sapply(seq_len(ncol(predict_Qstar)), function(i) {
      boundProbs(weighted.mean(
        predicted_Y[,i],
        w = (1 / boundProbs(g_preds_bounded[ids, obs.treatment[obs.rules[,i] == 1]], gbound)),
        na.rm = TRUE
      ), ybound)
    }),
    "Y" = observed_Y
  )
  
  print("TMLE calculations completed")
  return(results)
}

static_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs
  IDs <- unique(tmle_dat$ID)
  
  # Create index mapping for subsetting
  id_indices <- match(IDs, tmle_dat$ID)
  
  result <- data.frame(
    'ID' = IDs,
    "A0" = ifelse(
      tmle_dat$mdd[id_indices] == 1 & tmle_dat$t[id_indices] == 1, 1,
      ifelse(
        tmle_dat$bipolar[id_indices] == 1 & tmle_dat$t[id_indices] == 1, 4,
        ifelse(tmle_dat$schiz[id_indices] == 1 & tmle_dat$t[id_indices] == 1, 2, 0)
      )
    )
  )
  rownames(result) <- seq_len(nrow(result))
  return(result)
}

dynamic_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs
  IDs <- unique(tmle_dat$ID)
  
  # Create index mapping for subsetting
  id_indices <- match(IDs, tmle_dat$ID)
  
  result <- data.frame(
    'ID' = IDs,
    "A0" = ifelse(
      tmle_dat$mdd[id_indices] == 1 & 
        (tmle_dat$L1[id_indices] > 0 | tmle_dat$L2[id_indices] > 0 | tmle_dat$L3[id_indices] > 0), 1,
      ifelse(
        tmle_dat$bipolar[id_indices] == 1 & 
          (tmle_dat$L1[id_indices] > 0 | tmle_dat$L2[id_indices] > 0 | tmle_dat$L3[id_indices] > 0), 4,
        ifelse(tmle_dat$schiz[id_indices] == 1 & 
                 (tmle_dat$L1[id_indices] > 0 | tmle_dat$L2[id_indices] > 0 | tmle_dat$L3[id_indices] > 0), 2, 5)
      )
    )
  )
  rownames(result) <- seq_len(nrow(result))
  return(result)
}

stochastic_mtp_lstm <- function(tmle_dat) {
  # Get unique IDs 
  IDs <- unique(tmle_dat$ID)
  
  # Create index mapping for subsetting
  id_indices <- match(IDs, tmle_dat$ID)
  
  # Get previous treatment column
  A_cols <- grep("^A$", colnames(tmle_dat), value = TRUE)
  prev_treat <- as.numeric(tmle_dat[id_indices, A_cols])
  
  result <- data.frame(
    'ID' = IDs,
    "A0" = sapply(seq_along(prev_treat), function(x) {
      probs <- c(0, 0.01, 0.01, 0.01, 0.01, 0.96)
      sample(c(1:6), size = 1, prob = probs[prev_treat[x]])
    })
  )
  rownames(result) <- seq_len(nrow(result))
  return(result)
}