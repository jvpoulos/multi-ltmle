###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end, window.size) {
  
  print("Debug: Starting getTMLELongLSTM function")
  
  # Input validation
  print(paste("initial_model_for_Y_preds length:", length(initial_model_for_Y_preds)))
  print(paste("initial_model_for_Y_data dimensions:", paste(dim(initial_model_for_Y_data), collapse=" x ")))
  print(paste("t.end:", t.end))
  print(paste("Original window.size:", window.size))
  
  # Debug initial data structure
  print("First few rows of initial_model_for_Y_data:")
  print(head(initial_model_for_Y_data))
  print("Columns:")
  print(colnames(initial_model_for_Y_data))
  
  # Get unique IDs
  ids <- unique(initial_model_for_Y_data$ID)
  n_ids <- length(ids)
  print(paste("Number of unique IDs:", n_ids))
  
  # Check data dimensions
  n_time_points <- t.end + 1
  expected_rows <- length(ids) * n_time_points
  actual_rows <- nrow(initial_model_for_Y_data)
  print(paste("Expected rows:", expected_rows))
  print(paste("Actual rows:", actual_rows))
  
  # Create time values
  if (!"t" %in% colnames(initial_model_for_Y_data)) {
    print("Creating time sequence...")
    
    # Create sequence for one ID
    time_sequence <- rep(seq_len(n_time_points), each = ceiling(actual_rows/n_ids/n_time_points))
    print(paste("Single ID sequence length:", length(time_sequence)))
    print("First few time values:", paste(head(time_sequence), collapse=", "))
    
    # Replicate for each ID
    time_values <- rep(time_sequence, length.out = actual_rows)
    print(paste("Total time values created:", length(time_values)))
    print(paste("First few full sequence:", paste(head(time_values), collapse=", ")))
    
    # Safety check
    if (length(time_values) != actual_rows) {
      print("WARNING: Length mismatch in time values!")
      print(paste("Time values length:", length(time_values)))
      print(paste("Required length:", actual_rows))
      time_values <- rep(time_sequence, length.out = actual_rows)
    }
    
    # Add time column
    initial_model_for_Y_data$t <- time_values
    print("Time column added")
    print(paste("Updated dimensions:", paste(dim(initial_model_for_Y_data), collapse=" x ")))
  }
  
  # Update window size
  window.size <- length(initial_model_for_Y_preds) - t.end
  print(paste("Updated window.size:", window.size))
  
  # Reshape to wide format
  print("Reshaping data...")
  tmle_data_wide <- initial_model_for_Y_data %>%
    group_by(ID) %>%
    mutate(row_num = row_number()) %>%
    pivot_wider(
      id_cols = c("ID", "row_num"), 
      names_from = "covariate",
      values_from = "value"
    ) %>%
    ungroup() %>%
    select(-row_num)
  
  print("After reshape dimensions:")
  print(paste("Wide format shape:", paste(dim(tmle_data_wide), collapse=" x ")))
  print("Wide format columns:")
  print(colnames(tmle_data_wide))
  
  # Check required columns
  required_cols <- c("ID", "Y")
  if (!all(required_cols %in% colnames(tmle_data_wide))) {
    missing_cols <- setdiff(required_cols, colnames(tmle_data_wide))
    stop(paste("Missing required columns:", paste(missing_cols, collapse=", ")))
  }
  
  # Sort IDs
  IDs <- sort(unique(tmle_data_wide$ID))
  print(paste("Number of unique IDs in wide format:", length(IDs)))
  
  # Allocate output matrices
  predicted_Y <- predict_Qstar <- matrix(NA, nrow = length(IDs), ncol = length(tmle_rules))
  observed_Y <- rep(NA, length(IDs))
  epsilon <- rep(0, length(IDs))
  
  print("Processing treatment rules...")
  for (i in seq_along(tmle_rules)) {
    print(paste("Processing rule", i))
    obs.rule <- obs.rules[, i]
    
    pred_g <- g_preds_bounded[IDs, obs.treatment[obs.rule == 1]]
    pred_C <- C_preds_bounded[IDs]
    
    HAW <- tmle_rules[[i]](tmle_data_wide)
    print(paste("HAW dimensions:", paste(dim(HAW), collapse=" x ")))
    
    if (nrow(HAW) == 0) {
      print("Warning: Empty HAW matrix")
      next
    }
    
    HAW <- HAW[!is.na(HAW[, 1]), ]
    print(paste("Valid HAW rows:", nrow(HAW)))
    
    pred_Q <- initial_model_for_Y_preds
    observed_Y[as.character(IDs) %in% as.character(HAW[,"ID"])] <- tmle_data_wide[tmle_data_wide$ID %in% HAW$ID, "Y"]
    predicted_Y[, i] <- predict_Qstar[, i] <- pred_Q
    
    H <- (1 / boundProbs(pred_g * pred_C, gbound)) * HAW[, obs.treatment[obs.rule == 1]]
    suppressWarnings(epsilon <- coef(glm(observed_Y ~ -1 + offset(qlogis(pred_Q)) + H, family = binomial)))
    predict_Qstar[, i] <- plogis(qlogis(pred_Q) + epsilon * H)
  }
  
  print("Creating final results...")
  results <- list(
    "Qstar" = boundProbs(predict_Qstar, ybound),
    "epsilon" = epsilon,
    "Qstar_gcomp" = boundProbs(predicted_Y, ybound),
    "Qstar_iptw" = sapply(seq_len(ncol(predict_Qstar)), function(i) 
      boundProbs(weighted.mean(predicted_Y[, i], 
                               w = (1 / boundProbs(g_preds_bounded[IDs, obs.treatment[obs.rules[, i] == 1]], gbound)), 
                               na.rm = TRUE), 
                 ybound)),
    "Y" = observed_Y
  )
  
  print("TMLE calculations completed")
  return(results)
}

# Example modifications to treatment rule functions to ensure they handle missing values and edge cases

static_mtp_lstm <- function(tmle_dat) {
  IDs <- tmle_dat %>% group_by(ID) %>% filter(row_number() == n()) %>% ungroup() %>% pull(ID)
  data.frame(
    'ID' = IDs,
    "A0" = ifelse(
      tmle_dat[IDs, 'mdd'] == 1 & tmle_dat[IDs, 't'] == 1, 1,
      ifelse(
        tmle_dat[IDs, 'bipolar'] == 1 & tmle_dat[IDs, 't'] == 1, 4,
        ifelse(tmle_dat[IDs, 'schiz'] == 1 & tmle_dat[IDs, 't'] == 1, 2, 0)
      )
    )
  )
}

dynamic_mtp_lstm <- function(tmle_dat) {
  IDs <- tmle_dat %>% group_by(ID) %>% filter(row_number() == n()) %>% ungroup() %>% pull(ID)
  data.frame(
    'ID' = IDs,
    "A0" = ifelse(
      tmle_dat[IDs, 'mdd'] == 1 & (tmle_dat[IDs, 'L1'] > 0 | tmle_dat[IDs, 'L2'] > 0 | tmle_dat[IDs, 'L3'] > 0), 1,
      ifelse(
        tmle_dat[IDs, 'bipolar'] == 1 & (tmle_dat[IDs, 'L1'] > 0 | tmle_dat[IDs, 'L2'] > 0 | tmle_dat[IDs, 'L3'] > 0), 4,
        ifelse(tmle_dat[IDs, 'schiz'] == 1 & (tmle_dat[IDs, 'L1'] > 0 | tmle_dat[IDs, 'L2'] > 0 | tmle_dat[IDs, 'L3'] > 0), 2, 5)
      )
    )
  )
}

stochastic_mtp_lstm <- function(tmle_dat) {
  IDs <- tmle_dat %>% group_by(ID) %>% filter(row_number() == n()) %>% ungroup() %>% pull(ID)
  prev_treat <- as.numeric(tmle_dat[IDs, grep("A", colnames(tmle_dat), value = TRUE)])
  data.frame(
    'ID' = IDs,
    "A0" = sapply(seq_along(prev_treat), function(x) sample(c(1:6), size = 1, prob = c(0, 0.01, 0.01, 0.01, 0.01, 0.96)[prev_treat[x]]))
  )
}
