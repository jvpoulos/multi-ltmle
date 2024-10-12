###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end, window.size) {
  
  window.size <- length(initial_model_for_Y_preds) - t.end
  
  # Reshape tmle_data to wide format
  tmle_data_wide <- pivot_wider(initial_model_for_Y_data, names_from = "covariate", values_from = "value", id_cols = "ID")
  
  # Ensure all required columns exist
  required_cols <- c("ID", "Y")
  if (!all(required_cols %in% colnames(tmle_data_wide))) {
    stop(paste("Missing required columns:", paste(setdiff(required_cols, colnames(tmle_data_wide)), collapse = ", ")))
  }
  
  # Get IDs sorted by 't' and 'ID'
  IDs <- tmle_data_wide %>% arrange(ID) %>% pull(ID)
  
  # Pre-allocate space for predictions
  predicted_Y <- predict_Qstar <- matrix(NA, nrow = length(IDs), ncol = length(tmle_rules))
  observed_Y <- rep(NA, length(IDs))
  epsilon <- rep(0, length(IDs))
  
  for (i in seq_along(tmle_rules)) {
    # Subset observed rule covariate
    obs.rule <- obs.rules[, i]
    
    pred_g <- g_preds_bounded[IDs, obs.treatment[obs.rule == 1]] # observed
    pred_C <- C_preds_bounded[IDs] # censoring probs.
    
    HAW <- tmle_rules[[i]](tmle_data_wide)
    
    # Ensure HAW is not empty
    if (nrow(HAW) == 0) {
      next
    }
    
    HAW <- HAW[!is.na(HAW[, 1]), ] # exclude subjects that are censored before this time point
    
    # Obtain initial predictions
    pred_Q <- initial_model_for_Y_preds
    
    observed_Y[as.character(IDs) %in% as.character(HAW[,"ID"])] <- tmle_data_wide[tmle_data_wide$ID %in% HAW$ID, "Y"] # add observed
    
    predicted_Y[, i] <- predict_Qstar[, i] <- pred_Q  # add predictions
    
    # Clever covariate universal least favorable submodel 
    H <- (1 / boundProbs(pred_g * pred_C, gbound)) * HAW[, obs.treatment[obs.rule == 1]]
    
    # TMLE update
    suppressWarnings(epsilon <- coef(glm(observed_Y ~ -1 + offset(qlogis(pred_Q)) + H, family = binomial)))
    
    predict_Qstar[, i] <- plogis(qlogis(pred_Q) + epsilon * H)
  }
  
  list(
    "Qstar" = boundProbs(predict_Qstar, ybound),
    "epsilon" = epsilon,
    "Qstar_gcomp" = boundProbs(predicted_Y, ybound),
    "Qstar_iptw" = sapply(seq_len(ncol(predict_Qstar)), function(i) boundProbs(weighted.mean(x = predicted_Y[, i], w = (1 / boundProbs(g_preds_bounded[IDs, obs.treatment[obs.rules[, i] == 1]], gbound)), na.rm = TRUE), ybound)),
    "Y" = observed_Y
  )
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
