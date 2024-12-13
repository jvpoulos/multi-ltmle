###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

# Fast array preprocessing with dimension checks
prep_array <- function(x, n_ids, n_cols = 1, default = 0.5) {
  if(is.null(x)) return(matrix(default, nrow = n_ids, ncol = n_cols))
  
  if(!is.matrix(x)) {
    if(length(x) == n_ids) {
      x <- matrix(x, ncol = 1)
    } else {
      x <- matrix(rep_len(x, n_ids), ncol = 1)
    }
  }
  
  if(ncol(x) < n_cols) {
    x <- cbind(x, matrix(x[,ncol(x)], nrow = nrow(x), ncol = n_cols - ncol(x)))
  }
  
  if(nrow(x) != n_ids) {
    x <- matrix(rep_len(as.vector(x), n_ids * n_cols), nrow = n_ids)
  }
  
  return(x)
}

# Optimized getTMLELongLSTM
getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, 
                            tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, 
                            obs.treatment, obs.rules, gbound, ybound, t_end, window_size = 7) {
  
  # Fast initialization
  n_ids <- length(unique(initial_model_for_Y_data$ID))
  n_rules <- length(tmle_rules)
  unique_ids <- sort(unique(initial_model_for_Y_data$ID))
  id_map <- match(initial_model_for_Y_data$ID, unique_ids)
  
  # Pre-allocate arrays with proper dimensions
  predicted_Y <- predict_Qstar <- matrix(0.5, nrow = n_ids, ncol = n_rules)
  observed_Y <- rep(NA_real_, n_ids)
  epsilon <- rep(0, n_rules)
  
  # Process Y values once
  if("Y" %in% names(initial_model_for_Y_data)) {
    y_values <- pmin(pmax(initial_model_for_Y_data$Y, ybound[1]), ybound[2])
    observed_Y[id_map] <- y_values
  }
  
  # Ensure predictions have correct dimensions
  initial_Y_preds <- prep_array(initial_model_for_Y_preds, n_ids)
  g_preds <- prep_array(g_preds_bounded, n_ids)
  c_preds <- prep_array(C_preds_bounded, n_ids)
  
  # Process rules in parallel if possible
  use_parallel <- requireNamespace("parallel", quietly = TRUE) && 
    parallel::detectCores() > 1
  
  process_rule <- function(i) {
    tryCatch({
      # Get rule result
      rule_result <- tmle_rules[[i]](initial_model_for_Y_data)
      if(is.null(rule_result) || nrow(rule_result) == 0) return(NULL)
      
      # Match IDs efficiently
      valid_ids <- match(rule_result$ID, unique_ids)
      valid_ids <- valid_ids[!is.na(valid_ids)]
      if(length(valid_ids) == 0) return(NULL)
      
      # Calculate H vector efficiently
      denom <- boundProbs(g_preds[valid_ids] * c_preds[valid_ids], gbound)
      treat_values <- as.numeric(rule_result$A0[match(unique_ids[valid_ids], rule_result$ID)])
      
      H <- numeric(n_ids)
      H[valid_ids] <- (1 / denom) * treat_values
      
      # Process valid observations
      valid_obs <- !is.na(observed_Y[valid_ids])
      if(!any(valid_obs)) return(NULL)
      
      filtered_ids <- valid_ids[valid_obs]
      glm_data <- data.frame(
        y = observed_Y[filtered_ids],
        h = H[filtered_ids],
        offset = qlogis(initial_Y_preds[filtered_ids])
      )
      
      # Fit GLM and calculate predictions
      glm_fit <- glm(y ~ h + offset(offset), family = binomial(), data = glm_data)
      eps <- coef(glm_fit)["h"]
      
      if(is.finite(eps)) {
        list(
          epsilon = eps,
          predictions = plogis(qlogis(initial_Y_preds) + eps * H),
          orig_preds = initial_Y_preds
        )
      } else NULL
      
    }, error = function(e) NULL)
  }
  
  # Process rules either in parallel or sequentially
  if(use_parallel) {
    results <- parallel::mclapply(seq_len(n_rules), process_rule, 
                                  mc.cores = parallel::detectCores() - 1)
  } else {
    results <- lapply(seq_len(n_rules), process_rule)
  }
  
  # Update results with proper dimension checks
  for(i in seq_len(n_rules)) {
    if(!is.null(results[[i]])) {
      epsilon[i] <- results[[i]]$epsilon
      predict_Qstar[,i] <- results[[i]]$predictions
      predicted_Y[,i] <- results[[i]]$orig_preds[,1]
    }
  }
  
  # Return results with efficient bounds checking
  list(
    "Qstar" = boundProbs(predict_Qstar, ybound),
    "epsilon" = epsilon,
    "Qstar_gcomp" = boundProbs(predicted_Y, ybound),
    "Qstar_iptw" = apply(predicted_Y, 2, function(x) boundProbs(mean(x, na.rm=TRUE), ybound)),
    "Y" = observed_Y
  )
}

# Optimized process_time_points
process_time_points <- function(initial_model_for_Y, initial_model_for_Y_data, 
                                tmle_rules, tmle_covars_Y, 
                                g_preds_processed, C_preds_processed,
                                treatments, obs.rules, 
                                gbound, ybound, t_end, window_size,
                                cores = 1, debug = FALSE) {
  
  # Initialize with proper dimensions
  n_ids <- length(unique(initial_model_for_Y_data$ID))
  
  # Pre-process predictions once
  prep_predictions <- function(preds, t) {
    if(is.null(preds) || length(preds) == 0 || t > length(preds)) {
      matrix(0.5, nrow = n_ids, ncol = 1)
    } else {
      if(!is.matrix(preds[[t]])) {
        matrix(preds[[t]], ncol = 1)
      } else {
        preds[[t]]
      }
    }
  }
  
  # Process timepoints efficiently
  process_timepoint <- function(t) {
    # Get predictions for current time
    g_preds <- prep_predictions(g_preds_processed, t)
    c_preds <- prep_predictions(C_preds_processed, t)
    
    # Get Y predictions
    y_preds <- if(is.matrix(initial_model_for_Y)) {
      initial_model_for_Y[, min(t, ncol(initial_model_for_Y))]
    } else {
      initial_model_for_Y
    }
    
    # Process timepoint
    getTMLELongLSTM(
      initial_model_for_Y_preds = y_preds,
      initial_model_for_Y_data = initial_model_for_Y_data,
      tmle_rules = tmle_rules,
      tmle_covars_Y = tmle_covars_Y,
      g_preds_bounded = g_preds,
      C_preds_bounded = c_preds,
      obs.treatment = treatments[[min(t + 1, length(treatments))]],
      obs.rules = obs.rules[[min(t, length(obs.rules))]],
      gbound = gbound,
      ybound = ybound,
      t_end = t_end,
      window_size = window_size
    )
  }
  
  # Process timepoints in parallel if possible
  if(cores > 1 && requireNamespace("parallel", quietly = TRUE)) {
    tmle_contrasts <- parallel::mclapply(1:t_end, process_timepoint, 
                                         mc.cores = cores)
  } else {
    tmle_contrasts <- lapply(1:t_end, process_timepoint)
  }
  
  return(tmle_contrasts)
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