###################################################################
# treatment regime functions                                      #
###################################################################

static_mtp_lstm <- function(row){ 
  # Static: Everyone gets quetiap (if bipolar), halo (if schizophrenia), ari (if MDD) and stays on it
  shifted <- factor(0, levels=levels(row$A))
  if(row$schiz==1){
    shifted <- factor(2, levels=levels(row$A))
  }else if(row$bipolar==1){
    shifted <- factor(4, levels=levels(row$A))
  }else if(row$mdd==1){
    shifted <- factor(1, levels=levels(row$A))
  }else if(row$schiz==-1 | row$bipolar==-1 | row$mdd==-1){
    shifted <- factor(0, levels=levels(row$A))
  }
  return(shifted)
}

dynamic_mtp_lstm <- function(row){ 
  # Dynamic: Everyone starts with risp.
  # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to quetiap. (if bipolar), halo. (if schizophrenia), ari (if MDD); otherwise stay on risp.
  shifted <- factor(0, levels=levels(row$A))
  if(row$t==0){ 
    shifted <- factor(5, levels=levels(row$A))
  }else if(row$t>=1){
    if (!is.na(row$L1) && !is.na(row$L2) && !is.na(row$L3) && (row$L1 > 0 | row$L2 > 0 | row$L3 > 0)) {
      if(row$schiz==1){
        shifted <- factor(2, levels=levels(row$A)) # switch to halo
      }else if(row$bipolar==1){
        shifted <- factor(4, levels=levels(row$A)) # switch to quet.
      }else if(row$mdd==1){
        shifted <- factor(1, levels=levels(row$A)) # switch to arip
      }else if(row$schiz==-1 | row$bipolar==-1 | row$mdd==-1){
        shifted <- factor(0, levels=levels(row$A))
      }
    }else{
      shifted <- factor(5, levels=levels(row$A))  # otherwise stay on risp.
    }
  }
  return(shifted)
}

stochastic_mtp_lstm <- function(row){
  # Stochastic: at each t>0, 95% chance of staying with treatment at t-1, 5% chance of randomly switching according to Multinomial distribution
  shifted <- factor(0, levels=levels(row$A))
  if(row$t == 0) { 
    # Do nothing in the first period
    shifted <- row$A
  } else {
    if(row$A == 0) {
      # Do nothing if current treatment is 0
      shifted <- row$A 
    } else {
      # Calculate probabilities
      prob_vector <- StochasticFun(row$A, d = c(0,0,0,0,0,0))
      if(all(prob_vector <= 0)) {
        shifted <- row$A # Fallback in case of non-positive probabilities
      } else {
        # Generate a new treatment based on the probability vector
        shifted <- which(rmultinom(1, 1, prob_vector) == 1)
      }
    }
  }
  return(shifted)
}

###################################################################
# TMLE targeting step:                                            #
# estimate each treatment rule-specific mean                      #
###################################################################

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end, window.size) {
  
  library(parallel)
  library(data.table)
  
  C <- initial_model_for_Y_data$C
  
  n_rows <- nrow(initial_model_for_Y_data)
  batch_size <- 10000
  num_batches <- ceiling(n_rows / batch_size)
  
  batch_results <- vector("list", length = length(names(tmle_rules)))
  names(batch_results) <- names(tmle_rules)
  
  for (rule in names(tmle_rules)) {
    batch_results[[rule]] <- vector("list", num_batches)
    
    for (i in 1:num_batches) {
      batch_indices <- ((i - 1) * batch_size + 1):min(i * batch_size, n_rows)
      batch_data <- initial_model_for_Y_data[batch_indices, ]
      
      batch_results[[rule]][[i]] <- mclapply(seq_len(nrow(batch_data)), function(j) {
        row_df <- batch_data[j, , drop = FALSE]
        as.data.frame(tmle_rules[[rule]](row_df))  # Convert the result to a data.frame
      }, mc.cores = detectCores())
      
      batch_results[[rule]][[i]] <- rbindlist(batch_results[[rule]][[i]])  # Combine the data.frames
    }
    
    shifted <- rbindlist(batch_results[[rule]])
    
    data <- initial_model_for_Y_data[, !grepl("^A", names(initial_model_for_Y_data))]
    
    newdata <- cbind(data, shifted)
    
    # Check if there are any columns starting with "A"
    if (any(grepl("^A", names(newdata)))) {
      newdata <- newdata %>%
        pivot_longer(cols = starts_with("A"), names_to = "t", values_to = "A") %>%
        pivot_wider(names_from = t, values_from = A)
    }
    
    # Get the names of columns starting with "Y"
    y_cols <- grep("^Y", names(newdata), value = TRUE)
    
    # Combine the "Y" columns with the specified covariates
    selected_cols <- c(y_cols, tmle_covars_Y)
    
    # Filter out any non-existing columns
    selected_cols <- selected_cols[selected_cols %in% names(newdata)]
    
    print("Begin to update Qs")
    assign(paste0("Q_", rule), lstm(data = newdata[, selected_cols], outcome = y_cols, covariates = tmle_covars_Y, t_end = t.end, window_size = window.size, out_activation = "sigmoid", loss_fn = "binary_crossentropy", output_dir, J = J))
    print("Updated Qs")
    rm(data, shifted, newdata)
    print("Cleared temporary data")
  }
  
  # Rest of the code remains the same
  
  return(list("Qs" = Qs, 
              "QAW" = QAW, 
              "clever_covariates" = clever_covariates, 
              "weights" = weights, 
              "updated_model_for_Y" = updated_model_for_Y, 
              "Qstar" = Qstar, 
              "Qstar_iptw" = Qstar_iptw, 
              "Qstar_gcomp" = Qstar_gcomp, 
              "ID" = initial_model_for_Y_data$ID))
}