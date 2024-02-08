###################################################################
# treatment regime functions                                      #
###################################################################

static_mtp_lstm <- function(row){ 
  # Static: Everyone gets quetiap (if bipolar), halo (if schizophrenia), ari (if MDD) and stays on it
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
  if(row$t==0){ 
    shifted <- factor(5, levels=levels(row$A))
  }else if(row$t>=1){
    if((row$L1 >0 | row$L2 >0 | row$L3 >0)){
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

getTMLELongLSTM <- function(initial_model_for_Y_preds, initial_model_for_Y_data, tmle_rules, tmle_covars_Y, g_preds_bounded, C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end, window.size){
  
  C <- initial_model_for_Y$data$C # 1=Censored
  
  for(rule in names(tmle_rules)){
    # Determine batch size
    batch_size <- 2000  # Adjust based on your data size and memory constraints
    
    # Number of batches
    num_batches <- ceiling(nrow(initial_model_for_Y_data) / batch_size)
    
    # Initialize an empty list to store batch results
    batch_results <- vector("list", num_batches)
    
    # Process each batch
    for (i in 1:num_batches) {
      # Calculate row indices for the current batch
      batch_indices <- ((i - 1) * batch_size + 1):min(i * batch_size, nrow(initial_model_for_Y_data))
      
      # Extract the batch data
      batch_data <- initial_model_for_Y_data[batch_indices, ]
      
      # Apply the function to the batch and store the result
      batch_results[[i]] <- do.call(rbind, lapply(seq_len(nrow(batch_data)), function(j) {
        # Extract the j-th row as a data frame
        row_df <- batch_data[j, , drop = FALSE]
 
        # Apply the function
        return(tmle_rules[[rule]](row_df))
      }))
    }
    
    # Combine all batch results
    shifted <- do.call(rbind, batch_results)
    
    data <- initial_model_for_Y_data[grep("A",colnames(initial_model_for_Y_data), value=TRUE, invert=TRUE)]
    
    newdata <- cbind(data,"A"=shifted)
    newdata <- reshape(newdata, idvar = "t", timevar = "ID", direction = "wide") # T x N
    assign(paste0("Q_",rule), lstm(data=newdata[c(grep("Y",colnames(newdata),value = TRUE),tmle_covars_Y)], outcome=grep("Y",colnames(newdata),value = TRUE), covariates=tmle_covars_Y, t_end=t.end, window_size=window.size, out_activation="sigmoid", loss_fn = "binary_crossentropy", output_dir, inference=TRUE))
    rm(data,shifted,newdata)
  }
  
  # Initialize Qs with lists of predictions
  Qs <- list("Q_static"=Q_static, "Q_dynamic"=Q_dynamic, "Q_stochastic"=Q_stochastic) # list of 3, each 29 vectors of length n
  
  # Iterate over each rule in Qs and extend the predictions
  for (rule in names(Qs)) {
    # Extend the predictions to have t.end + 1 elements by replicating the first element
    Q_extended <- lapply(1:(t.end + 1), function(i) {
      if (i <= window.size + 1) {
        return(Qs[[rule]][[1]])
      } else {
        return(Qs[[rule]][[i - (window.size + 1)]])
      }
    })
    
    # Reassign the extended predictions back to Qs for the current rule
    Qs[[rule]] <- Q_extended
  }
  
  # Combine the predictions into a matrix
  Qs_matrix <- do.call(cbind, lapply(Qs, function(x) do.call(c, x))) # 370000 by 3
  combined_matrix <- cbind(QA = rep(initial_model_for_Y_preds, each = 37), Qs_matrix)
  
  # Bound the probabilities
  QAW <- data.frame(apply(combined_matrix, 2, boundProbs, bounds = ybound)) # data.frame': 370000 obs. of 4 variables
  
  # inverse probs. only used for subset of patients following treatment rule and uncensored (=0 for those excluded)
  clever_covariates <- (obs.rules[initial_model_for_Y_data$ID,]*(1-C)) # numerator
  
  if(dim(g_preds_bounded)[1]>length(initial_model_for_Y_data$ID)){
    weights <- clever_covariates/rowSums(obs.treatment[initial_model_for_Y_data$ID,]*boundProbs(g_preds_bounded[initial_model_for_Y_data$ID,]*C_preds_bounded[initial_model_for_Y_data$ID], bounds = gbound)) # denominator: clever covariate used as weight in regression
  }else{
    weights <- clever_covariates/rowSums(obs.treatment[initial_model_for_Y_data$ID,]*boundProbs(g_preds_bounded*C_preds_bounded, bounds = gbound)) # denominator: clever covariate used as weight in regression
  }
  
  # Initialize updated_model_for_Y as an empty list
  updated_model_for_Y <- vector("list", ncol(clever_covariates))
  
  # targeting step - refit outcome model using clever covariates
  for (i in 1:ncol(clever_covariates)) {
    print(paste("Processing column:", i))
    
    # Check if the column is numeric and print the result
    is_numeric <- is.numeric(QAW[, i])
    print(paste("Is numeric:", is_numeric))
    
    # Filter rows where Y is not -1 (NA)
    valid_rows <- initial_model_for_Y$data$Y != -1
    
    # Adjusted column index for glm functions (no need to +1 here since we've corrected loop index)
    if (all(initial_model_for_Y$data$t < t.end)) { # use actual Y for t=T
      updated_model_for_Y[[i]] <- glm(QAW$QA[valid_rows] ~ 1 + offset(qlogis(QAW[valid_rows, i])), 
                                      weights = weights[valid_rows, i], 
                                      family = quasibinomial())
    } else {
      updated_model_for_Y[[i]] <- glm(initial_model_for_Y$data$Y[valid_rows] ~ 1 + offset(qlogis(QAW[valid_rows, i])), 
                                      weights = weights[valid_rows, i], 
                                      family = quasibinomial())
    }
  }
  
  Qstar <- lapply(1:ncol(clever_covariates), function(i) predict(updated_model_for_Y[[i]], type="response"))
  names(Qstar) <- colnames(obs.rules)
  
  # IPTW estimate
  Qstar_iptw <- lapply(1:ncol(clever_covariates), function(i) boundProbs(weights[,i]*initial_model_for_Y$data$Y, bound=ybound)) 
  names(Qstar_iptw) <- colnames(obs.rules)
  
  # Assuming QAW is your dataframe and obs.rules is another dataframe or a list whose column names you want to use
  
  # First, ensure QAW has the expected number of columns
  # This is a basic check. You might need a more sophisticated check depending on your data structure
  if (ncol(QAW) < 2) {
    stop("QAW does not have enough columns.")
  }
  
  # Correctly populate Qstar_gcomp only with existing columns from QAW
  # Adjust the lapply function to only iterate over existing columns
  Qstar_gcomp <- lapply(1:(ncol(QAW) - 1), function(i) {
    if (i + 1 <= ncol(QAW)) {
      return(QAW[, i + 1])
    } else {
      return(NULL) # Return NULL for non-existing columns
    }
  })
  
  # Filter out NULL values which represent non-existing columns
  Qstar_gcomp <- Qstar_gcomp[sapply(Qstar_gcomp, is.null) == FALSE]
  
  # Now, assign names to Qstar_gcomp elements based on the available names from obs.rules
  if (length(Qstar_gcomp) <= length(colnames(obs.rules))) {
    names(Qstar_gcomp) <- colnames(obs.rules)[1:length(Qstar_gcomp)]
  } else {
    stop("The number of elements in Qstar_gcomp exceeds the number of columns in obs.rules")
  }
  
  # Debugging prints - Remove or comment out in production code
  print("Converting to list")
  print("Returning predictions")
  for (i in 1:length(Qstar_gcomp)) {
    print(paste("Processing column:", i))
    print(paste("Is numeric:", is.numeric(QAW[, i + 1])))
  }
  
  return(list("Qs"=Qs,"QAW"=QAW,"clever_covariates"=clever_covariates,"weights"=weights,"updated_model_for_Y"=updated_model_for_Y, "Qstar"=Qstar, "Qstar_iptw"=Qstar_iptw, "Qstar_gcomp"=Qstar_gcomp, "ID"=initial_model_for_Y_data$ID))
}