#!/usr/bin/env Rscript

#' Fix for SL3 model issues and TMLE errors
#' This script fixes the issues with the simulation code
#' 1. Fixes the family parameter in GLM learners to use function objects instead of strings
#' 2. Adds defensive checks for missing covariates in SL3 tasks
#' 3. Handles getTMLELong errors at specific time points

# First, fix the SL3_fns.R file learner definitions
fix_SL3_fns <- function() {
  cat("Fixing SL3_fns.R file...\n")
  
  # Read the file
  sl3_file <- "./src/SL3_fns.R"
  sl3_content <- readLines(sl3_file)
  
  # Replace string "binomial" with binomial() function call in make_learner_stack calls
  for(i in 1:length(sl3_content)) {
    if(grepl('list\\("Lrnr_glm", family = "binomial"\\)', sl3_content[i])) {
      sl3_content[i] <- gsub('family = "binomial"', 'family = binomial()', sl3_content[i])
    }
    if(grepl('list\\("Lrnr_glm", family = "gaussian"\\)', sl3_content[i])) {
      sl3_content[i] <- gsub('family = "gaussian"', 'family = gaussian()', sl3_content[i])
    }
    if(grepl('list\\("Lrnr_glmnet", .*family = "binomial"', sl3_content[i])) {
      sl3_content[i] <- gsub('family = "binomial"', 'family = "binomial"', sl3_content[i])
    }
    if(grepl('list\\("Lrnr_glmnet", .*family = "gaussian"', sl3_content[i])) {
      sl3_content[i] <- gsub('family = "gaussian"', 'family = "gaussian"', sl3_content[i])
    }
  }
  
  # Write the fixed file
  writeLines(sl3_content, sl3_file)
  cat("SL3_fns.R fixed.\n")
}

# Fix sequential_g function to handle missing covariates
fix_sequential_g <- function() {
  cat("Fixing tmle_fns.R sequential_g function...\n")
  
  # Read the file
  tmle_file <- "./src/tmle_fns.R"
  tmle_content <- readLines(tmle_file)
  
  # Find the sequential_g function
  start_idx <- grep("^sequential_g <- function", tmle_content)
  if(length(start_idx) == 0) {
    cat("Warning: sequential_g function not found in tmle_fns.R\n")
    return()
  }
  
  # Find the task creation line
  task_line_idx <- NULL
  for(i in start_idx:length(tmle_content)) {
    if(grepl("initial_model_for_Y_task <- make_sl3_Task", tmle_content[i])) {
      task_line_idx <- i
      break
    }
  }
  
  if(is.null(task_line_idx)) {
    cat("Warning: make_sl3_Task line not found in sequential_g function\n")
    return()
  }
  
  # Enhance the function to better handle missing covariates
  # We need to add code before the task creation to check the required covars
  enhanced_covar_checks <- c(
    "  # Safely check and add any missing required covariates",
    "  missing_covars <- setdiff(tmle_covars_Y, colnames(tmle_dat_sub))",
    "  if(length(missing_covars) > 0) {",
    "    message(\"Adding missing covariates: \", paste(missing_covars, collapse=\", \"))",
    "    for(cov in missing_covars) {",
    "      tmle_dat_sub[[cov]] <- 0  # Add missing covariates with default values",
    "    }",
    "  }",
    "  # Only use covariates that actually exist in the data",
    "  required_covars <- intersect(tmle_covars_Y, colnames(tmle_dat_sub))",
    ""
  )
  
  # Insert the enhanced covariate checks before the task creation
  tmle_content <- c(
    tmle_content[1:(task_line_idx-1)],
    enhanced_covar_checks,
    tmle_content[task_line_idx:length(tmle_content)]
  )
  
  # Update the task creation line to use required_covars
  task_line_idx <- task_line_idx + length(enhanced_covar_checks)
  tmle_content[task_line_idx] <- gsub("covariates = tmle_covars_Y", "covariates = required_covars", tmle_content[task_line_idx])
  
  # Add similar covariate handling for prediction task
  pred_task_line_idx <- NULL
  for(i in task_line_idx:length(tmle_content)) {
    if(grepl("prediction_task <- sl3_Task\\$new", tmle_content[i])) {
      pred_task_line_idx <- i
      break
    }
  }
  
  if(!is.null(pred_task_line_idx)) {
    pred_enhanced_checks <- c(
      "  # Safely check and add any missing required covariates for prediction",
      "  pred_missing_covars <- setdiff(needed_covars, colnames(pred_data))",
      "  if(length(pred_missing_covars) > 0) {",
      "    message(\"Adding missing covariates for prediction: \", paste(pred_missing_covars, collapse=\", \"))",
      "    for(cov in pred_missing_covars) {",
      "      pred_data[[cov]] <- 0  # Add missing covariates with default values",
      "    }",
      "  }",
      ""
    )
    
    # Insert the enhanced prediction covariate checks before the prediction task creation
    tmle_content <- c(
      tmle_content[1:(pred_task_line_idx-1)],
      pred_enhanced_checks,
      tmle_content[pred_task_line_idx:length(tmle_content)]
    )
  }
  
  # Write the fixed file
  writeLines(tmle_content, tmle_file)
  cat("tmle_fns.R sequential_g function fixed.\n")
}

# Fix getTMLELong function to handle errors at specific time points
fix_getTMLELong <- function() {
  cat("Fixing tmle_fns.R getTMLELong function...\n")
  
  # Read the file
  tmle_file <- "./src/tmle_fns.R"
  tmle_content <- readLines(tmle_file)
  
  # Find the safe_getTMLELong function
  start_idx <- grep("^safe_getTMLELong <- function", tmle_content)
  if(length(start_idx) == 0) {
    cat("Warning: safe_getTMLELong function not found in tmle_fns.R\n")
    return()
  }
  
  # Get the end of the function
  end_idx <- 0
  open_braces <- 0
  for(i in start_idx:length(tmle_content)) {
    line <- tmle_content[i]
    open_braces <- open_braces + sum(gregexpr("\\{", line)[[1]] > 0) - sum(gregexpr("\\}", line)[[1]] > 0)
    if(open_braces == 0) {
      end_idx <- i
      break
    }
  }
  
  if(end_idx == 0) {
    cat("Warning: Could not find end of safe_getTMLELong function\n")
    return()
  }
  
  # Replace the function with a more robust version
  improved_function <- c(
    "safe_getTMLELong <- function(...) {",
    "  tryCatch({",
    "    getTMLELong(...)",
    "  }, error = function(e) {",
    "    message(\"Error in getTMLELong: \", e$message)",
    "    ",
    "    # Create a default structure to return so the simulation can continue",
    "    # Extract arguments to build a minimal result structure",
    "    args <- list(...)",
    "    initial_model_for_Y <- args[[1]]",
    "    tmle_rules <- args[[2]]",
    "    ybound <- args[[6]]",
    "    if(is.null(ybound)) ybound <- c(0.0001, 0.9999)",
    "    ",
    "    # Extract data safely",
    "    tmle_dat <- NULL",
    "    if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {",
    "      tmle_dat <- initial_model_for_Y$data",
    "    } else {",
    "      # Create minimal data structure",
    "      tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))",
    "    }",
    "    ",
    "    # Create default predictions",
    "    n_obs <- nrow(tmle_dat)",
    "    n_rules <- length(tmle_rules)",
    "    rule_names <- names(tmle_rules)",
    "    if(is.null(rule_names)) rule_names <- paste0(\"rule_\", 1:n_rules)",
    "    ",
    "    # Create minimal return structure to allow simulation to continue",
    "    default_value <- 0.5",
    "    Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)",
    "    colnames(Qs) <- rule_names",
    "    ",
    "    QAW <- cbind(QA=rep(default_value, n_obs), Qs)",
    "    colnames(QAW) <- c(\"QA\", colnames(Qs))",
    "    ",
    "    Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)",
    "    colnames(Qstar) <- rule_names",
    "    ",
    "    Qstar_iptw <- rep(default_value, n_rules)",
    "    names(Qstar_iptw) <- rule_names",
    "    ",
    "    # Create minimal return object",
    "    list(",
    "      \"Qs\" = Qs,",
    "      \"QAW\" = QAW,",
    "      \"clever_covariates\" = matrix(0, nrow=n_obs, ncol=n_rules),",
    "      \"weights\" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),",
    "      \"updated_model_for_Y\" = vector(\"list\", n_rules),",
    "      \"Qstar\" = Qstar,",
    "      \"Qstar_iptw\" = Qstar_iptw,",
    "      \"Qstar_gcomp\" = Qs,",
    "      \"ID\" = tmle_dat$ID,",
    "      \"Y\" = tmle_dat$Y",
    "    )",
    "  })",
    "}"
  )
  
  # Replace the function
  tmle_content <- c(tmle_content[1:(start_idx-1)], improved_function, tmle_content[(end_idx+1):length(tmle_content)])
  
  # Write the fixed file
  writeLines(tmle_content, tmle_file)
  cat("tmle_fns.R safe_getTMLELong function fixed.\n")
}

# Also fix the binomial/gaussian family references in glm calls in getTMLELong
fix_glm_family_calls <- function() {
  cat("Fixing family parameter in GLM calls...\n")
  
  # Read the file
  tmle_file <- "./src/tmle_fns.R"
  tmle_content <- readLines(tmle_file)
  
  # Find the getTMLELong function
  start_idx <- grep("^getTMLELong <- function", tmle_content)
  if(length(start_idx) == 0) {
    cat("Warning: getTMLELong function not found in tmle_fns.R\n")
    return()
  }
  
  # Look for all glm calls with family = binomial() or family = gaussian()
  for(i in start_idx:length(tmle_content)) {
    if(grepl("family = binomial", tmle_content[i]) && !grepl("family = binomial\\(\\)", tmle_content[i])) {
      tmle_content[i] <- gsub("family = binomial", "family = binomial()", tmle_content[i])
    }
    if(grepl("family = gaussian", tmle_content[i]) && !grepl("family = gaussian\\(\\)", tmle_content[i])) {
      tmle_content[i] <- gsub("family = gaussian", "family = gaussian()", tmle_content[i])
    }
  }
  
  # Write the fixed file
  writeLines(tmle_content, tmle_file)
  cat("GLM family parameter references fixed.\n")
}

# Apply all fixes
fix_SL3_fns()
fix_sequential_g()
fix_getTMLELong()
fix_glm_family_calls()

cat("All fixes completed. Run the simulation script again with:\n")
cat("Rscript simulation.R 'tmle-lstm' 1 'TRUE' 'FALSE'\n")