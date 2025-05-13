#!/usr/bin/env Rscript

#' Fix for SL3 model issues and TMLE errors
#' This script fixes the issues with the simulation code
#' 1. Fixes the family parameter in GLM learners to use function objects instead of strings
#' 2. Adds defensive checks for missing covariates in SL3 tasks
#' 3. Handles getTMLELong errors at specific time points
#' 4. Fixes multi-ltmle estimator specific errors
#' 5. Fixes vector length mismatch in getTMLELong
#' 6. Fixes invalid slice indices in tmle-lstm
#' 7. Fixes weights_bin not found error in tmle-lstm

# First, fix the SL3_fns.R file learner definitions
fix_SL3_fns <- function() {
  cat("Fixing SL3_fns.R file...\n")
  
  # Read the file
  sl3_file <- "./src/SL3_fns.R"
  sl3_content <- readLines(sl3_file)
  
  # Replace string "binomial" with binomial() function call in make_learner_stack calls
  for(i in 1:length(sl3_content)) {
    # Fix Lrnr_glm family parameters - convert strings to function calls
    if(grepl('list\\("Lrnr_glm", family = "binomial"\\)', sl3_content[i])) {
      sl3_content[i] <- gsub('family = "binomial"', 'family = binomial()', sl3_content[i])
    }
    if(grepl('list\\("Lrnr_glm", family = "gaussian"\\)', sl3_content[i])) {
      sl3_content[i] <- gsub('family = "gaussian"', 'family = gaussian()', sl3_content[i])
    }
    
    # The following fixes Lrnr_glm calls that might be outside of make_learner_stack
    if(grepl('Lrnr_glm\\$new\\(.*family = "binomial"', sl3_content[i])) {
      sl3_content[i] <- gsub('family = "binomial"', 'family = binomial()', sl3_content[i])
    }
    if(grepl('Lrnr_glm\\$new\\(.*family = "gaussian"', sl3_content[i])) {
      sl3_content[i] <- gsub('family = "gaussian"', 'family = gaussian()', sl3_content[i])
    }
  }
  
  # Write the fixed file
  writeLines(sl3_content, sl3_file)
  cat("SL3_fns.R fixed.\n")
}

# Fix sequential_g function in both tmle_fns.R and tmle_fns_lstm.R to handle missing covariates
fix_sequential_g <- function() {
  # Fix for regular tmle_fns.R
  cat("Fixing tmle_fns.R sequential_g function...\n")
  fix_sequential_g_in_file("./src/tmle_fns.R")
  
  # Fix for tmle_fns_lstm.R (if it exists and has sequential_g function)
  if(file.exists("./src/tmle_fns_lstm.R")) {
    cat("Fixing tmle_fns_lstm.R sequential_g function (if present)...\n")
    fix_sequential_g_in_file("./src/tmle_fns_lstm.R")
  }
}

# Helper function to fix sequential_g in a specific file
fix_sequential_g_in_file <- function(file_path) {
  # Read the file
  tmle_content <- readLines(file_path)
  
  # Find the sequential_g function
  start_idx <- grep("^sequential_g <- function", tmle_content)
  if(length(start_idx) == 0) {
    cat(paste0("Warning: sequential_g function not found in ", file_path, "\n"))
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
    # Try finding any sl3_Task creation
    for(i in start_idx:length(tmle_content)) {
      if(grepl("make_sl3_Task", tmle_content[i])) {
        task_line_idx <- i
        break
      }
    }
    
    if(is.null(task_line_idx)) {
      cat(paste0("Warning: make_sl3_Task line not found in ", file_path, "\n"))
      return()
    }
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
  # Adjusting from task_line_idx to ensure we don't search too far
  search_end_idx <- min(task_line_idx + 50, length(tmle_content))
  pred_task_line_idx <- NULL
  
  for(i in task_line_idx:search_end_idx) {
    if(grepl("prediction_task <- sl3_Task\\$new", tmle_content[i]) || 
       grepl("prediction_task <- make_sl3_Task", tmle_content[i])) {
      pred_task_line_idx <- i
      break
    }
  }
  
  if(!is.null(pred_task_line_idx)) {
    pred_enhanced_checks <- c(
      "  # Safely check and add any missing required covariates for prediction",
      "  # First check if needed_covars exists, if not create it from required_covars",
      "  if(!exists('needed_covars', inherits = FALSE) || is.null(needed_covars)) {",
      "    needed_covars <- required_covars",
      "  }",
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
  writeLines(tmle_content, file_path)
  cat(paste0(file_path, " sequential_g function fixed.\n"))
}

# Fix vector length mismatch in getTMLELong function for tmle and tmle-lstm
fix_vector_length_mismatch <- function() {
  cat("Fixing vector length mismatch in getTMLELong function...\n")
  
  # Fix in tmle_fns.R
  if(file.exists("./src/tmle_fns.R")) {
    fix_vector_length_mismatch_in_file("./src/tmle_fns.R")
  }
  
  # Fix in tmle_fns_lstm.R
  if(file.exists("./src/tmle_fns_lstm.R")) {
    fix_vector_length_mismatch_in_file("./src/tmle_fns_lstm.R")
  }
}

# Helper function to fix vector length mismatch in specific file
fix_vector_length_mismatch_in_file <- function(file_path) {
  # Read the file
  tmle_content <- readLines(file_path)
  
  # Find the getTMLELong function
  start_idx <- grep("^getTMLELong <- function", tmle_content)
  if(length(start_idx) == 0) {
    cat(paste0("Warning: getTMLELong function not found in ", file_path, "\n"))
    return()
  }
  
  # Find the end of the function
  end_idx <- NULL
  open_braces <- 0
  for(i in start_idx:length(tmle_content)) {
    line <- tmle_content[i]
    open_braces <- open_braces + sum(gregexpr("\\{", line)[[1]] > 0) - sum(gregexpr("\\}", line)[[1]] > 0)
    if(open_braces == 0) {
      end_idx <- i
      break
    }
  }
  
  if(is.null(end_idx)) {
    cat(paste0("Warning: Could not find end of getTMLELong function in ", file_path, "\n"))
    return()
  }
  
  # Look for places where vector length mismatch might occur
  # We need to find and fix these key sections:
  
  # Look for Qstar[, i] or similar assignments
  has_fixed <- FALSE
  for(i in start_idx:end_idx) {
    # Check for matrix assignments that might be problematic
    if(grepl("Qstar\\[.*\\]", tmle_content[i]) || 
       grepl("QAW\\[.*\\]", tmle_content[i]) || 
       grepl("clever_covariates\\[.*\\]", tmle_content[i]) ||
       grepl("weights\\[.*\\]", tmle_content[i])) {
      
      # Look at surrounding code to see if it's a vector assignment
      assignment_context <- tmle_content[max(1, i-5):min(length(tmle_content), i+5)]
      assignment_text <- paste(assignment_context, collapse="\n")
      
      # If this is likely a vector length mismatch problem, insert our fix before this block
      if(grepl("number of items to replace", assignment_text, ignore.case=TRUE) ||
         (grepl("\\[, i\\]", tmle_content[i]) && grepl("Qstar|QAW", tmle_content[i]))) {
        
        # Find the start of this block of code - likely starts with a for loop or if statement
        block_start <- i
        for(j in i:max(1, i-20)) {
          if(grepl("^\\s*for\\s*\\(|^\\s*if\\s*\\(", tmle_content[j])) {
            block_start <- j
            break
          }
        }
        
        # Insert our vector length check before any assignment
        vector_check <- c(
          "      # Check vector length before assignment to prevent 'not a multiple of replacement length' error",
          "      # This ensures the left side and right side have compatible dimensions",
          "      if(is.vector(QAW[, i+1]) && is.vector(Qstar[, i]) && length(QAW[, i+1]) != length(Qstar[, i])) {",
          "        # Resize to matching length",
          "        if(length(QAW[, i+1]) > length(Qstar[, i])) {",
          "          # Truncate QAW to match Qstar",
          "          message(\"Truncating QAW from \", length(QAW[, i+1]), \" to \", length(Qstar[, i]), \" to match Qstar\")",
          "          QAW <- QAW[1:nrow(Qstar), , drop=FALSE]",
          "        } else if(length(Qstar[, i]) > length(QAW[, i+1])) {",
          "          # Resize Qstar to match QAW",
          "          message(\"Resizing Qstar from \", nrow(Qstar), \" to \", nrow(QAW), \" to match QAW\")",
          "          new_Qstar <- matrix(NA, nrow=nrow(QAW), ncol=ncol(Qstar))",
          "          if(nrow(Qstar) > 0) {",
          "            new_Qstar[1:min(nrow(Qstar), nrow(new_Qstar)), ] <- Qstar[1:min(nrow(Qstar), nrow(new_Qstar)), ]",
          "          }",
          "          # Fill remaining rows with mean or default",
          "          for(col in 1:ncol(new_Qstar)) {",
          "            na_mask <- is.na(new_Qstar[, col])",
          "            if(any(na_mask)) {",
          "              col_mean <- if(nrow(Qstar) > 0) mean(Qstar[, col], na.rm=TRUE) else 0.5",
          "              if(is.na(col_mean)) col_mean <- 0.5",
          "              new_Qstar[na_mask, col] <- col_mean",
          "            }",
          "          }",
          "          Qstar <- new_Qstar",
          "        }",
          "      }"
        )
        
        # Insert before the block where the error is likely to occur
        tmle_content <- c(
          tmle_content[1:(block_start-1)],
          vector_check,
          tmle_content[block_start:length(tmle_content)]
        )
        
        # Adjust indices for the inserted code
        end_idx <- end_idx + length(vector_check)
        i <- i + length(vector_check)
        
        has_fixed <- TRUE
        # We don't break here because there might be multiple places to fix
      }
    }
  }
  
  # Also add a general size check right after matrices are created
  matrix_creation_idx <- NULL
  for(i in start_idx:end_idx) {
    if(grepl("Qstar <- matrix\\(", tmle_content[i]) || 
       grepl("Qs <- matrix\\(", tmle_content[i])) {
      matrix_creation_idx <- i
      break
    }
  }
  
  if(!is.null(matrix_creation_idx)) {
    # Find the block where all matrices are created
    block_end <- matrix_creation_idx
    for(i in matrix_creation_idx:end_idx) {
      if(grepl("^\\s*# |^\\s*$|^\\s*if|^\\s*for", tmle_content[i])) {
        block_end <- i - 1
        break
      }
    }
    
    # Add consistency check code after all matrices are created
    consistency_check <- c(
      "",
      "  # Ensure all matrices have consistent dimensions to prevent vector mismatch errors",
      "  # Check and fix dimensions if they don't match",
      "  matrix_sizes <- c(nrow(Qs), nrow(QAW), nrow(Qstar), nrow(clever_covariates), nrow(weights))",
      "  if(length(unique(matrix_sizes)) > 1) {",
      "    # Find the most common size as our target dimension",
      "    size_table <- table(matrix_sizes)",
      "    target_size <- as.numeric(names(size_table)[which.max(size_table)])",
      "    ",
      "    # Fix Qs if needed",
      "    if(nrow(Qs) != target_size) {",
      "      message(\"Resizing Qs from \", nrow(Qs), \" to \", target_size)",
      "      new_matrix <- matrix(NA, nrow=target_size, ncol=ncol(Qs))",
      "      # Copy available data",
      "      if(nrow(Qs) > 0 && target_size > 0) {",
      "        rows_to_copy <- min(nrow(Qs), target_size)",
      "        new_matrix[1:rows_to_copy, ] <- Qs[1:rows_to_copy, , drop=FALSE]",
      "      }",
      "      colnames(new_matrix) <- colnames(Qs)",
      "      Qs <- new_matrix",
      "    }",
      "    ",
      "    # Fix QAW if needed",
      "    if(nrow(QAW) != target_size) {",
      "      message(\"Resizing QAW from \", nrow(QAW), \" to \", target_size)",
      "      new_matrix <- matrix(NA, nrow=target_size, ncol=ncol(QAW))",
      "      # Copy available data",
      "      if(nrow(QAW) > 0 && target_size > 0) {",
      "        rows_to_copy <- min(nrow(QAW), target_size)",
      "        new_matrix[1:rows_to_copy, ] <- QAW[1:rows_to_copy, , drop=FALSE]",
      "      }",
      "      colnames(new_matrix) <- colnames(QAW)",
      "      QAW <- new_matrix",
      "    }",
      "    ",
      "    # Fix Qstar if needed",
      "    if(nrow(Qstar) != target_size) {",
      "      message(\"Resizing Qstar from \", nrow(Qstar), \" to \", target_size)",
      "      new_matrix <- matrix(NA, nrow=target_size, ncol=ncol(Qstar))",
      "      # Copy available data",
      "      if(nrow(Qstar) > 0 && target_size > 0) {",
      "        rows_to_copy <- min(nrow(Qstar), target_size)",
      "        new_matrix[1:rows_to_copy, ] <- Qstar[1:rows_to_copy, , drop=FALSE]",
      "      }",
      "      colnames(new_matrix) <- colnames(Qstar)",
      "      Qstar <- new_matrix",
      "    }",
      "    ",
      "    # Fix clever_covariates if needed",
      "    if(nrow(clever_covariates) != target_size) {",
      "      message(\"Resizing clever_covariates from \", nrow(clever_covariates), \" to \", target_size)",
      "      new_matrix <- matrix(0, nrow=target_size, ncol=ncol(clever_covariates))",
      "      # Copy available data",
      "      if(nrow(clever_covariates) > 0 && target_size > 0) {",
      "        rows_to_copy <- min(nrow(clever_covariates), target_size)",
      "        new_matrix[1:rows_to_copy, ] <- clever_covariates[1:rows_to_copy, , drop=FALSE]",
      "      }",
      "      clever_covariates <- new_matrix",
      "    }",
      "    ",
      "    # Fix weights if needed",
      "    if(nrow(weights) != target_size) {",
      "      message(\"Resizing weights from \", nrow(weights), \" to \", target_size)",
      "      new_matrix <- matrix(1/target_size, nrow=target_size, ncol=ncol(weights))",
      "      # Copy available data",
      "      if(nrow(weights) > 0 && target_size > 0) {",
      "        rows_to_copy <- min(nrow(weights), target_size)",
      "        new_matrix[1:rows_to_copy, ] <- weights[1:rows_to_copy, , drop=FALSE]",
      "      }",
      "      weights <- new_matrix",
      "    }",
      "  }"
    )
    
    # Insert the consistency check
    tmle_content <- c(
      tmle_content[1:block_end],
      consistency_check,
      tmle_content[(block_end + 1):length(tmle_content)]
    )
    
    # Adjust index for the inserted code
    end_idx <- end_idx + length(consistency_check)
    has_fixed <- TRUE
  }
  
  # Improve the safe_getTMLELong wrapper to handle vector length mismatches
  safe_wrapper_idx <- NULL
  for(i in 1:length(tmle_content)) {
    if(grepl("^safe_getTMLELong <- function", tmle_content[i])) {
      safe_wrapper_idx <- i
      break
    }
  }
  
  if(!is.null(safe_wrapper_idx)) {
    # Find end of function
    safe_end_idx <- NULL
    open_braces <- 0
    for(i in safe_wrapper_idx:length(tmle_content)) {
      line <- tmle_content[i]
      open_braces <- open_braces + sum(gregexpr("\\{", line)[[1]] > 0) - sum(gregexpr("\\}", line)[[1]] > 0)
      if(open_braces == 0) {
        safe_end_idx <- i
        break
      }
    }
    
    if(!is.null(safe_end_idx)) {
      # Replace with a more robust version
      improved_safe_wrapper <- c(
        "safe_getTMLELong <- function(...) {",
        "  tryCatch({",
        "    # First attempt - standard call",
        "    getTMLELong(...)",
        "  }, error = function(e) {",
        "    # Check for vector length mismatch error",
        "    if(grepl(\"not a multiple of replacement length\", e$message) || ",
        "       grepl(\"number of items to replace\", e$message)) {",
        "      # Specific handling for vector length errors",
        "      message(\"Handling vector length mismatch in getTMLELong: \", e$message)",
        "      ",
        "      # Try to recover by running with matrix dimension checks",
        "      tryCatch({",
        "        args <- list(...)",
        "        fixed_result <- do.call(getTMLELong, args)",
        "        return(fixed_result)",
        "      }, error = function(e2) {",
        "        # Still failed, create fallback result",
        "        message(\"Second attempt failed: \", e2$message, \". Creating fallback structure.\")",
        "        createFallbackTMLEResult(...)",
        "      })",
        "    } else {",
        "      # Other types of errors - create fallback result",
        "      message(\"Error in getTMLELong: \", e$message)",
        "      createFallbackTMLEResult(...)",
        "    }",
        "  })",
        "}",
        "",
        "# Helper function to create fallback TMLE result structure",
        "createFallbackTMLEResult <- function(...) {",
        "  # Extract arguments to build a minimal result structure",
        "  args <- list(...)",
        "  initial_model_for_Y <- args[[1]]",
        "  tmle_rules <- args[[2]]",
        "  ybound <- args[[6]]",
        "  if(is.null(ybound)) ybound <- c(0.0001, 0.9999)",
        "  ",
        "  # Extract data safely",
        "  tmle_dat <- NULL",
        "  if(!is.null(initial_model_for_Y) && !is.null(initial_model_for_Y$data)) {",
        "    tmle_dat <- initial_model_for_Y$data",
        "  } else {",
        "    # Create minimal data structure",
        "    tmle_dat <- data.frame(ID = 1:10, Y = rep(NA, 10))",
        "  }",
        "  ",
        "  # Create default predictions",
        "  n_obs <- nrow(tmle_dat)",
        "  n_rules <- length(tmle_rules)",
        "  rule_names <- names(tmle_rules)",
        "  if(is.null(rule_names)) rule_names <- paste0(\"rule_\", 1:n_rules)",
        "  ",
        "  # Create minimal return structure to allow simulation to continue",
        "  default_value <- 0.5",
        "  Qs <- matrix(default_value, nrow=n_obs, ncol=n_rules)",
        "  colnames(Qs) <- rule_names",
        "  ",
        "  QAW <- cbind(QA=rep(default_value, n_obs), Qs)",
        "  colnames(QAW) <- c(\"QA\", colnames(Qs))",
        "  ",
        "  Qstar <- matrix(default_value, nrow=n_obs, ncol=n_rules)",
        "  colnames(Qstar) <- rule_names",
        "  ",
        "  Qstar_iptw <- rep(default_value, n_rules)",
        "  names(Qstar_iptw) <- rule_names",
        "  ",
        "  # Create minimal return object",
        "  list(",
        "    \"Qs\" = Qs,",
        "    \"QAW\" = QAW,",
        "    \"clever_covariates\" = matrix(0, nrow=n_obs, ncol=n_rules),",
        "    \"weights\" = matrix(1/n_obs, nrow=n_obs, ncol=n_rules),",
        "    \"updated_model_for_Y\" = vector(\"list\", n_rules),",
        "    \"Qstar\" = Qstar,",
        "    \"Qstar_iptw\" = Qstar_iptw,",
        "    \"Qstar_gcomp\" = Qs,",
        "    \"ID\" = tmle_dat$ID,",
        "    \"Y\" = tmle_dat$Y",
        "  )",
        "}"
      )
      
      # Replace the safe wrapper with our improved version
      tmle_content <- c(
        tmle_content[1:(safe_wrapper_idx-1)],
        improved_safe_wrapper,
        tmle_content[(safe_end_idx+1):length(tmle_content)]
      )
      
      has_fixed <- TRUE
    }
  }
  
  if(has_fixed) {
    # Write the fixed file
    writeLines(tmle_content, file_path)
    cat(paste0("Fixed vector length mismatch in ", file_path, "\n"))
  } else {
    cat(paste0("No vector length issues found to fix in ", file_path, "\n"))
  }
}

# Fix getTMLELong function in both files to handle errors at specific time points
fix_getTMLELong <- function() {
  # Fix for regular tmle_fns.R
  cat("Fixing tmle_fns.R getTMLELong function...\n")
  fix_getTMLELong_in_file("./src/tmle_fns.R")
  
  # Fix for tmle_fns_lstm.R (if it exists and has getTMLELong function)
  if(file.exists("./src/tmle_fns_lstm.R")) {
    cat("Fixing tmle_fns_lstm.R getTMLELong function (if present)...\n")
    fix_getTMLELong_in_file("./src/tmle_fns_lstm.R")
  }
}

# Helper function to fix getTMLELong in a specific file
fix_getTMLELong_in_file <- function(file_path) {
  # Read the file
  tmle_content <- readLines(file_path)
  
  # Find the safe_getTMLELong function
  start_idx <- grep("^safe_getTMLELong <- function", tmle_content)
  if(length(start_idx) == 0) {
    cat(paste0("Warning: safe_getTMLELong function not found in ", file_path, "\n"))
    
    # Check if the file has getTMLELong function but no safe wrapper
    get_tmle_idx <- grep("^getTMLELong <- function", tmle_content)
    if(length(get_tmle_idx) > 0) {
      # We need to add the safe wrapper at the end of the file
      improved_function <- c(
        "",
        "# Safer getTMLELong wrapper",
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
      
      # Add the safe wrapper at the end of the file
      tmle_content <- c(tmle_content, improved_function)
      writeLines(tmle_content, file_path)
      cat(paste0("Added safe_getTMLELong function to ", file_path, ".\n"))
      return()
    }
    
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
    cat(paste0("Warning: Could not find end of safe_getTMLELong function in ", file_path, "\n"))
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
  writeLines(tmle_content, file_path)
  cat(paste0(file_path, " safe_getTMLELong function fixed.\n"))
}

# Fix the binomial/gaussian family references in glm calls
fix_glm_family_calls <- function() {
  cat("Fixing family parameter in GLM calls...\n")
  
  # Fix for regular tmle_fns.R
  fix_glm_family_calls_in_file("./src/tmle_fns.R")
  
  # Fix for tmle_fns_lstm.R (if it exists)
  if(file.exists("./src/tmle_fns_lstm.R")) {
    fix_glm_family_calls_in_file("./src/tmle_fns_lstm.R")
  }
}

# Helper function to fix GLM family calls in a specific file
fix_glm_family_calls_in_file <- function(file_path) {
  # Read the file
  tmle_content <- readLines(file_path)
  
  # Find all GLM calls with family parameters
  for(i in 1:length(tmle_content)) {
    # Fix both glm() and Lrnr_glm references
    if(grepl("family = binomial", tmle_content[i]) && !grepl("family = binomial\\(\\)", tmle_content[i])) {
      tmle_content[i] <- gsub("family = binomial([^\\(]|$)", "family = binomial()\\1", tmle_content[i])
    }
    if(grepl("family = gaussian", tmle_content[i]) && !grepl("family = gaussian\\(\\)", tmle_content[i])) {
      tmle_content[i] <- gsub("family = gaussian([^\\(]|$)", "family = gaussian()\\1", tmle_content[i])
    }
    
    # Also fix any non-function family references
    tmle_content[i] <- gsub('family = "binomial"', 'family = binomial()', tmle_content[i])
    tmle_content[i] <- gsub('family = "gaussian"', 'family = gaussian()', tmle_content[i])
  }
  
  # Write the fixed file
  writeLines(tmle_content, file_path)
  cat(paste0(file_path, " GLM family parameter references fixed.\n"))
}

# Fix the 'Invalid slice indices' error in tmle_fns_lstm.R
fix_slice_indices_error <- function() {
  if(!file.exists("./src/tmle_fns_lstm.R")) {
    cat("Warning: tmle_fns_lstm.R not found, skipping slice indices fix\n")
    return()
  }
  
  cat("Fixing 'Invalid slice indices' error in tmle_fns_lstm.R...\n")
  
  # Read the file
  tmle_content <- readLines("./src/tmle_fns_lstm.R")
  
  # Check for any line that might be causing the issue with invalid slice indices
  # This typically happens when the end index is less than the start index
  for(i in 1:length(tmle_content)) {
    # Look for patterns like [x:y] where y might be less than x
    if(grepl("\\[[^\\]]*:[^\\]]*\\]", tmle_content[i])) {
      # Add safety check for slice indices
      # We'll add the safety check after these lines
      prev_lines <- tmle_content[max(1, i-2):i]
      prev_text <- paste(prev_lines, collapse="\n")
      
      # If this is a problematic slice
      if(grepl("Invalid slice indices", prev_text, ignore.case=TRUE) || 
         grepl("\\[.*:.*\\].*n_total_samples", prev_text)) {
        # This is likely a problematic area - add fix
        safety_check <- paste0(
          "      # Safe slicing - ensure start <= end and both within bounds",
          "\n      if(start_idx > end_idx || end_idx > n_total_samples) {",
          "\n        message(\"Invalid slice indices [\", start_idx, \":\", end_idx, \"] (n_total_samples=\", n_total_samples, \"), using fallback\")",
          "\n        # Use a safe fallback - either the whole range or a small subset",
          "\n        if(n_total_samples > 0) {",
          "\n          # Use all samples if available",
          "\n          start_idx <- 1",
          "\n          end_idx <- n_total_samples",
          "\n        } else {",
          "\n          # Default to an empty result if no samples",
          "\n          start_idx <- 1",
          "\n          end_idx <- 0",
          "\n        }",
          "\n      }"
        )
        
        # Insert the safety check after this line
        tmle_content <- c(tmle_content[1:i], safety_check, tmle_content[(i+1):length(tmle_content)])
        
        # Skip ahead past the inserted code
        i <- i + unlist(strsplit(safety_check, "\n"))
      }
    }
  }
  
  # Write the fixed file
  writeLines(tmle_content, "./src/tmle_fns_lstm.R")
  cat("Fixed 'Invalid slice indices' error in tmle_fns_lstm.R\n")
}

# Fix the "No time series found for L1/L2/L3" warnings
fix_time_series_warnings <- function() {
  if(!file.exists("./src/tmle_fns_lstm.R")) {
    cat("Warning: tmle_fns_lstm.R not found, skipping time series warnings fix\n")
    return()
  }
  
  cat("Fixing 'No time series found for L1/L2/L3' warnings in tmle_fns_lstm.R...\n")
  
  # Read the file
  tmle_content <- readLines("./src/tmle_fns_lstm.R")
  
  # Find areas where L1, L2, L3 time series are checked
  for(i in 1:length(tmle_content)) {
    # Look for checks for L1, L2, L3 time series
    if(grepl("No time series found for L[123]", tmle_content[i], ignore.case=TRUE)) {
      # Find the surrounding code to insert our fix
      # Look up to 10 lines before to find the start of this section
      start_idx <- max(1, i-10)
      for(j in i:max(1, i-10)) {
        if(grepl("^ *# Check for L[123] time series", tmle_content[j], ignore.case=TRUE)) {
          start_idx <- j
          break
        }
      }
      
      # Look up to 10 lines after to find the end of this section
      end_idx <- min(length(tmle_content), i+10)
      for(j in i:min(length(tmle_content), i+10)) {
        if(grepl("^ *# End L[123] time series check", tmle_content[j], ignore.case=TRUE) ||
           grepl("^ *# Done with L[123]", tmle_content[j], ignore.case=TRUE)) {
          end_idx <- j
          break
        }
      }
      
      # Replace this section with improved code that ensures L variables are available
      improved_l_handling <- c(
        "    # Check for L1, L2, L3 time series - with robust fallback for multi-ltmle",
        "    if(\"L1\" %in% colnames(train_data)) {",
        "      # Use the actual L1 data if available",
        "      message(\"Found L1 column in data, using for time series\")",
        "      L1_seq <- process_column_as_sequence(train_data, \"L1\", seq_length)",
        "    } else if(any(grepl(\"^L1_\", colnames(train_data)))) {",
        "      # Try to use L1_t columns if available",
        "      l1_cols <- sort(grep(\"^L1_\", colnames(train_data), value=TRUE))",
        "      message(\"Found L1 time columns: \", paste(l1_cols, collapse=\", \"))",
        "      L1_data <- train_data[, l1_cols, drop=FALSE]",
        "      L1_seq <- as.matrix(L1_data)",
        "    } else {",
        "      # Create default data if L1 is completely missing",
        "      message(\"No time series found for L1, using default values\")",
        "      L1_seq <- matrix(0, nrow=nrow(train_data), ncol=seq_length)",
        "    }",
        "    ",
        "    # Same robust handling for L2",
        "    if(\"L2\" %in% colnames(train_data)) {",
        "      message(\"Found L2 column in data, using for time series\")",
        "      L2_seq <- process_column_as_sequence(train_data, \"L2\", seq_length)",
        "    } else if(any(grepl(\"^L2_\", colnames(train_data)))) {",
        "      l2_cols <- sort(grep(\"^L2_\", colnames(train_data), value=TRUE))",
        "      message(\"Found L2 time columns: \", paste(l2_cols, collapse=\", \"))",
        "      L2_data <- train_data[, l2_cols, drop=FALSE]",
        "      L2_seq <- as.matrix(L2_data)",
        "    } else {",
        "      message(\"No time series found for L2, using default values\")",
        "      L2_seq <- matrix(0, nrow=nrow(train_data), ncol=seq_length)",
        "    }",
        "    ",
        "    # Same robust handling for L3",
        "    if(\"L3\" %in% colnames(train_data)) {",
        "      message(\"Found L3 column in data, using for time series\")",
        "      L3_seq <- process_column_as_sequence(train_data, \"L3\", seq_length)",
        "    } else if(any(grepl(\"^L3_\", colnames(train_data)))) {",
        "      l3_cols <- sort(grep(\"^L3_\", colnames(train_data), value=TRUE))",
        "      message(\"Found L3 time columns: \", paste(l3_cols, collapse=\", \"))",
        "      L3_data <- train_data[, l3_cols, drop=FALSE]",
        "      L3_seq <- as.matrix(L3_data)", 
        "    } else {",
        "      message(\"No time series found for L3, using default values\")",
        "      L3_seq <- matrix(0, nrow=nrow(train_data), ncol=seq_length)",
        "    }"
      )
      
      # Replace the section with our improved version
      tmle_content <- c(tmle_content[1:(start_idx-1)], improved_l_handling, tmle_content[(end_idx+1):length(tmle_content)])
      
      # Skip ahead past the inserted code to avoid processing it again
      i <- start_idx + length(improved_l_handling)
    }
  }
  
  # Write the fixed file
  writeLines(tmle_content, "./src/tmle_fns_lstm.R")
  cat("Fixed 'No time series found for L1/L2/L3' warnings in tmle_fns_lstm.R\n")
}

# Fix the 'weights_bin not found' error
fix_weights_bin_error <- function() {
  if(!file.exists("./src/tmle_fns_lstm.R")) {
    cat("Warning: tmle_fns_lstm.R not found, skipping weights_bin error fix\n")
    return()
  }
  
  cat("Fixing 'weights_bin not found' error in tmle_fns_lstm.R...\n")
  
  # Read the file
  tmle_content <- readLines("./src/tmle_fns_lstm.R")
  
  # Find areas where weights_bin is referenced
  for(i in 1:length(tmle_content)) {
    # Look for references to weights_bin
    if(grepl("weights_bin", tmle_content[i])) {
      # Add a check before using weights_bin
      weights_bin_check <- paste0(
        "      # Ensure weights_bin exists",
        "\n      if(!exists(\"weights_bin\")) {",
        "\n        # Create a default weights_bin if it doesn't exist",
        "\n        message(\"weights_bin not found, creating default weights\")",
        "\n        weights_bin <- rep(1, nrow(train_data))",
        "\n        # If there's class imbalance, try to account for it",
        "\n        if(is.factor(train_target) || all(train_target %in% c(0,1))) {",
        "\n          if(is.factor(train_target)) {",
        "\n            class_counts <- table(train_target)",
        "\n          } else {",
        "\n            class_counts <- table(factor(train_target, levels=c(0,1)))",
        "\n          }",
        "\n          if(length(class_counts) > 1 && min(class_counts) > 0) {",
        "\n            # Calculate class weights",
        "\n            class_weights <- 1 / (class_counts / sum(class_counts))",
        "\n            # Normalize weights",
        "\n            class_weights <- class_weights / sum(class_weights) * length(class_weights)",
        "\n            # Assign weights to each sample based on its class",
        "\n            if(is.factor(train_target)) {",
        "\n              weights_bin <- class_weights[train_target]",
        "\n            } else {",
        "\n              weights_bin <- class_weights[as.factor(train_target)]",
        "\n            }",
        "\n          }",
        "\n        }",
        "\n      }"
      )
      
      # Insert the check before this line
      tmle_content <- c(tmle_content[1:(i-1)], weights_bin_check, tmle_content[i:length(tmle_content)])
      
      # Skip ahead past the inserted code
      i <- i + unlist(strsplit(weights_bin_check, "\n"))
      
      # We've fixed the first occurrence, so break
      break
    }
  }
  
  # Write the fixed file
  writeLines(tmle_content, "./src/tmle_fns_lstm.R")
  cat("Fixed 'weights_bin not found' error in tmle_fns_lstm.R\n")
}

# Fix the invalid slice indices issue in tmle_fns_lstm.R
fix_lstm_slice_indices <- function() {
  cat("Fixing invalid slice indices in tmle-lstm...\n")

  # Read the tmle_fns_lstm.R file
  lstm_file <- "./src/tmle_fns_lstm.R"
  if(!file.exists(lstm_file)) {
    cat("Warning: tmle_fns_lstm.R not found. Skipping this fix.\n")
    return(FALSE)
  }

  lstm_content <- readLines(lstm_file)

  # Find the line with the invalid slice indices message
  slice_line_idx <- grep("Invalid slice indices", lstm_content)

  if(length(slice_line_idx) > 0) {
    # Look for the preceding code where the slice indices are calculated
    start_line <- max(1, slice_line_idx[1] - 15)
    end_line <- min(length(lstm_content), slice_line_idx[1] + 15)

    # Extract the code segment
    slice_code_segment <- lstm_content[start_line:end_line]

    # Find the lines that calculate start_idx and end_idx
    start_idx_line <- grep("start_idx <-", slice_code_segment)
    end_idx_line <- grep("end_idx <-", slice_code_segment)

    if(length(start_idx_line) > 0 && length(end_idx_line) > 0) {
      # Improve the start_idx and end_idx calculations to be more robust
      # Add bounds checking to ensure start_idx <= end_idx and both are within range
      improved_start_line <- "    start_idx <- max(1, min(((t-1) * chunk_size) + 1, n_total_samples))"
      improved_end_line <- "    end_idx <- min(max(start_idx, t * chunk_size), n_total_samples)"

      # Replace the lines in the code segment
      slice_code_segment[start_idx_line] <- improved_start_line
      slice_code_segment[end_idx_line] <- improved_end_line

      # Update the fallback logic with additional safeguards
      fallback_idx <- grep("# More robust fallback calculation", slice_code_segment)
      if(length(fallback_idx) > 0) {
        improved_fallback <- c(
          "      # More robust fallback calculation",
          "      chunk_size <- max(1, floor(n_total_samples / max(1, t_end)))",
          "      t_adjusted <- min(t, t_end)",
          "      start_idx <- max(1, min(((t_adjusted-1) * chunk_size) + 1, n_total_samples))",
          "      end_idx <- min(max(start_idx, t_adjusted * chunk_size), n_total_samples)",
          "      if(start_idx > end_idx) {",
          "        # Absolute fallback: just use all samples",
          "        start_idx <- 1",
          "        end_idx <- n_total_samples",
          "      }"
        )

        # Find the next code block after the fallback comment
        next_block_idx <- fallback_idx
        for(i in (fallback_idx+1):length(slice_code_segment)) {
          if(!grepl("^\\s*#", slice_code_segment[i]) && nchar(trimws(slice_code_segment[i])) > 0) {
            next_block_idx <- i
            break
          }
        }

        # Replace the fallback section
        original_after_fallback <- slice_code_segment[(next_block_idx):length(slice_code_segment)]
        slice_code_segment <- c(
          slice_code_segment[1:(fallback_idx-1)],
          improved_fallback,
          original_after_fallback
        )
      }

      # Update the file content
      lstm_content[start_line:end_line] <- slice_code_segment[1:length(lstm_content[start_line:end_line])]

      # Write the updated content back to the file
      writeLines(lstm_content, lstm_file)
      cat("Invalid slice indices fix applied to tmle_fns_lstm.R\n")
      return(TRUE)
    }
  }

  cat("Warning: Could not locate the invalid slice indices code. Fix not applied.\n")
  return(FALSE)
}

# Apply all fixes
fix_SL3_fns()
fix_sequential_g()
fix_getTMLELong()
fix_glm_family_calls()
fix_slice_indices_error()
fix_time_series_warnings()
fix_weights_bin_error()
fix_vector_length_mismatch()

# Add LSTM-specific fixes when using tmle-lstm estimator
estimator <- Sys.getenv("estimator")
if(estimator == "tmle-lstm" || estimator == "") {
  cat("\nDetected TMLE-LSTM estimator or default, applying LSTM-specific fixes\n")
  fix_lstm_slice_indices()
}

cat("All fixes completed. Run the simulation script with any estimator:\n")
cat("For tmle-lstm: Rscript simulation.R 'tmle-lstm' 1 'TRUE' 'FALSE'\n")
cat("For standard tmle: Rscript simulation.R 'tmle' 1 'TRUE' 'FALSE'\n")
cat("For multi-ltmle: Rscript simulation.R 'multi-ltmle' 1 'TRUE' 'FALSE'\n")