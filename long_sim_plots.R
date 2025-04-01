############################################################################################
# Combine plots from longitudinal simulations                                             #
############################################################################################

library(tidyverse)
library(ggplot2)
library(data.table)
library(latex2exp)
library(dplyr)
library(grid)
library(gtable)
library(xtable)
library(progress) # For progress bar

# Define parameters
J <- 6
n <- 10000
R <- 325
t.end <- 36

treatment.rules <- c("static", "dynamic", "stochastic")

# Process command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cat("No directories specified. Please provide at least one directory path.\n")
  cat("Usage: Rscript long_sim_plots.R 'outputs/dir1' 'outputs/dir2' ...\n")
  stop("No directories specified.")
} else {
  # Define directories from command line arguments
  base_dirs <- args
  cat("Processing directories:", paste(base_dirs, collapse = ", "), "\n")
}

# Function to find all results files
find_results_files <- function(base_dir, estimator_type) {
  pattern <- paste0("longitudinal_simulation_results_estimator_", estimator_type, 
                    "_treatment_rule_all.*\\.rds$")
  files <- list.files(path = base_dir, pattern = pattern, recursive = TRUE, full.names = TRUE)
  return(files)
}

# Function to extract iteration number from filename
get_iteration <- function(filename) {
  r_match <- regexpr("r_\\d+", basename(filename))
  if(r_match != -1) {
    r_str <- substr(basename(filename), r_match, r_match + attr(r_match, "match.length") - 1)
    return(as.numeric(gsub("r_", "", r_str)))
  }
  return(NA)
}

# Function to safely extract nested list elements and convert to matrix
safe_extract_nested <- function(result, metric_name) {
  if(is.null(result[[metric_name]])) return(NULL)
  
  # Get the metric data
  metric_data <- result[[metric_name]]
  if(length(metric_data) == 0) return(NULL)
  
  # If it's already a matrix, return as is
  if(is.matrix(metric_data)) return(metric_data)
  
  # If it's a list of vectors, convert to matrix
  if(is.list(metric_data)) {
    # Try to convert list to matrix
    tryCatch({
      metric_matrix <- do.call(rbind, metric_data)
      return(metric_matrix)
    }, error = function(e) {
      # If rbind fails, try another approach
      cat("Error converting list to matrix:", e$message, "\n")
      # Try to extract vectors from list elements
      metric_vectors <- lapply(metric_data, function(x) {
        if(is.vector(x)) return(x)
        if(is.matrix(x)) return(x[1,]) # Take first row if matrix
        return(NULL)
      })
      metric_vectors <- metric_vectors[!sapply(metric_vectors, is.null)]
      if(length(metric_vectors) > 0) {
        metric_matrix <- do.call(rbind, metric_vectors)
        return(metric_matrix)
      }
      return(NULL)
    })
  }
  
  # If it's a vector, convert to matrix
  if(is.vector(metric_data)) {
    return(matrix(metric_data, nrow=1))
  }
  
  # Default return NULL if we can't process
  return(NULL)
}

# Function to process a single results file
process_results_file <- function(file_path) {
  # Load the result file
  result <- readRDS(file_path)
  
  # Extract results field if it exists
  if(!is.null(result$results)) {
    result_data <- result$results
  } else {
    result_data <- result
  }
  
  # Extract metrics
  metrics <- list(
    bias_tmle = safe_extract_nested(result_data, "bias_tmle"),
    bias_tmle_bin = safe_extract_nested(result_data, "bias_tmle_bin"),
    CP_tmle = safe_extract_nested(result_data, "CP_tmle"),
    CP_tmle_bin = safe_extract_nested(result_data, "CP_tmle_bin"),
    CIW_tmle = safe_extract_nested(result_data, "CIW_tmle"),
    CIW_tmle_bin = safe_extract_nested(result_data, "CIW_tmle_bin"),
    bias_iptw = safe_extract_nested(result_data, "bias_iptw"),
    bias_iptw_bin = safe_extract_nested(result_data, "bias_iptw_bin"),
    CP_iptw = safe_extract_nested(result_data, "CP_iptw"),
    CP_iptw_bin = safe_extract_nested(result_data, "CP_iptw_bin"),
    CIW_iptw = safe_extract_nested(result_data, "CIW_iptw"),
    CIW_iptw_bin = safe_extract_nested(result_data, "CIW_iptw_bin"),
    bias_gcomp = safe_extract_nested(result_data, "bias_gcomp"),
    CP_gcomp = safe_extract_nested(result_data, "CP_gcomp"),
    CIW_gcomp = safe_extract_nested(result_data, "CIW_gcomp"),
    prob_share_tmle = safe_extract_nested(result_data, "prob_share_tmle"),
    prob_share_tmle_bin = safe_extract_nested(result_data, "prob_share_tmle_bin")
  )
  
  # Filter out NULL values
  metrics <- metrics[!sapply(metrics, is.null)]
  
  return(metrics)
}

# Function to combine all results with progress bar
combine_results <- function(base_dirs, estimator_type) {
  # Initialize results storage
  all_results <- list()
  processed_iterations <- numeric()
  
  # Get a list of all files to be processed
  all_files <- list()
  for(dir in base_dirs) {
    files <- find_results_files(dir, estimator_type)
    if(length(files) > 0) {
      all_files[[dir]] <- files
    }
  }
  
  # Flatten the list of files
  flat_files <- unlist(all_files, use.names = FALSE)
  if(length(flat_files) == 0) {
    cat("No files found for estimator type:", estimator_type, "\n")
    return(list())
  }
  
  # Set up progress bar
  cat("\nProcessing", length(flat_files), "files for estimator type:", estimator_type, "\n")
  pb <- progress_bar$new(
    format = "  Processing [:bar] :percent eta: :eta",
    total = length(flat_files),
    clear = FALSE,
    width = 60
  )
  
  # Process each file with progress bar
  for(file in flat_files) {
    # Update progress bar
    pb$tick()
    
    # Extract iteration
    iteration <- get_iteration(file)
    if(is.na(iteration)) {
      next
    }
    
    # Process all iterations
    tryCatch({
      results <- process_results_file(file)
      if(length(results) > 0) {
        all_results[[as.character(iteration)]] <- results
        processed_iterations <- c(processed_iterations, iteration)
      }
    }, error = function(e) {
      cat("\nError processing file:", file, "\n")
      cat("Error message:", e$message, "\n")
    })
  }
  
  cat("\nTotal iterations processed for", estimator_type, ":", length(processed_iterations), "\n")
  if(length(processed_iterations) > 0) {
    cat("Iterations processed:", paste(sort(processed_iterations), collapse = ", "), "\n")
  }
  
  # Combine results into final format
  final_results <- list()
  
  if(length(all_results) > 0) {
    metric_names <- unique(unlist(lapply(all_results, names)))
    
    for(metric in metric_names) {
      # Get all results that have this metric
      valid_results <- Filter(function(x) !is.null(x[[metric]]), all_results)
      if(length(valid_results) > 0) {
        metric_values <- lapply(valid_results, function(x) x[[metric]])
        
        # Combine the matrices - check dimensions first
        if(all(sapply(metric_values, is.matrix))) {
          # Check if all matrices have the same dimensions
          dims <- unique(lapply(metric_values, dim))
          if(length(dims) == 1) {
            # All matrices have the same dimensions, safe to rbind
            final_results[[metric]] <- do.call(rbind, metric_values)
          } else {
            # Different dimensions, need to standardize
            cat("Warning: Different dimensions found for metric", metric, "\n")
            # Find the most common dimension
            dim_counts <- table(sapply(metric_values, function(x) paste(dim(x), collapse="x")))
            most_common_dim <- as.numeric(strsplit(names(which.max(dim_counts)), "x")[[1]])
            
            # Filter matrices with the most common dimension
            standard_matrices <- metric_values[sapply(metric_values, function(x) all(dim(x) == most_common_dim))]
            if(length(standard_matrices) > 0) {
              final_results[[metric]] <- do.call(rbind, standard_matrices)
            }
          }
        } else {
          # Not all elements are matrices, try to convert
          cat("Warning: Non-matrix elements found for metric", metric, "\n")
          # Try to convert all to matrices with same dimensions
          standardized <- lapply(metric_values, function(x) {
            if(is.matrix(x)) return(x)
            if(is.vector(x)) return(matrix(x, nrow=1))
            return(NULL)
          })
          standardized <- standardized[!sapply(standardized, is.null)]
          if(length(standardized) > 0) {
            # Check dimensions again
            dims <- unique(lapply(standardized, dim))
            if(length(dims) == 1) {
              final_results[[metric]] <- do.call(rbind, standardized)
            }
          }
        }
      }
    }
  }
  
  return(final_results)
}

# Function to reshape matrix from wide to long format
reshape_matrix <- function(mat) {
  # Check if input is NULL or empty
  if(is.null(mat) || length(mat) == 0) return(NULL)
  
  # Handle non-matrix inputs
  if(!is.matrix(mat)) {
    if(is.vector(mat)) {
      return(mat)  # Already a vector
    } else if(is.list(mat)) {
      # Try to convert list to vector
      tryCatch({
        vec <- unlist(mat)
        return(vec)
      }, error = function(e) {
        cat("Error converting list to vector:", e$message, "\n")
        return(NULL)
      })
    } else {
      # Unknown type
      cat("Warning: Unknown type in reshape_matrix\n")
      return(NULL)
    }
  }
  
  # Reshape the matrix to vector by columns
  as.vector(mat)
}

# Function to create results dataframe from combined results
create_results_df <- function(results, estimator_names) {
  # Check if we have any of the required metrics
  relevant_metrics <- c("bias_tmle", "bias_tmle_bin", "bias_gcomp", "bias_iptw", "bias_iptw_bin",
                        "CP_tmle", "CP_tmle_bin", "CP_gcomp", "CP_iptw", "CP_iptw_bin",
                        "CIW_tmle", "CIW_tmle_bin", "CIW_gcomp", "CIW_iptw", "CIW_iptw_bin")
  
  # Filter for metrics we actually have
  available_metrics <- intersect(names(results), relevant_metrics)
  
  if(length(available_metrics) == 0) {
    cat("Warning: No relevant metrics found in results\n")
    return(NULL)
  }
  
  # Determine which estimator patterns we have
  has_tmle <- any(grepl("tmle", available_metrics))
  has_gcomp <- any(grepl("gcomp", available_metrics))
  has_iptw <- any(grepl("iptw", available_metrics))
  
  # Determine which metrics to use for each column
  abs_bias_metrics <- Filter(function(x) grepl("bias_", x), available_metrics)
  coverage_metrics <- Filter(function(x) grepl("CP_", x), available_metrics)
  ciw_metrics <- Filter(function(x) grepl("CIW_", x), available_metrics)
  
  # Extract data - initialize with empty vectors
  abs.bias <- c()
  coverage <- c() 
  ciw <- c()
  
  # Add data from each available metric
  for(metric in abs_bias_metrics) {
    vals <- reshape_matrix(abs(results[[metric]]))
    if(!is.null(vals)) abs.bias <- c(abs.bias, vals)
  }
  
  for(metric in coverage_metrics) {
    vals <- reshape_matrix(results[[metric]])
    if(!is.null(vals)) coverage <- c(coverage, vals)
  }
  
  for(metric in ciw_metrics) {
    vals <- reshape_matrix(results[[metric]])
    if(!is.null(vals)) ciw <- c(ciw, vals)
  }
  
  # Get dimensions from first available metric
  first_metric <- available_metrics[1]
  reference_matrix <- results[[first_metric]]
  
  if(is.null(reference_matrix)) {
    cat("Error: Reference matrix is NULL, cannot determine dimensions\n")
    return(NULL)
  }
  
  # Assume matrix has columns for different time points and rows for different treatments
  # Each matrix has t.end-1 columns (t=2:t.end) and 3 rows (treatment.rules)
  n_timesteps <- ncol(reference_matrix)
  n_treatments <- 3  # Static, dynamic, stochastic
  
  # Filter out any non-finite values
  abs.bias <- abs.bias[is.finite(abs.bias)]
  coverage <- coverage[is.finite(coverage)]
  ciw <- ciw[is.finite(ciw)]
  
  # Determine the actual number of estimators we have data for
  n_estimators <- length(estimator_names)
  
  # Check length consistency and adjust if needed
  expected_length <- n_timesteps * n_treatments * n_estimators
  
  # If we have less data than expected, use available data
  if(length(abs.bias) < expected_length) {
    cat("Warning: Less data than expected. Adjusting dimensions.\n")
    # Try to calculate a reasonable timestep count
    n_timesteps_calc <- floor(length(abs.bias) / (n_treatments * n_estimators))
    if(n_timesteps_calc > 0) {
      n_timesteps <- n_timesteps_calc
      cat("Adjusted n_timesteps to", n_timesteps, "\n")
    }
  }
  
  # Create dataframe with all available data
  data_length <- min(length(abs.bias), length(coverage), length(ciw))
  if(data_length == 0) {
    cat("Error: No valid data for results dataframe\n")
    return(NULL)
  }
  
  # Truncate to valid length
  abs.bias <- abs.bias[1:data_length]
  coverage <- coverage[1:data_length]
  ciw <- ciw[1:data_length]
  
  # Create repetition patterns for estimator, rule, and time
  # Make sure we don't exceed the data length
  est_pattern <- rep(estimator_names, each = n_timesteps * n_treatments)
  est_pattern <- est_pattern[1:data_length]
  
  rule_pattern <- rep(rep(treatment.rules, each = n_timesteps), n_estimators)
  rule_pattern <- rule_pattern[1:data_length]
  
  time_pattern <- rep(2:t.end, n_estimators * n_treatments)
  if(length(time_pattern) > data_length) {
    time_pattern <- time_pattern[1:data_length]
  } else if(length(time_pattern) < data_length) {
    # Repeat the pattern if we need more values
    repeat_times <- ceiling(data_length / length(time_pattern))
    time_pattern <- rep(time_pattern, repeat_times)[1:data_length]
  }
  
  # Create dataframe with estimator names and correct dimensions
  df <- data.frame(
    "abs.bias" = abs.bias,
    "coverage" = coverage,
    "CIW" = ciw,
    "Estimator" = est_pattern,
    "Rule" = rule_pattern,
    "t" = time_pattern
  )
  
  return(df)
}

# Function to create LaTeX table for estimator metrics
create_estimator_table <- function(results_df) {
  # Calculate summary statistics by estimator
  est_summary <- results_df %>%
    group_by(Estimator) %>%
    summarize(
      Coverage_Mean = mean(coverage, na.rm=TRUE),
      Coverage_SD = sd(coverage, na.rm=TRUE),
      CIW_Mean = mean(CIW, na.rm=TRUE),
      CIW_SD = sd(CIW, na.rm=TRUE),
      AbsBias_Mean = mean(abs.bias, na.rm=TRUE),
      AbsBias_SD = sd(abs.bias, na.rm=TRUE)
    )
  
  # Format the scientific notation for small values with ± notation for SD
  format_sci_with_sd <- function(mean_val, sd_val) {
    if(abs(mean_val) < 0.001 & mean_val != 0) {
      # Scientific notation for very small values
      exponent <- floor(log10(abs(mean_val)))
      mantissa <- mean_val * 10^abs(exponent)
      sd_scaled <- sd_val * 10^abs(exponent)
      return(sprintf("$%.2f \\pm %.2f \\times 10^{%d}$", mantissa, sd_scaled, exponent))
    } else {
      # Regular notation for normal values
      return(sprintf("$%.4f \\pm %.4f$", mean_val, sd_val))
    }
  }
  
  # Group estimators by type
  sl_estimators <- grep("SL", est_summary$Estimator, value = TRUE)
  rnn_estimators <- grep("RNN", est_summary$Estimator, value = TRUE)
  has_sl <- length(sl_estimators) > 0
  has_rnn <- length(rnn_estimators) > 0
  
  # Create LaTeX table header
  latex_table <- "\\begin{table}[ht]
\\centering
\\caption{Performance metrics by estimator, aggregated across time periods and treatment rules. Values are presented as mean $\\pm$ standard deviation.}
\\label{tab:results-estimator}
\\resizebox{0.8\\textwidth}{!}{%
\\begin{tabular}{l|c|c|c}
\\hline
\\textbf{Estimator} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias} \\\\
\\hline\n"
  
  # Add SL estimators if available
  if(has_sl) {
    for(estimator in sl_estimators) {
      row <- est_summary[est_summary$Estimator == estimator, ]
      if(nrow(row) > 0) {
        latex_table <- paste0(latex_table, 
                              estimator, " & ", 
                              format_sci_with_sd(row$Coverage_Mean[1], row$Coverage_SD[1]), " & ", 
                              format_sci_with_sd(row$CIW_Mean[1], row$CIW_SD[1]), " & ", 
                              format_sci_with_sd(row$AbsBias_Mean[1], row$AbsBias_SD[1]), " \\\\ \n")
      }
    }
    
    # Add horizontal line between SL and RNN if both are present
    if(has_rnn) {
      latex_table <- paste0(latex_table, "\\hline\n")
    }
  }
  
  # Add RNN estimators if available
  if(has_rnn) {
    for(estimator in rnn_estimators) {
      row <- est_summary[est_summary$Estimator == estimator, ]
      if(nrow(row) > 0) {
        latex_table <- paste0(latex_table, 
                              estimator, " & ", 
                              format_sci_with_sd(row$Coverage_Mean[1], row$Coverage_SD[1]), " & ", 
                              format_sci_with_sd(row$CIW_Mean[1], row$CIW_SD[1]), " & ", 
                              format_sci_with_sd(row$AbsBias_Mean[1], row$AbsBias_SD[1]), " \\\\ \n")
      }
    }
  }
  
  # Complete the table
  latex_table <- paste0(latex_table, "\\hline
\\end{tabular}
}
\\end{table}")
  
  return(latex_table)
}

# Function to create LaTeX table for estimator and rule metrics
create_estimator_rule_table <- function(results_df) {
  # Calculate summary statistics by estimator and rule
  est_rule_summary <- results_df %>%
    group_by(Estimator, Rule) %>%
    summarize(
      Coverage_Mean = mean(coverage, na.rm=TRUE),
      Coverage_SD = sd(coverage, na.rm=TRUE),
      CIW_Mean = mean(CIW, na.rm=TRUE),
      CIW_SD = sd(CIW, na.rm=TRUE),
      AbsBias_Mean = mean(abs.bias, na.rm=TRUE),
      AbsBias_SD = sd(abs.bias, na.rm=TRUE)
    )
  
  # Format the scientific notation for small values with ± notation for SD
  format_sci_with_sd <- function(mean_val, sd_val) {
    if(abs(mean_val) < 0.001 & mean_val != 0) {
      # Scientific notation for very small values
      exponent <- floor(log10(abs(mean_val)))
      mantissa <- mean_val * 10^abs(exponent)
      sd_scaled <- sd_val * 10^abs(exponent)
      return(sprintf("$%.2f \\pm %.2f \\times 10^{%d}$", mantissa, sd_scaled, exponent))
    } else {
      # Regular notation for normal values
      return(sprintf("$%.4f \\pm %.4f$", mean_val, sd_val))
    }
  }
  
  # Group estimators by type
  sl_estimator_rules <- est_rule_summary[grep("SL", est_rule_summary$Estimator), ]
  rnn_estimator_rules <- est_rule_summary[grep("RNN", est_rule_summary$Estimator), ]
  has_sl <- nrow(sl_estimator_rules) > 0
  has_rnn <- nrow(rnn_estimator_rules) > 0
  
  # Create LaTeX table
  latex_table <- "\\begin{table}[ht]
\\centering
\\caption{Performance metrics by estimator and treatment rule, aggregated across time periods. Values are presented as mean $\\pm$ standard deviation.}
\\label{tab:results-estimator-rule}
\\resizebox{0.8\\textwidth}{!}{%
\\begin{tabular}{ll|c|c|c}
\\hline
\\multicolumn{2}{c|}{\\textbf{Group}} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias} \\\\
\\hline\n"
  
  # Add SL rows if available
  if(has_sl) {
    for(i in 1:nrow(sl_estimator_rules)) {
      row <- sl_estimator_rules[i,]
      latex_table <- paste0(latex_table, 
                            row$Estimator[1], " & ", 
                            row$Rule[1], " & ", 
                            format_sci_with_sd(row$Coverage_Mean[1], row$Coverage_SD[1]), " & ", 
                            format_sci_with_sd(row$CIW_Mean[1], row$CIW_SD[1]), " & ", 
                            format_sci_with_sd(row$AbsBias_Mean[1], row$AbsBias_SD[1]), " \\\\ \n")
    }
    
    # Add horizontal line between SL and RNN if both are present
    if(has_rnn) {
      latex_table <- paste0(latex_table, "\\hline\n")
    }
  }
  
  # Add RNN rows if available
  if(has_rnn) {
    for(i in 1:nrow(rnn_estimator_rules)) {
      row <- rnn_estimator_rules[i,]
      latex_table <- paste0(latex_table, 
                            row$Estimator[1], " & ", 
                            row$Rule[1], " & ", 
                            format_sci_with_sd(row$Coverage_Mean[1], row$Coverage_SD[1]), " & ", 
                            format_sci_with_sd(row$CIW_Mean[1], row$CIW_SD[1]), " & ", 
                            format_sci_with_sd(row$AbsBias_Mean[1], row$AbsBias_SD[1]), " \\\\ \n")
    }
  }
  
  # Complete the table
  latex_table <- paste0(latex_table, "\\hline
\\end{tabular}
}
\\end{table}")
  
  return(latex_table)
}

# Function to create LaTeX table for rule metrics by implementation
create_rule_table <- function(results_df) {
  # Determine which estimator names correspond to which implementation
  sl_estimators <- grep("SL", unique(results_df$Estimator), value = TRUE)
  rnn_estimators <- grep("RNN", unique(results_df$Estimator), value = TRUE)
  
  has_sl <- length(sl_estimators) > 0
  has_rnn <- length(rnn_estimators) > 0
  
  # Create implementation column
  results_df$Implementation <- NA
  if(has_sl) {
    results_df$Implementation[results_df$Estimator %in% sl_estimators] <- "SL"
  }
  if(has_rnn) {
    results_df$Implementation[results_df$Estimator %in% rnn_estimators] <- "RNN"
  }
  
  # Calculate summary statistics by implementation and rule
  rule_summary <- results_df %>%
    filter(!is.na(Implementation)) %>%
    group_by(Implementation, Rule) %>%
    summarize(
      Coverage_Mean = mean(coverage, na.rm=TRUE),
      Coverage_SD = sd(coverage, na.rm=TRUE),
      CIW_Mean = mean(CIW, na.rm=TRUE),
      CIW_SD = sd(CIW, na.rm=TRUE),
      AbsBias_Mean = mean(abs.bias, na.rm=TRUE),
      AbsBias_SD = sd(abs.bias, na.rm=TRUE)
    )
  
  # Format the scientific notation for small values with ± notation for SD
  format_sci_with_sd <- function(mean_val, sd_val) {
    # Ensure we're dealing with scalar values
    if(length(mean_val) > 1 || length(sd_val) > 1) {
      # For vector inputs, just use the first element
      mean_val <- mean_val[1]
      sd_val <- sd_val[1]
    }
    
    # Handle NA values
    if(is.na(mean_val) || is.na(sd_val) || !is.finite(mean_val) || !is.finite(sd_val)) {
      return("NA")
    }
    
    if(abs(mean_val) < 0.001 & mean_val != 0) {
      # Scientific notation for very small values
      exponent <- floor(log10(abs(mean_val)))
      mantissa <- mean_val * 10^abs(exponent)
      sd_scaled <- sd_val * 10^abs(exponent)
      return(sprintf("$%.3f \\pm %.3f \\times 10^{%d}$", mantissa, sd_scaled, exponent))
    } else {
      # Regular notation for normal values - now with 3 digits
      return(sprintf("$%.3f \\pm %.3f$", mean_val, sd_val))
    }
  }
  
  # Create LaTeX table
  latex_table <- "\\begin{table}[ht]
\\centering
\\caption{Performance metrics by treatment rule, aggregated across SL and RNN implementations separately. Values are presented as mean $\\pm$ standard deviation.}
\\label{tab:results-rule}
\\resizebox{0.8\\textwidth}{!}{%
\\begin{tabular}{ll|c|c|c}
\\hline
\\textbf{Implementation} & \\textbf{Rule} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias} \\\\
\\hline\n"
  
  # Process implementations if available
  for(impl in c("SL", "RNN")) {
    impl_data <- filter(rule_summary, Implementation == impl)
    if(nrow(impl_data) == 0) {
      next
    }
    
    # Add multirow if we have multiple rules
    if(nrow(impl_data) > 1) {
      latex_table <- paste0(latex_table, "\\multirow{", nrow(impl_data), "}{*}{", impl, "} ")
    } else {
      latex_table <- paste0(latex_table, impl)
    }
    
    # Add first row
    first_row <- impl_data[1,]
    latex_table <- paste0(latex_table, 
                          " & ", first_row$Rule[1], " & ", 
                          format_sci_with_sd(first_row$Coverage_Mean[1], first_row$Coverage_SD[1]), " & ", 
                          format_sci_with_sd(first_row$CIW_Mean[1], first_row$CIW_SD[1]), " & ", 
                          format_sci_with_sd(first_row$AbsBias_Mean[1], first_row$AbsBias_SD[1]), " \\\\ \n")
    
    # Add remaining rows if any
    if(nrow(impl_data) > 1) {
      for(i in 2:nrow(impl_data)) {
        row <- impl_data[i,]
        latex_table <- paste0(latex_table, 
                              " & ", row$Rule[1], " & ", 
                              format_sci_with_sd(row$Coverage_Mean[1], row$Coverage_SD[1]), " & ", 
                              format_sci_with_sd(row$CIW_Mean[1], row$CIW_SD[1]), " & ", 
                              format_sci_with_sd(row$AbsBias_Mean[1], row$AbsBias_SD[1]), " \\\\ \n")
      }
    }
    
    # Add horizontal line between implementations
    latex_table <- paste0(latex_table, "\\hline\n")
  }
  
  # Complete the table - prevent duplication by not using unnecessary curly braces
  latex_table <- paste0(latex_table, "\\end{tabular}
}
\\end{table}")
  
  return(latex_table)
}

# Function to create positivity table from results
create_positivity_table <- function(results_lstm = NULL, results_sl = NULL) {
  # Check if we have any positivity metrics
  has_lstm_data <- !is.null(results_lstm) && 
    (!is.null(results_lstm$prob_share_tmle) || !is.null(results_lstm$prob_share_tmle_bin))
  has_sl_data <- !is.null(results_sl) && 
    (!is.null(results_sl$prob_share_tmle) || !is.null(results_sl$prob_share_tmle_bin))
  
  if(!has_lstm_data && !has_sl_data) {
    cat("No positivity metrics found in results. Skipping positivity table.\n")
    return(NULL)
  }
  
  # Debug output for positivity metrics
  if(has_lstm_data) {
    cat("Examining LSTM positivity metrics...\n")
    if(!is.null(results_lstm$prob_share_tmle)) {
      cat("LSTM multi-class metrics available. Length:", length(results_lstm$prob_share_tmle), "\n")
      # Check structure of first few entries
      for(i in 1:min(3, length(results_lstm$prob_share_tmle))) {
        cat("Time point", i, "structure:\n")
        if(!is.null(results_lstm$prob_share_tmle[[i]])) {
          cat("  Dimensions:", paste(dim(results_lstm$prob_share_tmle[[i]]), collapse="x"), "\n")
        } else {
          cat("  NULL value\n")
        }
      }
    }
    if(!is.null(results_lstm$prob_share_tmle_bin)) {
      cat("LSTM binary metrics available. Length:", length(results_lstm$prob_share_tmle_bin), "\n")
    }
  }
  
  # Extract positivity metrics from both result sets
  prob_share_sl_multi <- if(has_sl_data) results_sl$prob_share_tmle else NULL
  prob_share_sl_bin <- if(has_sl_data) results_sl$prob_share_tmle_bin else NULL
  prob_share_rnn_multi <- if(has_lstm_data) results_lstm$prob_share_tmle else NULL
  prob_share_rnn_bin <- if(has_lstm_data) results_lstm$prob_share_tmle_bin else NULL
  
  # Time points to report - check for maximum available time point
  max_t_lstm <- 0
  max_t_sl <- 0
  
  if(has_lstm_data) {
    if(!is.null(prob_share_rnn_multi)) max_t_lstm <- max(max_t_lstm, length(prob_share_rnn_multi))
    if(!is.null(prob_share_rnn_bin)) max_t_lstm <- max(max_t_lstm, length(prob_share_rnn_bin))
  }
  
  if(has_sl_data) {
    if(!is.null(prob_share_sl_multi)) max_t_sl <- max(max_t_sl, length(prob_share_sl_multi))
    if(!is.null(prob_share_sl_bin)) max_t_sl <- max(max_t_sl, length(prob_share_sl_bin))
  }
  
  max_t <- max(max_t_lstm, max_t_sl)
  if(max_t <= 4) {
    time_points <- 1:max_t
  } else {
    time_points <- c(1, 2, 3, 4, min(max_t, 12))
  }
  
  # Create LaTeX table
  latex_table <- "\\begin{table}[h]
\\caption{Proportion of cumulative probabilities smaller than 0.025 for static, dynamic, and stochastic treatment rules under different implementations.}
\\label{tab:positivity}
\\resizebox{0.8\\textwidth}{!}{%
\\begin{tabular}{l|"
  
  # Add columns for each time point
  latex_table <- paste0(latex_table, paste(rep("c", length(time_points)), collapse=""), "}\n")
  
  # Add header row
  latex_table <- paste0(latex_table, "\\toprule\n\\multicolumn{", length(time_points) + 1, 
                        "}{c}{\\textbf{Static Rule (\\(\\bar{\\mathbf{d}}^1\\))}} \\\\\n")
  
  latex_table <- paste0(latex_table, "\\textbf{Implementation} & ", 
                        paste0("t=", time_points, collapse=" & "), " \\\\\n")
  
  latex_table <- paste0(latex_table, "\\midrule\n")
  
  # Helper function to correctly calculate proportion below threshold
  format_prob_share <- function(prob_share, t, rule_idx) {
    if(is.null(prob_share) || length(prob_share) < t || is.null(prob_share[[t]])) {
      return("NA")
    }
    
    # Debug the structure of the probability share
    if(is.list(prob_share[[t]])) {
      cat("Positivity metric at time", t, "is a list instead of matrix\n")
      return("NA") # Can't handle list directly
    }
    
    # Make sure it's a matrix or data frame
    prob_mat <- prob_share[[t]]
    if(!is.matrix(prob_mat) && !is.data.frame(prob_mat)) {
      cat("Converting non-matrix to matrix for time", t, "\n")
      # Try to convert to matrix if possible
      if(is.vector(prob_mat)) {
        prob_mat <- matrix(prob_mat, ncol=1)
      } else {
        return("NA")
      }
    }
    
    # Extract column for specific rule
    if((is.matrix(prob_mat) || is.data.frame(prob_mat)) && ncol(prob_mat) >= rule_idx) {
      # Extract the column for this rule
      col_data <- prob_mat[, rule_idx]
      
      # Calculate proportion of values < 0.025
      if(length(col_data) > 0) {
        # Count values below 0.025
        below_threshold <- sum(col_data < 0.025, na.rm=TRUE)
        total_count <- sum(!is.na(col_data))
        
        if(total_count > 0) {
          # Calculate percentage
          percent <- (below_threshold / total_count) * 100
          return(sprintf("%.2f\\%%", percent))
        }
      }
    }
    
    return("NA")
  }
  
  # Add rows for static rule
  rule_idx <- 1 # static rule index
  
  # Only add SL rows if we have SL data
  if(has_sl_data) {
    # Add SL (bin) row
    latex_table <- paste0(latex_table, "SL (bin.)     & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_sl_bin, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
    
    # Add SL (multi) row
    latex_table <- paste0(latex_table, "SL (multi.)   & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_sl_multi, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
  }
  
  # Only add RNN rows if we have RNN data
  if(has_lstm_data) {
    # Add RNN (bin) row
    latex_table <- paste0(latex_table, "RNN (bin.)    & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_rnn_bin, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
    
    # Add RNN (multi) row
    latex_table <- paste0(latex_table, "RNN (multi.)  & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_rnn_multi, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
  }
  
  # Continue with dynamic rule
  latex_table <- paste0(latex_table, "\\midrule\n\\multicolumn{", length(time_points) + 1, 
                        "}{c}{\\textbf{Dynamic Rule (\\(\\bar{\\mathbf{d}}^2\\))}} \\\\\n")
  
  latex_table <- paste0(latex_table, "\\textbf{Implementation} & ", 
                        paste0("t=", time_points, collapse=" & "), " \\\\\n")
  
  latex_table <- paste0(latex_table, "\\midrule\n")
  
  rule_idx <- 2 # dynamic rule index
  
  # Only add SL rows if we have SL data
  if(has_sl_data) {
    # Add SL (bin) row
    latex_table <- paste0(latex_table, "SL (bin.)     & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_sl_bin, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
    
    # Add SL (multi) row
    latex_table <- paste0(latex_table, "SL (multi.)   & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_sl_multi, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
  }
  
  # Only add RNN rows if we have RNN data
  if(has_lstm_data) {
    # Add RNN (bin) row
    latex_table <- paste0(latex_table, "RNN (bin.)    & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_rnn_bin, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
    
    # Add RNN (multi) row
    latex_table <- paste0(latex_table, "RNN (multi.)  & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_rnn_multi, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
  }
  
  # Continue with stochastic rule
  latex_table <- paste0(latex_table, "\\midrule\n\\multicolumn{", length(time_points) + 1, 
                        "}{c}{\\textbf{Stochastic Rule (\\(\\bar{\\mathbf{d}}^3\\))}} \\\\\n")
  
  latex_table <- paste0(latex_table, "\\textbf{Implementation} & ", 
                        paste0("t=", time_points, collapse=" & "), " \\\\\n")
  
  latex_table <- paste0(latex_table, "\\midrule\n")
  
  rule_idx <- 3 # stochastic rule index
  
  # Only add SL rows if we have SL data
  if(has_sl_data) {
    # Add SL (bin) row
    latex_table <- paste0(latex_table, "SL (bin.)     & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_sl_bin, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
    
    # Add SL (multi) row
    latex_table <- paste0(latex_table, "SL (multi.)   & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_sl_multi, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
  }
  
  # Only add RNN rows if we have RNN data
  if(has_lstm_data) {
    # Add RNN (bin) row
    latex_table <- paste0(latex_table, "RNN (bin.)    & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_rnn_bin, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
    
    # Add RNN (multi) row
    latex_table <- paste0(latex_table, "RNN (multi.)  & ")
    for(t in time_points) {
      latex_table <- paste0(latex_table, format_prob_share(prob_share_rnn_multi, t, rule_idx), " & ")
    }
    latex_table <- substr(latex_table, 1, nchar(latex_table) - 3) # Remove trailing " & "
    latex_table <- paste0(latex_table, " \\\\ \n")
  }
  
  # Complete the table
  latex_table <- paste0(latex_table, "\\bottomrule
\\end{tabular}
}
\\end{table}")
  
  return(latex_table)
}

# Create 'sim_results' directory if it doesn't exist
if (!dir.exists("sim_results")) {
  dir.create("sim_results")
}

# Create 'tables' directory for LaTeX tables
if (!dir.exists("tables")) {
  dir.create("tables")
}

# Process LSTM results
cat("Processing LSTM results...\n")
results_lstm <- combine_results(base_dirs, "tmle-lstm")
has_lstm_results <- length(results_lstm) > 0

if(has_lstm_results) {
  saveRDS(results_lstm, "intermediate_results_lstm.rds")
  cat("Saved LSTM results to intermediate_results_lstm.rds\n")
} else {
  cat("No LSTM results found in specified directories, trying to load from file...\n")
  # Try to load existing file if no results found
  if(file.exists("intermediate_results_lstm.rds")) {
    results_lstm <- readRDS("intermediate_results_lstm.rds")
    has_lstm_results <- length(results_lstm) > 0
    if(has_lstm_results) {
      cat("Loaded existing LSTM results from intermediate_results_lstm.rds\n")
    }
  }
}

# Process SL results
cat("Processing SuperLearner (SL) results...\n")
results_sl <- combine_results(base_dirs, "tmle")
has_sl_results <- length(results_sl) > 0

if(has_sl_results) {
  saveRDS(results_sl, "intermediate_results_sl.rds")
  cat("Saved SL results to intermediate_results_sl.rds\n")
} else {
  cat("No SL results found in specified directories, trying to load from file...\n")
  # Try to load existing file if no results found
  if(file.exists("intermediate_results_sl.rds")) {
    results_sl <- readRDS("intermediate_results_sl.rds")
    has_sl_results <- length(results_sl) > 0
    if(has_sl_results) {
      cat("Loaded existing SL results from intermediate_results_sl.rds\n")
    }
  }
}

# Setup dataframes based on available results
if(has_sl_results && has_lstm_results) {
  # Both types of results available
  cat("Both SL and LSTM results available. Creating combined dataframe.\n")
  
  # Define estimator names for both LSTM and SL
  estimator_names <- c(
    "LTMLE-SL (multi.)", "LTMLE-SL (bin.)", "G-Comp-SL",
    "IPTW-SL (multi.)", "IPTW-SL (bin.)",
    "LTMLE-RNN (multi.)", "LTMLE-RNN (bin.)", "G-Comp-RNN",
    "IPTW-RNN (multi.)", "IPTW-RNN (bin.)"
  )
  
  # Create dataframe for SL results
  results_df_sl <- create_results_df(results_sl, estimator_names[1:5])
  
  # Create dataframe for LSTM results
  results_df_lstm <- create_results_df(results_lstm, estimator_names[6:10])
  
  # Check if we have both dataframes
  if(!is.null(results_df_sl) && !is.null(results_df_lstm)) {
    # Combine dataframes
    results.df <- rbind(results_df_sl, results_df_lstm)
    n.estimators <- 10  # 5 estimators * 2 (SL and LSTM)
  } else if(!is.null(results_df_sl)) {
    # Only SL results
    results.df <- results_df_sl
    n.estimators <- 5
  } else if(!is.null(results_df_lstm)) {
    # Only LSTM results
    results.df <- results_df_lstm
    n.estimators <- 5
  } else {
    stop("No valid results dataframes created")
  }
} else if(has_sl_results) {
  # Only SL results available
  cat("Only SL results available.\n")
  
  estimator_names <- c(
    "LTMLE-SL (multi.)", "LTMLE-SL (bin.)", "G-Comp-SL",
    "IPTW-SL (multi.)", "IPTW-SL (bin.)"
  )
  
  results.df <- create_results_df(results_sl, estimator_names)
  if(is.null(results.df)) {
    stop("No valid SL results dataframe created")
  }
  
  n.estimators <- 5  # 5 SL estimators only
} else if(has_lstm_results) {
  # Only LSTM results available
  cat("Only LSTM (RNN) results available.\n")
  
  estimator_names <- c(
    "LTMLE-RNN (multi.)", "LTMLE-RNN (bin.)", "G-Comp-RNN",
    "IPTW-RNN (multi.)", "IPTW-RNN (bin.)"
  )
  
  results.df <- create_results_df(results_lstm, estimator_names)
  if(is.null(results.df)) {
    stop("No valid LSTM results dataframe created")
  }
  
  n.estimators <- 5  # 5 LSTM estimators only
} else {
  stop("No results found. Please check the specified directories.")
}

# Format rule names
proper <- function(x) paste0(toupper(substr(x, 1, 1)), tolower(substring(x, 2)))
results.df$Rule <- proper(results.df$Rule)

# Create coverage rate variable
results.df <- results.df %>% 
  group_by(Estimator, Rule) %>% 
  mutate(CP = mean(coverage, na.rm=TRUE)) 

# Filter out non-finite values for plotting
results.df_filtered <- results.df %>%
  filter(is.finite(abs.bias), is.finite(coverage), is.finite(CIW))

# Fix prob_share_tmle structure if needed
if(has_lstm_results && !is.null(results_lstm$prob_share_tmle)) {
  cat("Checking structure of LSTM positivity metrics...\n")
  
  fixed_prob_share_tmle <- list()
  fixed_prob_share_tmle_bin <- list()
  
  # Fix LSTM multi-class metrics
  if(!is.null(results_lstm$prob_share_tmle)) {
    for(t in 1:length(results_lstm$prob_share_tmle)) {
      # Check if the entry exists
      if(!is.null(results_lstm$prob_share_tmle[[t]])) {
        prob_mat <- results_lstm$prob_share_tmle[[t]]
        
        # Try to fix matrix structure if needed
        if(is.list(prob_mat)) {
          cat("Converting list to matrix for time", t, "\n")
          # Try to extract from list
          if(length(prob_mat) == 3) { # One entry per rule
            # Extract matrices for each rule
            fixed_mat <- do.call(cbind, prob_mat)
            fixed_prob_share_tmle[[t]] <- fixed_mat
          }
        } else if(!is.matrix(prob_mat)) {
          # Try to convert to matrix
          if(is.data.frame(prob_mat)) {
            fixed_prob_share_tmle[[t]] <- as.matrix(prob_mat)
          } else if(is.vector(prob_mat)) {
            fixed_prob_share_tmle[[t]] <- matrix(prob_mat, ncol=3)
          }
        } else {
          # Already a matrix, use as is
          fixed_prob_share_tmle[[t]] <- prob_mat
        }
      }
    }
    
    # Replace original with fixed version
    results_lstm$prob_share_tmle <- fixed_prob_share_tmle
  }
  
  # Fix LSTM binary metrics if available
  if(!is.null(results_lstm$prob_share_tmle_bin)) {
    for(t in 1:length(results_lstm$prob_share_tmle_bin)) {
      # Similar fixes for binary metrics
      if(!is.null(results_lstm$prob_share_tmle_bin[[t]])) {
        prob_mat <- results_lstm$prob_share_tmle_bin[[t]]
        
        # Try to fix matrix structure
        if(is.list(prob_mat)) {
          cat("Converting list to matrix for binary time", t, "\n")
          # Try to extract from list
          if(length(prob_mat) == 3) { # One entry per rule
            fixed_mat <- do.call(cbind, prob_mat)
            fixed_prob_share_tmle_bin[[t]] <- fixed_mat
          }
        } else if(!is.matrix(prob_mat)) {
          # Try to convert to matrix
          if(is.data.frame(prob_mat)) {
            fixed_prob_share_tmle_bin[[t]] <- as.matrix(prob_mat)
          } else if(is.vector(prob_mat)) {
            fixed_prob_share_tmle_bin[[t]] <- matrix(prob_mat, ncol=3)
          }
        } else {
          # Already a matrix, use as is
          fixed_prob_share_tmle_bin[[t]] <- prob_mat
        }
      }
    }
    
    # Replace original with fixed version
    results_lstm$prob_share_tmle_bin <- fixed_prob_share_tmle_bin
  }
}

# Generate LaTeX tables
cat("Generating LaTeX tables...\n")

estimator_table <- create_estimator_table(results.df)
estimator_rule_table <- create_estimator_rule_table(results.df)
rule_table <- create_rule_table(results.df)

# Only create positivity table if necessary data is available
positivity_table <- NULL
if(has_lstm_results || has_sl_results) {
  positivity_table <- create_positivity_table(
    results_lstm = if(has_lstm_results) results_lstm else NULL,
    results_sl = if(has_sl_results) results_sl else NULL
  )
}

# Save LaTeX tables to files - add code to prevent duplicate tables
write(estimator_table, "tables/results_estimator.tex")
write(estimator_rule_table, "tables/results_estimator_rule.tex")
write(rule_table, "tables/results_rule.tex")
if(!is.null(positivity_table)) {
  write(positivity_table, "tables/positivity.tex")
}

# Print confirmation
cat("\nLaTeX tables saved to 'tables' directory.\n")

# Reshape and plot
cat("Creating visualization plots...\n")
results_long <- reshape2::melt(results.df_filtered, id.vars=c("Estimator","Rule", "t"))

# Bias plot
sim.results.bias <- ggplot(
  data=results_long[results_long$variable=="abs.bias" & results_long$value <0.25,],
  aes(x=factor(t), y=value, fill=forcats::fct_rev(Estimator))
) + 
  geom_boxplot(outlier.alpha = 0.3, outlier.size = 1, outlier.stroke = 0.1, lwd=0.25) +
  facet_grid(Rule ~  ., scales = "free") +  
  xlab("Time") + 
  ylab("Abs. diff. btwn. true and estimated counterfactual outcomes") +
  ggtitle(paste0("Absolute bias")) +
  scale_fill_discrete(name = "Estimator:  ") +
  theme(legend.position="bottom") +
  theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
  theme(axis.title=element_text(family="serif", size=14)) +
  theme(axis.text.y=element_text(family="serif", size=12)) +
  theme(axis.text.x=element_text(family="serif", size=12, angle = 0, vjust = 0.5, hjust=0.25)) +
  theme(legend.text=element_text(family="serif", size=12)) +
  theme(legend.title=element_text(family="serif", size=12)) +
  theme(strip.text.x = element_text(family="serif", size=14)) +
  theme(strip.text.y = element_text(family="serif", size=14)) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l =0))) +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l =0))) +
  theme(panel.spacing = unit(1, "lines"))

# Get the ggplot grob
z.bias <- ggplotGrob(sim.results.bias)

# Labels 
labelR <- "Treatment rule"

# Get the positions of the strips in the gtable
posR <- subset(z.bias$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips
width <- z.bias$widths[max(posR$r)]
z.bias <- gtable_add_cols(z.bias, width, max(posR$r))  

# Construct the new strip grobs
stripR <- gTree(name = "Strip_right", children = gList(
  rectGrob(gp = gpar(col = NA, fill = "grey85")),
  textGrob(labelR, rot = -90, gp = gpar(fontsize=16, col = "grey10"))))

# Position the grobs in the gtable
z.bias <- gtable_add_grob(z.bias, stripR, t = min(posR$t)+0.1, l = max(posR$r) + 1, b = max(posR$b)+1, name = "strip-right")

# Add small gaps between strips
z.bias <- gtable_add_cols(z.bias, unit(1/5, "line"), max(posR$r))

# Draw it
grid.newpage()
grid.draw(z.bias)

# Save the plot
ggsave(paste0("sim_results/long_simulation_bias_estimand","_J_",J,"_n_",n,"_R_",R,".png"), plot = z.bias, scale=1.75)

# Coverage plot
sim.results.coverage <- ggplot(
  data=results_long[results_long$variable=="coverage",],
  aes(x=factor(t), y=value, colour=forcats::fct_rev(Estimator), group=forcats::fct_rev(Estimator))
) +
  geom_line() +
  facet_grid(Rule ~  ., scales = "free") +  
  xlab("Time") + 
  ylab("Share of estimated CIs containing true target quantity") + 
  ggtitle(paste0("Coverage probability")) + 
  scale_colour_discrete(name = "Estimator:  ") +
  geom_hline(yintercept = 0.95, linetype="dotted") +
  theme(legend.position="bottom") +
  theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
  theme(axis.title=element_text(family="serif", size=14)) +
  theme(axis.text.y=element_text(family="serif", size=12)) +
  theme(axis.text.x=element_text(family="serif", size=12, angle = 0, vjust = 0.5, hjust=0.25)) +
  theme(legend.text=element_text(family="serif", size=12)) +
  theme(legend.title=element_text(family="serif", size=12)) +
  theme(strip.text.x = element_text(family="serif", size=14)) +
  theme(strip.text.y = element_text(family="serif", size=14)) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l =0))) +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l =0))) +
  theme(panel.spacing = unit(1, "lines"))

# Get the ggplot grob
z.coverage <- ggplotGrob(sim.results.coverage)

# Get the positions of the strips in the gtable
posR <- subset(z.coverage$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips
width <- z.coverage$widths[max(posR$r)]
z.coverage <- gtable_add_cols(z.coverage, width, max(posR$r))  

# Position the grobs in the gtable
z.coverage <- gtable_add_grob(z.coverage, stripR, t = min(posR$t)+0.1, l = max(posR$r) + 1, b = max(posR$b)+1, name = "strip-right")

# Add small gaps between strips
z.coverage <- gtable_add_cols(z.coverage, unit(1/5, "line"), max(posR$r))

# Draw it
grid.newpage()
grid.draw(z.coverage)

# Save the plot
ggsave(paste0("sim_results/long_simulation_coverage_estimand","_J_",J,"_n_",n,"_R_",R,".png"), plot = z.coverage, scale=1.75)

# CI width plot
sim.results.CI.width <- ggplot(
  data=results_long[results_long$variable=="CIW",],
  aes(x=factor(t), y=value, fill=forcats::fct_rev(Estimator))
) + 
  geom_boxplot(outlier.alpha = 0.3, outlier.size = 1, outlier.stroke = 0.1, lwd=0.25) +
  facet_grid(Rule ~  ., scales = "free") +  
  xlab("Time") + 
  ylab("Difference btwn. upper & lower bounds of estimated CIs") + 
  ggtitle(paste0("Confidence interval width")) +
  scale_fill_discrete(name = "Estimator:  ") +
  theme(legend.position="bottom") +
  theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
  theme(axis.title=element_text(family="serif", size=14)) +
  theme(axis.text.y=element_text(family="serif", size=12)) +
  theme(axis.text.x=element_text(family="serif", size=12, angle = 0, vjust = 0.5, hjust=0.25)) +
  theme(legend.text=element_text(family="serif", size=12)) +
  theme(legend.title=element_text(family="serif", size=12)) +
  theme(strip.text.x = element_text(family="serif", size=14)) +
  theme(strip.text.y = element_text(family="serif", size=14)) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l =0))) +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l =0))) +
  theme(panel.spacing = unit(1, "lines"))

# Get the ggplot grob
z.width <- ggplotGrob(sim.results.CI.width)

# Get the positions of the strips in the gtable
posR <- subset(z.width$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips
width <- z.width$widths[max(posR$r)]
z.width <- gtable_add_cols(z.width, width, max(posR$r))  

# Position the grobs in the gtable
z.width <- gtable_add_grob(z.width, stripR, t = min(posR$t)+0.1, l = max(posR$r) + 1, b = max(posR$b)+1, name = "strip-right")

# Add small gaps between strips
z.width <- gtable_add_cols(z.width, unit(1/5, "line"), max(posR$r))

# Draw it
grid.newpage()
grid.draw(z.width)

# Save the plot
ggsave(paste0("sim_results/long_simulation_ci_width_estimand","_J_",J,"_n_",n,"_R_",R,".png"), plot = z.width, scale=1.75)

cat("\nAnalysis complete. Plots saved to the 'sim_results' directory and LaTeX tables saved to the 'tables' directory.\n")