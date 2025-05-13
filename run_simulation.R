#!/usr/bin/env Rscript

# Simple script to run simulation.R with proper error handling

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Default values if no arguments provided
if (length(args) < 1) {
  args <- c("tmle", "1", "TRUE", "FALSE")
  cat("No arguments provided. Using defaults:", paste(args, collapse=" "), "\n")
}

# Extract estimator from args
estimator <- args[1]
cat("Using estimator:", estimator, "\n")

# Apply general fixes
cat("Applying general fixes with fix_simulation.R...\n")
if (file.exists("fix_simulation.R")) {
  source("fix_simulation.R")
  cat("General fixes applied.\n")
} else {
  cat("Warning: fix_simulation.R not found. Skipping general fixes.\n")
}

# Apply specific patch for standard tmle estimator
if (estimator == "tmle") {
  cat("Applying specific patch for standard tmle estimator...\n")
  if (file.exists("makeshift_tmle.R")) {
    source("makeshift_tmle.R")
    cat("TMLE-specific patches applied.\n")
  } else {
    cat("Warning: makeshift_tmle.R not found. Standard tmle may experience errors.\n")
  }
}

# Setup output directory
simulation_version <- format(Sys.time(), "%Y%m%d")
output_dir <- paste0("./outputs/", simulation_version, "/")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

# Run the simulation script
cat("Running simulation.R with arguments:", paste(args, collapse=" "), "\n")
source("simulation.R")

cat("Simulation completed!\n")