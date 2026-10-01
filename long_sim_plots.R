############################################################################################
# Combine results from longitudinal simulations: tables and plots                          #
# (rebuilt 2026-10 for the per-replicate `metrics` / `positivity` data frames saved by     #
#  simLong(); labels come from the data, nothing is hard-coded)                            #
# Usage: Rscript long_sim_plots.R 'outputs/dir1' ['outputs/dir2' ...]                       #
############################################################################################

library(ggplot2)
library(dplyr)

J <- 6
treatment.rules <- c("static", "dynamic", "stochastic")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cat("Usage: Rscript long_sim_plots.R 'outputs/dir1' 'outputs/dir2' ...\n")
  stop("No directories specified.")
}
base_dirs <- args
cat("Processing directories:", paste(base_dirs, collapse = ", "), "\n")

# Find all per-replicate results files
find_results_files <- function(base_dir, estimator_type) {
  pattern <- paste0("longitudinal_simulation_results_estimator_", estimator_type, "_treatment_rule_all_r_\\d+_.*\\.rds$")
  list.files(path = base_dir, pattern = pattern, recursive = TRUE, full.names = TRUE)
}

# Extract replicate number from filename
get_iteration <- function(filename) {
  as.numeric(sub(".*_r_(\\d+)_.*", "\\1", basename(filename)))
}

# Read one replicate; returns NULL (with a message) for files without the metrics data frame
process_results_file <- function(file_path) {
  result <- tryCatch(readRDS(file_path), error = function(e) { message("Cannot read ", file_path, ": ", conditionMessage(e)); NULL })
  if (is.null(result) || is.null(result$metrics)) {
    message("Skipping ", basename(file_path), " (no metrics data frame; produced by an older version of simulation.R)")
    return(NULL)
  }
  list(metrics = result$metrics, positivity = result$positivity, failures = result$failures,
       n = result$n, elapsed = as.numeric(result$elapsed_time, units = "mins"))
}

# Combine all replicates (de-duplicated by estimator type and replicate number)
combine_results <- function(base_dirs, estimator_type) {
  files <- unlist(lapply(base_dirs, find_results_files, estimator_type = estimator_type))
  files <- files[!duplicated(get_iteration(files))]
  res <- Filter(Negate(is.null), lapply(files, process_results_file))
  if (length(res) == 0) return(NULL)
  list(metrics = do.call(rbind, lapply(res, `[[`, "metrics")),
       positivity = do.call(rbind, lapply(res, `[[`, "positivity")),
       failures = do.call(rbind, lapply(res, `[[`, "failures")),
       n = unique(sapply(res, `[[`, "n")), R = length(res),
       elapsed = sapply(res, `[[`, "elapsed"))
}

results <- list(tmle = combine_results(base_dirs, "tmle"), "tmle-lstm" = combine_results(base_dirs, "tmle-lstm"))
results <- Filter(Negate(is.null), results)
if (length(results) == 0) stop("No results found. Please check the specified directories.")
for (nm in names(results)) {
  cat(nm, ": R =", results[[nm]]$R, "replicates, n =", paste(results[[nm]]$n, collapse = ","),
      ", median minutes per replicate =", round(median(results[[nm]]$elapsed), 1), "\n")
  cat("  failures (sum over replicates):\n"); print(colSums(results[[nm]]$failures))
}

proper <- function(x) paste0(toupper(substr(x, 1, 1)), tolower(substring(x, 2)))

# Replicate-level data frame with implementation (SL / GLM / RNN) and readable rule labels
create_results_df <- function(results) {
  df <- do.call(rbind, lapply(results, `[[`, "metrics"))
  df$Implementation <- sub(".*-(SL|GLM|RNN).*", "\\1", df$estimator)
  df$Rule <- factor(proper(df$rule), levels = proper(treatment.rules))
  df$Estimator <- df$estimator
  df
}
results.df <- create_results_df(results)
R <- max(sapply(results, `[[`, "R"))
n <- max(unlist(lapply(results, `[[`, "n")))

# Summaries: mean (SD) of absolute bias and CI width over replicates and time points; coverage as a
# proportion with its Monte Carlo standard error sqrt(p (1 - p) / N)
summarise_metrics <- function(df, ...) {
  df %>% group_by(...) %>% summarise(
    bias_mean = mean(abs_bias, na.rm = TRUE), bias_sd = sd(abs_bias, na.rm = TRUE),
    cover_n = sum(!is.na(cover)), cover = mean(cover, na.rm = TRUE),
    ciw_mean = mean(ciw, na.rm = TRUE), ciw_sd = sd(ciw, na.rm = TRUE), .groups = "drop") %>%
    mutate(cover_mcse = sqrt(cover * (1 - cover) / cover_n))
}
fmt <- function(m, s) ifelse(is.na(m) | is.nan(m), "--", sprintf("$%.3f \\pm %.3f$", m, s))
fmt_cover <- function(p, se) ifelse(is.na(p) | is.nan(p), "--", sprintf("$%.3f$ (%.3f)", p, se))

latex_table <- function(body, header, caption, label, align) {
  paste0("\\begin{table}[ht]\n\\centering\n\\caption{", caption, "}\n\\label{", label, "}\n",
         "\\resizebox{0.8\\textwidth}{!}{%\n\\begin{tabular}{", align, "}\n\\hline\n", header, " \\\\\n\\hline\n",
         paste(body, collapse = " \\\\\n"), " \\\\\n\\hline\n\\end{tabular}\n}\n\\end{table}\n")
}
caption_note <- function() paste0("Absolute bias and CI width: mean $\\pm$ SD over ", R, " simulation runs and target time points ($n=",
                                  format(n, big.mark = ","), "$); coverage: proportion of 95\\% CIs containing the truth (Monte Carlo SE). ",
                                  "G-computation has no influence-curve-based CI.")

create_estimator_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Estimator) %>% arrange(Implementation, Estimator)
  body <- sprintf("%s & %s & %s & %s", s$Estimator, fmt_cover(s$cover, s$cover_mcse), fmt(s$ciw_mean, s$ciw_sd), fmt(s$bias_mean, s$bias_sd))
  latex_table(body, "\\textbf{Estimator} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias}",
              paste("Performance metrics by estimator, aggregated across time points and treatment rules.", caption_note()),
              "tab:results-estimator", "l|c|c|c")
}

create_estimator_rule_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Estimator, Rule) %>% arrange(Implementation, Estimator, Rule)
  body <- sprintf("%s & %s & %s & %s & %s", s$Estimator, s$Rule, fmt_cover(s$cover, s$cover_mcse), fmt(s$ciw_mean, s$ciw_sd), fmt(s$bias_mean, s$bias_sd))
  latex_table(body, "\\multicolumn{2}{c|}{\\textbf{Group}} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias}",
              paste("Performance metrics by estimator and treatment rule, aggregated across time points.", caption_note()),
              "tab:results-estimator-rule", "ll|c|c|c")
}

create_rule_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Rule) %>% arrange(Implementation, Rule)
  body <- sprintf("%s & %s & %s & %s & %s", s$Implementation, s$Rule, fmt_cover(s$cover, s$cover_mcse), fmt(s$ciw_mean, s$ciw_sd), fmt(s$bias_mean, s$bias_sd))
  latex_table(body, "\\textbf{Implementation} & \\textbf{Rule} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias}",
              paste("Performance metrics by treatment rule, aggregated across estimators and time points, separately by implementation.", caption_note()),
              "tab:results-rule", "ll|c|c|c")
}

# Positivity: mean over replicates of the share of uncensored rule-followers whose unbounded cumulative
# probability of the rule path (stochastic rule: cumulative g / g*) is below 0.025 (multinomial treatment model)
create_positivity_table <- function(results, times = c(1, 2, 3, 4, 12, 24, 36)) {
  pos <- do.call(rbind, lapply(names(results), function(nm) {
    p <- results[[nm]]$positivity
    if (is.null(p)) return(NULL)
    p$Implementation <- if (nm == "tmle-lstm") "RNN" else unique(sub(".*-(SL|GLM).*", "\\1", results[[nm]]$metrics$estimator))[1]
    p
  }))
  pos <- pos[pos$arm == "multinomial", ]
  times <- intersect(times, sort(unique(pos$t)))
  s <- pos %>% filter(t %in% times) %>% group_by(Implementation, rule, t) %>%
    summarise(prop = mean(prop_small, na.rm = TRUE), followers = mean(n_followers), .groups = "drop")
  rows <- unlist(lapply(treatment.rules, function(rule) {
    c(sprintf("\\multicolumn{%d}{l}{\\textbf{%s rule}}", length(times) + 1, proper(rule)),
      sapply(unique(s$Implementation), function(impl) {
        v <- sapply(times, function(t) { x <- s$prop[s$Implementation == impl & s$rule == rule & s$t == t]; if (length(x) == 0 || is.na(x)) "--" else sprintf("%.3f", x) })
        paste(c(impl, v), collapse = " & ")
      }))
  }))
  paste0("\\begin{table}[h]\n\\centering\n\\caption{Proportion of uncensored rule-followers with cumulative probability of the rule path below 0.025, ",
         "by treatment rule and time point (mean over ", R, " simulation runs; multinomial treatment model).}\n\\label{tab:positivity}\n",
         "\\begin{tabular}{l", paste(rep("c", length(times)), collapse = ""), "}\n\\toprule\n\\textbf{Implementation} & ",
         paste0("$t=", times, "$", collapse = " & "), " \\\\\n\\midrule\n", paste(rows, collapse = " \\\\\n"),
         " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
}

cat("Generating LaTeX tables...\n")
dir.create("tables", showWarnings = FALSE)
write(create_estimator_table(results.df), "tables/results_estimator.tex")
write(create_estimator_rule_table(results.df), "tables/results_estimator_rule.tex")
write(create_rule_table(results.df), "tables/results_rule.tex")
write(create_positivity_table(results), "tables/positivity.tex")
cat("LaTeX tables saved to 'tables' directory.\n")

# Plots over time: mean absolute bias, coverage proportion (with nominal 0.95 line), mean CI width
cat("Creating plots...\n")
dir.create("sim_results", showWarnings = FALSE)
by_time <- summarise_metrics(results.df, Implementation, Estimator, Rule, t)
plot_metric <- function(df, y, ylab, title, hline = NULL) {
  p <- ggplot(df, aes(x = t, y = .data[[y]], colour = Estimator, shape = Estimator)) +
    geom_line() + geom_point(size = 1.8) +
    facet_grid(Rule ~ ., scales = "free_y") +
    xlab("Time (months)") + ylab(ylab) + ggtitle(title) +
    theme_bw(base_family = "serif") + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5))
  if (!is.null(hline)) p <- p + geom_hline(yintercept = hline, linetype = "dotted")
  p
}
ggsave(paste0("sim_results/long_simulation_bias_estimand_J_", J, "_n_", n, "_R_", R, ".png"),
       plot_metric(by_time, "bias_mean", "Mean absolute bias", "Absolute bias"), width = 9, height = 8)
ggsave(paste0("sim_results/long_simulation_coverage_estimand_J_", J, "_n_", n, "_R_", R, ".png"),
       plot_metric(by_time[!is.na(by_time$cover), ], "cover", "Coverage probability", "Coverage of 95% CIs", hline = 0.95), width = 9, height = 8)
ggsave(paste0("sim_results/long_simulation_ci_width_estimand_J_", J, "_n_", n, "_R_", R, ".png"),
       plot_metric(by_time[!is.na(by_time$ciw_mean), ], "ciw_mean", "Mean CI width", "Confidence interval width"), width = 9, height = 8)
cat("Plots saved to 'sim_results' directory.\n")
