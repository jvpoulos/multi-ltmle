############################################################################################
# Combine results from longitudinal simulations: tables and plots                          #
# (rebuilt 2026-10 for the per-replicate `metrics` / `positivity` data frames saved by     #
#  simLong(); labels come from the data, nothing is hard-coded)                            #
# Usage: Rscript long_sim_plots.R 'outputs/dir1' ['outputs/dir2' ...] [--dgp-figures]        #
#   --dgp-figures: also draw the treatment-adherence and survival figures from one simulated #
#   dataset (replicate r = 1) under the current DGP                                         #
############################################################################################

library(ggplot2)
library(dplyr)

J <- 6
treatment.rules <- c("static", "dynamic", "stochastic")

args <- commandArgs(trailingOnly = TRUE)
dgp_figures <- "--dgp-figures" %in% args
args <- setdiff(args, "--dgp-figures")
# repository root (the directory of this script), used for the DGP figures
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
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
  df$Implementation <- factor(sub(".*-(SL|GLM|RNN).*", "\\1", df$estimator), levels = c("SL", "GLM", "RNN"))
  df$Rule <- factor(proper(df$rule), levels = proper(treatment.rules))
  df$Estimator <- df$estimator
  df
}
results.df <- create_results_df(results)
R <- max(sapply(results, `[[`, "R"))
n <- max(unlist(lapply(results, `[[`, "n")))

# Summaries over replicates and time points: absolute bias as mean (SD); CI width as mean and median
# (a few replicates with extreme weights inflate the mean); coverage as a proportion with its Monte
# Carlo standard error sqrt(p (1 - p) / N)
summarise_metrics <- function(df, ...) {
  df %>% group_by(...) %>% summarise(
    bias_mean = mean(abs_bias, na.rm = TRUE), bias_sd = sd(abs_bias, na.rm = TRUE),
    cover_n = sum(!is.na(cover)), cover = mean(cover, na.rm = TRUE),
    ciw_mean = mean(ciw, na.rm = TRUE), ciw_median = median(ciw, na.rm = TRUE), .groups = "drop") %>%
    mutate(cover_mcse = sqrt(cover * (1 - cover) / cover_n))
}
fmt <- function(m, s) ifelse(is.na(m) | is.nan(m), "--", sprintf("$%.4f \\pm %.4f$", m, s))
fmt_ciw <- function(m, md) ifelse(is.na(m) | is.nan(m), "--", sprintf("$%.4f$ (%.4f)", m, md))
fmt_cover <- function(p, se) ifelse(is.na(p) | is.nan(p), "--", sprintf("$%.3f$ (%.3f)", p, se))

# body: list of row blocks (one per implementation), separated by horizontal rules
latex_table <- function(body, header, caption, label, align) {
  blocks <- sapply(body, function(b) paste(b, collapse = " \\\\ \n"))
  paste0("\\begin{table}[ht]\n\\centering\n\\caption{", caption, "}\n\\label{", label, "}\n",
         "\\resizebox{0.8\\textwidth}{!}{%\n\\begin{tabular}{", align, "}\n\\hline\n", header, " \\\\\n\\hline\n",
         paste(blocks, collapse = " \\\\ \n\\hline\n"), " \\\\ \n\\hline\n\\end{tabular}\n}\n\\end{table}\n")
}
caption_note <- function() paste0("Values are over ", R, " simulation runs ($n=", format(n, big.mark = "{,}"), "$) and target time points ",
                                  "$t=", paste(sort(unique(results.df$t)), collapse = ", "), "$. Coverage probability: proportion of 95\\% CIs ",
                                  "containing the true value (Monte Carlo SE); CI widths: mean (median); absolute bias: mean $\\pm$ SD. ",
                                  "G-computation intervals are approximate (influence curve evaluated at the untargeted fits).")
by_impl <- function(s, row_fun) lapply(split(s, s$Implementation, drop = TRUE), row_fun)

create_estimator_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Estimator) %>% arrange(Implementation, Estimator)
  body <- by_impl(s, function(b) sprintf("%s & %s & %s & %s", b$Estimator, fmt_cover(b$cover, b$cover_mcse), fmt_ciw(b$ciw_mean, b$ciw_median), fmt(b$bias_mean, b$bias_sd)))
  latex_table(body, "\\textbf{Estimator} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias}",
              paste("Performance metrics by estimator, aggregated across time points and treatment rules.", caption_note()),
              "tab:results-estimator", "l|c|c|c")
}

create_estimator_rule_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Estimator, Rule) %>% arrange(Implementation, Estimator, Rule)
  body <- by_impl(s, function(b) sprintf("%s & %s & %s & %s & %s", b$Estimator, b$Rule, fmt_cover(b$cover, b$cover_mcse), fmt_ciw(b$ciw_mean, b$ciw_median), fmt(b$bias_mean, b$bias_sd)))
  latex_table(body, "\\multicolumn{2}{c|}{\\textbf{Group}} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias}",
              paste("Performance metrics by estimator and treatment rule, aggregated across time points.", caption_note()),
              "tab:results-estimator-rule", "ll|c|c|c")
}

create_rule_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Rule) %>% arrange(Implementation, Rule)
  body <- by_impl(s, function(b) sprintf("%s & %s & %s & %s & %s", c(sprintf("\\multirow{%d}{*}{%s}", nrow(b), b$Implementation[1]), rep("", nrow(b) - 1)),
                                         b$Rule, fmt_cover(b$cover, b$cover_mcse), fmt_ciw(b$ciw_mean, b$ciw_median), fmt(b$bias_mean, b$bias_sd)))
  latex_table(body, "\\textbf{Implementation} & \\textbf{Rule} & \\textbf{Coverage prob.} & \\textbf{CI Widths} & \\textbf{Abs. Bias}",
              paste("Performance metrics by treatment rule, aggregated across estimators (LTMLE, IPTW, and G-computation) and time points, separately by implementation.", caption_note()),
              "tab:results-rule", "ll|c|c|c")
}

# Positivity: mean over replicates of the share of uncensored rule-followers whose unbounded cumulative
# probability of the rule path (stochastic rule: cumulative g / g*) is below 0.025 (multinomial treatment model)
create_positivity_table <- function(results, times = c(12, 24, 36)) {
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
  impls <- intersect(c("SL", "GLM", "RNN"), unique(s$Implementation))
  blocks <- sapply(treatment.rules, function(rule) {
    rows <- sapply(impls, function(impl) {
      v <- sapply(times, function(t) { x <- s$prop[s$Implementation == impl & s$rule == rule & s$t == t]; if (length(x) == 0 || is.na(x)) "--" else sprintf("%.3f", x) })
      paste(c(impl, v), collapse = " & ")
    })
    paste0(sprintf("\\multicolumn{%d}{l}{\\textbf{%s Rule}} \\\\ \n", length(times) + 1, proper(rule)),
           "\\textbf{Implementation} & ", paste0("$t=", times, "$", collapse = " & "), " \\\\ \n\\midrule\n",
           paste(rows, collapse = " \\\\ \n"))
  })
  paste0("\\begin{table}[h]\n\\centering\n\\caption{Proportion of uncensored rule-followers whose estimated cumulative probability of following the rule ",
         "is smaller than 0.025, for static, dynamic, and stochastic treatment rules under the SL and RNN implementations (mean over ", R,
         " simulation runs; multinomial treatment model). For the stochastic rule, the cumulative ratio of the estimated to the rule's treatment probabilities is used.}\n",
         "\\label{tab:positivity}\n\\begin{tabular}{l|", paste(rep("c", length(times)), collapse = ""), "}\n\\toprule\n",
         paste(blocks, collapse = " \\\\ \n\\midrule\n"), " \\\\ \n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
}

cat("Generating LaTeX tables...\n")
dir.create("tables", showWarnings = FALSE)
write(create_estimator_table(results.df), "tables/results_estimator.tex")
write(create_estimator_rule_table(results.df), "tables/results_estimator_rule.tex")
write(create_rule_table(results.df), "tables/results_rule.tex")
write(create_positivity_table(results), "tables/positivity.tex")
cat("LaTeX tables saved to 'tables' directory.\n")

# Plots over time: mean absolute bias, coverage proportion (with nominal 0.95 line), median CI width.
# Colour = estimator (treatment model), line type = implementation (SL / RNN)
cat("Creating plots...\n")
dir.create("sim_results", showWarnings = FALSE)
by_time <- summarise_metrics(results.df, Implementation, Estimator, Rule, t)
by_time$Method <- sub("-(SL|GLM|RNN)", "", by_time$Estimator)
plot_metric <- function(df, y, ylab, title, hline = NULL) {
  p <- ggplot(df, aes(x = t, y = .data[[y]], colour = Method, shape = Method, linetype = Implementation,
                      group = interaction(Method, Implementation))) +
    geom_line() + geom_point(size = 1.8) + scale_x_continuous(breaks = sort(unique(df$t))) +
    facet_grid(Rule ~ ., scales = "free_y") +
    xlab("Time (months)") + ylab(ylab) + ggtitle(title) +
    theme_bw(base_family = "serif") + theme(legend.position = "bottom", legend.box = "vertical", plot.title = element_text(hjust = 0.5))
  if (!is.null(hline)) p <- p + geom_hline(yintercept = hline, linetype = "dotted")
  p
}
ggsave(paste0("sim_results/long_simulation_bias_estimand_J_", J, "_n_", n, "_R_", R, ".png"),
       plot_metric(by_time, "bias_mean", "Mean absolute bias", "Absolute bias"), width = 9, height = 8)
ggsave(paste0("sim_results/long_simulation_coverage_estimand_J_", J, "_n_", n, "_R_", R, ".png"),
       plot_metric(by_time[!is.na(by_time$cover), ], "cover", "Coverage probability", "Coverage of 95% CIs", hline = 0.95), width = 9, height = 8)
ggsave(paste0("sim_results/long_simulation_ci_width_estimand_J_", J, "_n_", n, "_R_", R, ".png"),
       plot_metric(by_time[!is.na(by_time$ciw_median), ], "ciw_median", "Median CI width", "Confidence interval width"), width = 9, height = 8)
cat("Plots saved to 'sim_results' directory.\n")

############################################################################################
# Treatment-rule adherence and observed / counterfactual survival in one simulated dataset  #
# (replicate r = 1, current DGP). Adherence: share of uncensored patients whose treatments  #
# up to t (or their event) equal those assigned by the static or dynamic rule (rule_assign). #
# The stochastic rule gives every treatment path a positive probability, so all uncensored  #
# patients are compatible with it. Counterfactual curves: cached large-n truth.             #
############################################################################################
if (dgp_figures) {
  suppressPackageStartupMessages(library(simcausal))
  options(simcausal.verbose = FALSE)
  t.end <- 36
  source(file.path(script_dir, "src", "simcausal_fns.R"))
  owd <- setwd(script_dir); source(file.path("src", "simcausal_dgp.R"), local = TRUE)
  Dset <- set.DAG(D, vecfun = c("StochasticFun"))
  truth <- compute_truth(Dset, t.end)$truth
  setwd(owd)
  Odat <- sim(DAG = Dset, n = n, LTCF = "Y", rndseed = 1, verbose = FALSE)
  A <- sapply(0:t.end, function(k) as.integer(as.character(Odat[[paste0("A_", k)]])))
  Y <- sapply(0:t.end, function(k) as.numeric(Odat[[paste0("Y_", k)]]))
  uncens <- !is.na(Y)                                   # observed through t (EFU after censoring)
  at_risk <- cbind(TRUE, Y[, -ncol(Y), drop = FALSE] == 0); at_risk[is.na(at_risk)] <- FALSE
  follow <- lapply(c("static", "dynamic"), function(rule) {
    f <- rep(TRUE, n); out <- matrix(NA, n, t.end + 1)
    for (k in 0:t.end) {
      d <- rule_assign(rule, k, Odat$V2_0, Odat[[paste0("L1_", k)]], Odat[[paste0("L2_", k)]], Odat[[paste0("L3_", k)]])
      ok <- A[, k + 1] == d; ok[is.na(ok)] <- FALSE
      f <- f & (ok | !at_risk[, k + 1])                # rule only binds while event-free
      out[, k + 1] <- f
    }
    out
  })
  names(follow) <- c("static", "dynamic")
  adherence <- sapply(follow, function(f) sapply(0:t.end, function(k) mean(f[uncens[, k + 1], k + 1])))
  surv_obs <- cbind(sapply(follow, function(f) sapply(0:t.end, function(k) 1 - mean(Y[uncens[, k + 1] & f[, k + 1], k + 1]))),
                    stochastic = sapply(0:t.end, function(k) 1 - mean(Y[uncens[, k + 1], k + 1])))
  surv_true <- rbind(1, 1 - truth[, treatment.rules])
  cat("\nAdherence (share of uncensored patients following the rule) at t = 0, 12, 24, 36:\n"); print(round(adherence[c(1, 13, 25, 37), ], 3))
  cat("Observed survival among uncensored rule-followers (stochastic: all uncensored) at t = 12, 24, 36:\n"); print(round(surv_obs[c(13, 25, 37), ], 3))
  cat("Counterfactual survival (truth) at t = 12, 24, 36:\n"); print(round(surv_true[c(13, 25, 37), ], 3))
  months <- 0:t.end
  png(paste0("sim_results/treatment_adherence_", n, ".png"))
  matplot(months, adherence, type = "l", lty = 1:2, col = 1:2, lwd = 2, ylim = c(0, max(adherence)), xaxt = "n",
          xlab = "Month", ylab = "Share of uncensored patients following the rule", main = "Treatment rule adherence")
  axis(1, at = seq(0, t.end, by = 6)); legend("topright", c("Static", "Dynamic"), lty = 1:2, col = 1:2, lwd = 2, bty = "n")
  dev.off()
  png(paste0("sim_results/survival_plot_observed_", n, ".png"))
  matplot(months, surv_obs, type = "l", lty = 1:3, col = 1:3, lwd = 2, ylim = c(min(surv_obs, surv_true), 1), xaxt = "n",
          xlab = "Month", ylab = "Share of patients without diabetes diagnosis", main = "Observed outcomes")
  axis(1, at = seq(0, t.end, by = 6)); legend("bottomleft", c("Static", "Dynamic", "Stochastic (all uncensored)"), lty = 1:3, col = 1:3, lwd = 2, bty = "n")
  dev.off()
  png(paste0("sim_results/survival_plot_truth_", n, ".png"))
  matplot(months, surv_true, type = "l", lty = 1:3, col = 1:3, lwd = 2, ylim = c(min(surv_obs, surv_true), 1), xaxt = "n",
          xlab = "Month", ylab = "Share of patients without diabetes diagnosis", main = "Counterfactual outcomes")
  axis(1, at = seq(0, t.end, by = 6)); legend("bottomleft", c("Static", "Dynamic", "Stochastic"), lty = 1:3, col = 1:3, lwd = 2, bty = "n")
  dev.off()
  cat("DGP figures saved to 'sim_results' directory.\n")
}
