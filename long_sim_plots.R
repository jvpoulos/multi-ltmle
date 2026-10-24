############################################################################################
# Tables and figures of the paper from the per-replicate results saved by simLong()       #
# Usage: Rscript long_sim_plots.R 'outputs/dir1' ['outputs/dir2' ...] [--dgp-figures]      #
#   --dgp-figures: also draw the DAG and the treatment-adherence and survival figures from  #
#   one simulated dataset (replicate r = 1)                                                  #
# Run from the repository root. Outputs (file name = number in the paper; "S" = Supporting  #
# Information):                                                                             #
#   tables/Table1_results_estimator.tex         Table 1  (by estimator)                     #
#   tables/Table2_results_rule.tex              Table 2  (by treatment rule)                #
#   tables/TableS2_results_estimator_rule.tex   Table S2 (by estimator and rule)            #
#   tables/TableS3_positivity.tex               Table S3 (positivity diagnostic)            #
#   tables/monte_carlo_se.csv                   Monte Carlo SEs of the summaries            #
#   tables/TableS4_run_times.tex                Table S4 (computation time per replicate)   #
#   tables/run_times.csv                        computation times of each replicate         #
#   sim_results/FigureS2a_bias.png              Figure S2a (absolute bias over time)        #
#   sim_results/FigureS2b_coverage.png          Figure S2b (coverage over time)             #
#   sim_results/FigureS2c_ci_width.png          Figure S2c (CI width over time)             #
#   with --dgp-figures:                                                                     #
#   sim_results/FigureA1_DAG.png (.pdf)         Figure A1 (DAG of the DGP)                  #
#   sim_results/FigureS1a_treatment_adherence.png, FigureS1b_survival_observed.png,         #
#   sim_results/FigureS1c_survival_truth.png    Figure S1a-c                                #
# Table S1 (hyperparameters) is not generated: it lists settings from src/SL3_fns.R and     #
# src/utils.py.                                                                             #
############################################################################################

library(ggplot2)
library(dplyr)

J <- 6
treatment.rules <- c("static", "dynamic", "stochastic")

args <- commandArgs(trailingOnly = TRUE)
dgp_figures <- "--dgp-figures" %in% args
args <- setdiff(args, "--dgp-figures")
# repository root (the directory of this script), used for the DGP figures
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
script_dir <- dirname(normalizePath(script_file))
if (length(args) == 0) {
  cat("Usage: Rscript long_sim_plots.R 'outputs/dir1' 'outputs/dir2' ...\n")
  stop("No directories specified.")
}
base_dirs <- args
cat("Processing directories:", paste(base_dirs, collapse = ", "), "\n")

# All per-replicate results files of one estimator ("tmle" or "tmle-lstm") under base_dir
find_results_files <- function(base_dir, estimator_type) {
  pattern <- paste0("longitudinal_simulation_results_estimator_", estimator_type,
                    "_treatment_rule_all_r_\\d+_.*\\.rds$")
  list.files(path = base_dir, pattern = pattern, recursive = TRUE, full.names = TRUE)
}

# Replicate number r from a results file name
get_iteration <- function(filename) {
  as.numeric(sub(".*_r_(\\d+)_.*", "\\1", basename(filename)))
}

# Read one replicate; returns NULL (with a message) for files without the metrics data frame
process_results_file <- function(file_path) {
  result <- tryCatch(readRDS(file_path), error = function(e) {
    message("Cannot read ", file_path, ": ", conditionMessage(e))
    NULL
  })
  if (is.null(result) || is.null(result$metrics)) {
    message("Skipping ", basename(file_path), " (no metrics data frame)")
    return(NULL)
  }
  list(metrics = result$metrics, positivity = result$positivity, failures = result$failures,
       n = result$n, elapsed = as.numeric(result$elapsed_time, units = "mins"),
       timings = result$timings, r = result$iteration)
}

# Combine all replicates of one estimator (de-duplicated by replicate number). Returns
# list(metrics, positivity, failures, n, R, elapsed, timings, r) or NULL if there are no results.
combine_results <- function(base_dirs, estimator_type) {
  files <- unlist(lapply(base_dirs, find_results_files, estimator_type = estimator_type))
  files <- files[!duplicated(get_iteration(files))]
  res <- Filter(Negate(is.null), lapply(files, process_results_file))
  if (length(res) == 0) return(NULL)
  list(metrics = do.call(rbind, lapply(res, `[[`, "metrics")),
       positivity = do.call(rbind, lapply(res, `[[`, "positivity")),
       failures = do.call(rbind, lapply(res, `[[`, "failures")),
       n = unique(sapply(res, `[[`, "n")), R = length(res),
       elapsed = sapply(res, `[[`, "elapsed"), timings = lapply(res, `[[`, "timings"),
       r = sapply(res, `[[`, "r"))
}

results <- list(tmle = combine_results(base_dirs, "tmle"),
                "tmle-lstm" = combine_results(base_dirs, "tmle-lstm"))
results <- Filter(Negate(is.null), results)
if (length(results) == 0) stop("No results found. Please check the specified directories.")
for (nm in names(results)) {
  cat(nm, ": R =", results[[nm]]$R, "replicates, n =", paste(results[[nm]]$n, collapse = ","),
      ", median minutes per replicate =", round(median(results[[nm]]$elapsed), 1), "\n")
  cat("  failures (sum over replicates):\n")
  print(colSums(results[[nm]]$failures))
}

proper <- function(x) paste0(toupper(substr(x, 1, 1)), tolower(substring(x, 2)))

# Replicate-level data frame with implementation (SL / GLM / RNN) and readable rule labels
create_results_df <- function(results) {
  df <- do.call(rbind, lapply(results, `[[`, "metrics"))
  df$Implementation <- factor(sub(".*-(SL|GLM|RNN).*", "\\1", df$estimator),
                              levels = c("SL", "GLM", "RNN"))
  df$Rule <- factor(proper(df$rule), levels = proper(treatment.rules))
  df$Estimator <- df$estimator
  df
}
results.df <- create_results_df(results)
R <- max(sapply(results, `[[`, "R"))
n <- max(unlist(lapply(results, `[[`, "n")))

# Summaries over replicates and time points: absolute bias as mean (SD); CI width as mean and
# median (a few replicates with extreme weights inflate the mean); coverage as a proportion
# with its Monte Carlo standard error sqrt(p (1 - p) / N)
summarise_metrics <- function(df, ...) {
  df %>% group_by(...) %>% summarise(
    bias_mean = mean(abs_bias, na.rm = TRUE), bias_sd = sd(abs_bias, na.rm = TRUE),
    cover_n = sum(!is.na(cover)), cover = mean(cover, na.rm = TRUE),
    ciw_mean = mean(ciw, na.rm = TRUE), ciw_median = median(ciw, na.rm = TRUE),
    .groups = "drop") %>%
    mutate(cover_mcse = sqrt(cover * (1 - cover) / cover_n))
}
fmt <- function(m, s) ifelse(is.na(m) | is.nan(m), "--", sprintf("$%.4f \\pm %.4f$", m, s))
fmt_ciw <- function(m, md) ifelse(is.na(m) | is.nan(m), "--", sprintf("$%.4f$ (%.4f)", m, md))
fmt_cover <- function(p, se) ifelse(is.na(p) | is.nan(p), "--", sprintf("$%.3f$ (%.3f)", p, se))

# LaTeX table; body: list of row blocks (one per implementation), separated by horizontal rules
latex_table <- function(body, header, caption, label, align) {
  blocks <- sapply(body, function(b) paste(b, collapse = " \\\\ \n"))
  paste0("\\begin{table}[ht]\n\\centering\n\\caption{", caption, "}\n\\label{", label, "}\n",
         "\\resizebox{0.8\\textwidth}{!}{%\n\\begin{tabular}{", align, "}\n\\hline\n", header,
         " \\\\\n\\hline\n", paste(blocks, collapse = " \\\\ \n\\hline\n"),
         " \\\\ \n\\hline\n\\end{tabular}\n}\n\\end{table}\n")
}
caption_note <- function() {
  paste0("Values are over ", R, " simulation runs ($n=", format(n, big.mark = "{,}"),
         "$) and target time points $t=", paste(sort(unique(results.df$t)), collapse = ", "),
         "$. Coverage probability: proportion of 95\\% CIs containing the true value (Monte ",
         "Carlo SE); CI widths: mean (median); absolute bias: mean $\\pm$ SD over all ",
         "replicate-by-cell values. G-computation intervals are approximate (influence curve ",
         "evaluated at the untargeted fits).")
}
by_impl <- function(s, row_fun) lapply(split(s, s$Implementation, drop = TRUE), row_fun)
metric_cells <- function(b) {
  paste(fmt_cover(b$cover, b$cover_mcse), fmt_ciw(b$ciw_mean, b$ciw_median),
        fmt(b$bias_mean, b$bias_sd), sep = " & ")
}

# Table 1: performance by estimator
create_estimator_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Estimator) %>%
    arrange(Implementation, Estimator)
  body <- by_impl(s, function(b) paste(b$Estimator, metric_cells(b), sep = " & "))
  latex_table(body, paste("\\textbf{Estimator} & \\textbf{Coverage prob.} &",
                          "\\textbf{CI Widths} & \\textbf{Abs. Bias}"),
              paste("Performance metrics by estimator, aggregated across time points and",
                    "treatment rules.", caption_note()),
              "tab:results-estimator", "l|c|c|c")
}

# Table S2: performance by estimator and treatment rule
create_estimator_rule_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Estimator, Rule) %>%
    arrange(Implementation, Estimator, Rule)
  body <- by_impl(s, function(b) paste(b$Estimator, b$Rule, metric_cells(b), sep = " & "))
  latex_table(body, paste("\\multicolumn{2}{c|}{\\textbf{Group}} & \\textbf{Coverage prob.} &",
                          "\\textbf{CI Widths} & \\textbf{Abs. Bias}"),
              paste("Performance metrics by estimator and treatment rule, aggregated across",
                    "time points.", caption_note()),
              "tab:results-estimator-rule", "ll|c|c|c")
}

# Table 2: performance by treatment rule
create_rule_table <- function(results_df) {
  s <- summarise_metrics(results_df, Implementation, Rule) %>% arrange(Implementation, Rule)
  body <- by_impl(s, function(b) {
    impl <- c(sprintf("\\multirow{%d}{*}{%s}", nrow(b), b$Implementation[1]),
              rep("", nrow(b) - 1))
    paste(impl, b$Rule, metric_cells(b), sep = " & ")
  })
  latex_table(body, paste("\\textbf{Implementation} & \\textbf{Rule} & \\textbf{Coverage prob.}",
                          "& \\textbf{CI Widths} & \\textbf{Abs. Bias}"),
              paste("Performance metrics by treatment rule, aggregated across estimators (LTMLE,",
                    "IPTW, and G-computation) and time points, separately by implementation.",
                    caption_note()),
              "tab:results-rule", "ll|c|c|c")
}

# Table S3 (positivity): mean over replicates of the share of uncensored rule-followers whose
# unbounded cumulative probability of the rule path (stochastic rule: cumulative g / g*) is
# below 0.025 (multinomial treatment model)
create_positivity_table <- function(results, times = c(12, 24, 36)) {
  pos <- do.call(rbind, lapply(names(results), function(nm) {
    p <- results[[nm]]$positivity
    if (is.null(p)) return(NULL)
    p$Implementation <- if (nm == "tmle-lstm") {
      "RNN"
    } else {
      unique(sub(".*-(SL|GLM).*", "\\1", results[[nm]]$metrics$estimator))[1]
    }
    p
  }))
  pos <- pos[pos$arm == "multinomial", ]
  times <- intersect(times, sort(unique(pos$t)))
  s <- pos %>% filter(t %in% times) %>% group_by(Implementation, rule, t) %>%
    summarise(prop = mean(prop_small, na.rm = TRUE), followers = mean(n_followers),
              .groups = "drop")
  impls <- intersect(c("SL", "GLM", "RNN"), unique(s$Implementation))
  blocks <- sapply(treatment.rules, function(rule) {
    rows <- sapply(impls, function(impl) {
      v <- sapply(times, function(t) {
        x <- s$prop[s$Implementation == impl & s$rule == rule & s$t == t]
        if (length(x) == 0 || is.na(x)) "--" else sprintf("%.3f", x)
      })
      paste(c(impl, v), collapse = " & ")
    })
    paste0(sprintf("\\multicolumn{%d}{l}{\\textbf{%s Rule}} \\\\ \n", length(times) + 1,
                   proper(rule)),
           "\\textbf{Implementation} & ", paste0("$t=", times, "$", collapse = " & "),
           " \\\\ \n\\midrule\n", paste(rows, collapse = " \\\\ \n"))
  })
  paste0("\\begin{table}[h]\n\\centering\n\\caption{Proportion of uncensored rule-followers ",
         "whose estimated cumulative probability of following the rule is smaller than 0.025, ",
         "for static, dynamic, and stochastic treatment rules under the SL and RNN ",
         "implementations (mean over ", R, " simulation runs; multinomial treatment model). ",
         "For the stochastic rule, the cumulative ratio of the estimated to the rule's ",
         "treatment probabilities is used.}\n",
         "\\label{tab:positivity}\n\\begin{tabular}{l|",
         paste(rep("c", length(times)), collapse = ""), "}\n\\toprule\n",
         paste(blocks, collapse = " \\\\ \n\\midrule\n"),
         " \\\\ \n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
}

# Monte Carlo standard errors of the summaries in Table S2: the replicate-level mean over target
# times is averaged over replicates, so MCSE = SD of the replicate means / sqrt(R)
monte_carlo_se <- function(results_df) {
  results_df %>%
    group_by(Implementation, Estimator, Rule, r) %>%
    summarise(abs_bias = mean(abs_bias, na.rm = TRUE), cover = mean(cover, na.rm = TRUE),
              ciw = mean(ciw, na.rm = TRUE), .groups = "drop") %>%
    group_by(Implementation, Estimator, Rule) %>%
    summarise(R = n(), abs_bias_mean = mean(abs_bias, na.rm = TRUE),
              abs_bias_mcse = sd(abs_bias, na.rm = TRUE) / sqrt(sum(!is.na(abs_bias))),
              cover_mean = mean(cover, na.rm = TRUE),
              cover_mcse = sd(cover, na.rm = TRUE) / sqrt(sum(!is.na(cover))),
              ciw_mean = mean(ciw, na.rm = TRUE),
              ciw_mcse = sd(ciw, na.rm = TRUE) / sqrt(sum(!is.na(ciw))), .groups = "drop") %>%
    arrange(Implementation, Estimator, Rule)
}

# Computation time of each replicate (minutes): total wall clock, and wall clock and CPU time
# (including the forked workers) of the treatment and censoring models and of the outcome
# regressions (backward recursion with targeting)
run_times <- function(results) {
  do.call(rbind, lapply(names(results), function(nm) {
    x <- results[[nm]]
    tm <- lapply(x$timings, function(m) if (is.null(m)) rep(NA_real_, 4) else as.vector(m))
    tm <- do.call(rbind, tm)
    data.frame(Implementation = if (nm == "tmle-lstm") "RNN" else
                 unique(sub(".*-(SL|GLM).*", "\\1", x$metrics$estimator))[1],
               r = x$r, total_wall = x$elapsed, g_C_wall = tm[, 1], Q_wall = tm[, 2],
               g_C_cpu = tm[, 3], Q_cpu = tm[, 4])
  }))
}
# Table S4: median (IQR) over replicates of the computation times, and the total over replicates
runtime_hardware <- paste("Each replicate used 3 worker processes on a laptop with a 4-core",
                          "Intel Core i7-8550U CPU (no GPU).")
create_runtime_table <- function(rt) {
  miqr <- function(x) {
    if (all(is.na(x))) return("--")
    q <- quantile(x, c(0.5, 0.25, 0.75), na.rm = TRUE)
    sprintf("%.1f (%.1f--%.1f)", q[1], q[2], q[3])
  }
  rows <- sapply(split(rt, factor(rt$Implementation, levels = c("SL", "GLM", "RNN")),
                       drop = TRUE), function(b) {
    paste(b$Implementation[1], miqr(b$g_C_wall), miqr(b$Q_wall), miqr(b$total_wall),
          miqr(b$g_C_cpu + b$Q_cpu), sprintf("%.1f", sum(b$total_wall) / 60), sep = " & ")
  })
  paste0("\\begin{table}[ht]\n\\centering\n\\caption{Computation time per simulation run ",
         "(minutes; median and interquartile range over ", R, " runs with $n=",
         format(n, big.mark = "{,}"), "$ and target time points $t=",
         paste(sort(unique(results.df$t)), collapse = ", "), "$). Treatment and censoring ",
         "models: multinomial and binomial treatment models and censoring models at all time ",
         "points; outcome regressions: the backward recursions with targeting for all target ",
         "times and rules. The LTMLE, IPTW and g-computation estimates of a run share these ",
         "fits. CPU time includes the worker processes. ", runtime_hardware, "}\n",
         "\\label{tab:run-times}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l|cccc|c}\n",
         "\\hline\n & \\multicolumn{3}{c}{\\textbf{Wall clock}} & \\textbf{CPU time} & ",
         "\\textbf{Total} \\\\\n\\textbf{Implementation} & \\textbf{Treatment, censoring} & ",
         "\\textbf{Outcome regressions} & \\textbf{Run} & \\textbf{Run} & \\textbf{(hours)} \\\\\n",
         "\\hline\n", paste(rows, collapse = " \\\\ \n"),
         " \\\\ \n\\hline\n\\end{tabular}\n}\n\\end{table}\n")
}

cat("Generating LaTeX tables...\n")
dir.create("tables", showWarnings = FALSE)
write(create_estimator_table(results.df), file.path("tables", "Table1_results_estimator.tex"))
write(create_rule_table(results.df), file.path("tables", "Table2_results_rule.tex"))
write(create_estimator_rule_table(results.df),
      file.path("tables", "TableS2_results_estimator_rule.tex"))
write(create_positivity_table(results), file.path("tables", "TableS3_positivity.tex"))
write.csv(monte_carlo_se(results.df), file.path("tables", "monte_carlo_se.csv"),
          row.names = FALSE)
rt <- run_times(results)
write(create_runtime_table(rt), file.path("tables", "TableS4_run_times.tex"))
write.csv(rt, file.path("tables", "run_times.csv"), row.names = FALSE)
cat("Tables saved to 'tables' directory.\n")

# Figure S2a-c: mean absolute bias, coverage proportion (with nominal 0.95 line) and median CI
# width over time. Colour = estimator (treatment model), line type = implementation (SL / RNN)
cat("Creating plots...\n")
dir.create("sim_results", showWarnings = FALSE)
by_time <- summarise_metrics(results.df, Implementation, Estimator, Rule, t)
by_time$Method <- sub("-(SL|GLM|RNN)", "", by_time$Estimator)
plot_metric <- function(df, y, ylab, title, hline = NULL) {
  p <- ggplot(df, aes(x = t, y = .data[[y]], colour = Method, shape = Method,
                      linetype = Implementation, group = interaction(Method, Implementation))) +
    geom_line() + geom_point(size = 1.8) + scale_x_continuous(breaks = sort(unique(df$t))) +
    facet_grid(Rule ~ ., scales = "free_y") +
    xlab("Time (months)") + ylab(ylab) + ggtitle(title) +
    theme_bw(base_family = "serif") +
    theme(legend.position = "bottom", legend.box = "vertical",
          plot.title = element_text(hjust = 0.5))
  if (!is.null(hline)) p <- p + geom_hline(yintercept = hline, linetype = "dotted")
  p
}
ggsave(file.path("sim_results", "FigureS2a_bias.png"),
       plot_metric(by_time, "bias_mean", "Mean absolute bias", "Absolute bias"),
       width = 9, height = 8)
ggsave(file.path("sim_results", "FigureS2b_coverage.png"),
       plot_metric(by_time[!is.na(by_time$cover), ], "cover", "Coverage probability",
                   "Coverage of 95% CIs", hline = 0.95),
       width = 9, height = 8)
ggsave(file.path("sim_results", "FigureS2c_ci_width.png"),
       plot_metric(by_time[!is.na(by_time$ciw_median), ], "ciw_median", "Median CI width",
                   "Confidence interval width"),
       width = 9, height = 8)
cat("Plots saved to 'sim_results' directory.\n")

############################################################################################
# DGP figures (--dgp-figures):                                                             #
# Figure A1: DAG of the DGP for t = 0..3.                                                  #
# Figure S1a-c: treatment-rule adherence and observed / counterfactual survival in one     #
# simulated dataset (replicate r = 1). Adherence: share of uncensored patients whose        #
# treatments up to t (or their event) equal those assigned by the static or dynamic rule   #
# (rule_assign). The stochastic rule gives every treatment path a positive probability, so  #
# all uncensored patients are compatible with it. Counterfactual curves: cached large-n     #
# truth.                                                                                    #
############################################################################################
if (dgp_figures) {
  suppressPackageStartupMessages(library(simcausal))
  options(simcausal.verbose = FALSE)
  t.end <- 36
  source(file.path(script_dir, "src", "simcausal_fns.R"))
  source(file.path(script_dir, "src", "plotDAG_improved.R"))
  owd <- setwd(script_dir)
  source(file.path("src", "simcausal_dgp.R"), local = TRUE)
  Dset <- set.DAG(D, vecfun = c("StochasticFun"))
  truth <- compute_truth(Dset, t.end)$truth
  setwd(owd)

  # Figure A1 (simcausal node order: by time, then L1, L2, L3, A, C, Y)
  dag_labels <- c("V^1", "V^2", "V^3",
                  "L^1_0", "L^2_0", "L^3_0", "A_0",
                  "L^1_1", "L^2_1", "L^3_1", "A_1", "C_1", "Y_1",
                  "L^1_2", "L^2_2", "L^3_2", "A_2", "C_2", "Y_2",
                  "L^1_3", "L^2_3", "L^3_3", "A_3", "C_3", "Y_3")
  draw_dag <- function() {
    plotDAG_improved(Dset, excludeattrs = c("C_0", "Y_0"), xjitter = 0, yjitter = 0, tmax = 3,
                     customvlabs = dag_labels, node_size = 8, label_size = 1.5, label_dist = 0,
                     use_latex = TRUE, arrow_size_mult = 1.5,
                     vertex_attrs = list(frame.width = 1.0, color = "white"),
                     edge_attrs = list(width = 0.8, color = "darkgray"), mar = c(0, 0, 0, 0))
  }
  png(file.path("sim_results", "FigureA1_DAG.png"), width = 1600, height = 820, res = 150,
      pointsize = 12, bg = "white")
  draw_dag()
  dev.off()
  pdf(file.path("sim_results", "FigureA1_DAG.pdf"), width = 12, height = 6.2)
  draw_dag()
  dev.off()

  # Figure S1a-c
  Odat <- sim(DAG = Dset, n = n, LTCF = "Y", rndseed = 1, verbose = FALSE)
  A <- sapply(0:t.end, function(k) as.integer(as.character(Odat[[paste0("A_", k)]])))
  Y <- sapply(0:t.end, function(k) as.numeric(Odat[[paste0("Y_", k)]]))
  uncens <- !is.na(Y) # observed through t (EFU after censoring)
  at_risk <- cbind(TRUE, Y[, -ncol(Y), drop = FALSE] == 0)
  at_risk[is.na(at_risk)] <- FALSE
  follow <- lapply(c("static", "dynamic"), function(rule) {
    f <- rep(TRUE, n)
    out <- matrix(NA, n, t.end + 1)
    for (k in 0:t.end) {
      d <- rule_assign(rule, k, Odat$V2_0, Odat[[paste0("L1_", k)]], Odat[[paste0("L2_", k)]],
                       Odat[[paste0("L3_", k)]])
      ok <- A[, k + 1] == d
      ok[is.na(ok)] <- FALSE
      f <- f & (ok | !at_risk[, k + 1]) # rule only binds while event-free
      out[, k + 1] <- f
    }
    out
  })
  names(follow) <- c("static", "dynamic")
  adherence <- sapply(follow, function(f) {
    sapply(0:t.end, function(k) mean(f[uncens[, k + 1], k + 1]))
  })
  surv_obs <- cbind(sapply(follow, function(f) {
    sapply(0:t.end, function(k) 1 - mean(Y[uncens[, k + 1] & f[, k + 1], k + 1]))
  }), stochastic = sapply(0:t.end, function(k) 1 - mean(Y[uncens[, k + 1], k + 1])))
  surv_true <- rbind(1, 1 - truth[, treatment.rules])
  cat("\nAdherence (share of uncensored patients following the rule) at t = 0, 12, 24, 36:\n")
  print(round(adherence[c(1, 13, 25, 37), ], 3))
  cat("Observed survival among uncensored rule-followers (stochastic: all uncensored)",
      "at t = 12, 24, 36:\n")
  print(round(surv_obs[c(13, 25, 37), ], 3))
  cat("Counterfactual survival (truth) at t = 12, 24, 36:\n")
  print(round(surv_true[c(13, 25, 37), ], 3))
  months <- 0:t.end
  ylab_surv <- "Share of patients without diabetes diagnosis"
  png(file.path("sim_results", "FigureS1a_treatment_adherence.png"))
  matplot(months, adherence, type = "l", lty = 1:2, col = 1:2, lwd = 2,
          ylim = c(0, max(adherence)), xaxt = "n", xlab = "Month",
          ylab = "Share of uncensored patients following the rule",
          main = "Treatment rule adherence")
  axis(1, at = seq(0, t.end, by = 6))
  legend("topright", c("Static", "Dynamic"), lty = 1:2, col = 1:2, lwd = 2, bty = "n")
  dev.off()
  png(file.path("sim_results", "FigureS1b_survival_observed.png"))
  matplot(months, surv_obs, type = "l", lty = 1:3, col = 1:3, lwd = 2,
          ylim = c(min(surv_obs, surv_true), 1), xaxt = "n", xlab = "Month", ylab = ylab_surv,
          main = "Observed outcomes")
  axis(1, at = seq(0, t.end, by = 6))
  legend("bottomleft", c("Static", "Dynamic", "Stochastic (all uncensored)"), lty = 1:3,
         col = 1:3, lwd = 2, bty = "n")
  dev.off()
  png(file.path("sim_results", "FigureS1c_survival_truth.png"))
  matplot(months, surv_true, type = "l", lty = 1:3, col = 1:3, lwd = 2,
          ylim = c(min(surv_obs, surv_true), 1), xaxt = "n", xlab = "Month", ylab = ylab_surv,
          main = "Counterfactual outcomes")
  axis(1, at = seq(0, t.end, by = 6))
  legend("bottomleft", c("Static", "Dynamic", "Stochastic"), lty = 1:3, col = 1:3, lwd = 2,
         bty = "n")
  dev.off()
  cat("DGP figures saved to 'sim_results' directory.\n")
}
