############################################################################################
# simLong(): one replicate of the longitudinal simulation study                            #
############################################################################################

# Simulates one observed dataset from the DGP and estimates psi_t = E[Y_t^d] (cumulative event
# probability at month t under rule d, no censoring) for the static, dynamic and stochastic
# rules with LTMLE (multinomial and binomial treatment models), Hajek IPTW (both treatment
# models) and sequential g-computation. Requires src/tmle_fns.R, src/SL3_fns.R, src/tmle_IC.R,
# src/simcausal_fns.R (and src/lstm.R for estimator = "tmle-lstm") to be sourced.
#
# Inputs:
#   r                 replicate number; also the seed of the simulated data and of the learners
#   Dset              locked simcausal DAG of the data-generating process (src/simcausal_dgp.R)
#   output_dir        directory (with trailing "/") where the replicate's results are saved
#   J                 number of treatments (must be 6)
#   n                 sample size
#   t.end             number of months after baseline (36)
#   gbound, ybound    bounds for the treatment/censoring probabilities and outcome predictions
#   n.folds           cross-validation folds of the Super Learner
#   cores             cores used within the replicate (forked workers)
#   estimator         "tmle" (Super Learner or GLM learners) or "tmle-lstm" (LSTM learners)
#   treatment.rule    label in the results file name ("all": all three rules are estimated)
#   use.SL            "tmle" only: TRUE = Super Learner, FALSE = GLM / multinomial regression
#   scale.continuous  label in the results file name (continuous covariates are not scaled)
#   target.times      months t at which psi_t is estimated (default 1..t.end)
# Output: list with the data frames `metrics` (estimate, SE, truth, bias, coverage and CI width
#   per estimator, rule and target time) and `positivity`, fit failures, timings and settings.
#   The list is also saved to output_dir as longitudinal_simulation_results_estimator_
#   <estimator>_treatment_rule_<rule>_r_<r>_n_<n>_J_<J>_n_folds_<n.folds>_scale_continuous_
#   <scale.continuous>_use_SL_<use.SL>.rds
simLong <- function(r, Dset, output_dir, J = 6, n = 10000, t.end = 36, gbound = c(0.05, 1),
                    ybound = c(0.0001, 0.9999), n.folds = 3, cores = 1, estimator = "tmle",
                    treatment.rule = "all", use.SL = TRUE, scale.continuous = FALSE,
                    target.times = NULL) {
  start_time <- Sys.time()
  if (J != 6) stop("J must be 6")
  if (t.end < 4 || t.end > 36) stop("t.end must be at least 4 and no more than 36")
  if (n.folds < 3) stop("n.folds must be at least 3")
  if (t.end != 36 && estimator == "tmle-lstm") stop("estimator 'tmle-lstm' requires t.end = 36")
  rules <- c("static", "dynamic", "stochastic")
  if (is.null(target.times)) target.times <- 1:t.end
  target.times <- sort(unique(target.times))
  K <- max(target.times)
  truth <- compute_truth(Dset, t.end) # large-n Monte Carlo truth, cached under ./data
  # observed data (C = 1: censored)
  Odat <- sim(DAG = Dset, n = n, LTCF = "Y", rndseed = r, verbose = FALSE)
  # simcausal::sim() re-seeds the RNG at random when it finishes, so seed the learners here
  RNGkind("L'Ecuyer-CMRG")
  set.seed(r)

  # per-component timings: wall clock and CPU (incl. forked workers), in minutes
  clock <- function() {
    pt <- proc.time()
    c(wall = unname(pt["elapsed"]),
      cpu = sum(pt[c("user.self", "sys.self", "user.child", "sys.child")], na.rm = TRUE))
  }
  tic <- clock()
  # treatment (multinomial and binomial) and censoring models, k = 0..K
  if (estimator == "tmle-lstm") {
    # LSTM learners: TensorFlow runs only in forked workers (one TF thread each) when cores > 1,
    # so the parent never initialises TF before forking; with cores = 1 everything runs
    # in-process.
    rnn_spec <- list(learner = "rnn", use_sl = FALSE, n.folds = n.folds, K = K,
                     data_token = paste0("r", r, "_n", n), threads = 1, seed = r * 1000, hp = NULL)
    rnn_seed <- function(offset) modifyList(rnn_spec, list(seed = r * 1000 + offset))
    g_tasks <- list(
      multinomial = function() fit_treatment_models(Odat, K, arm = "multinomial", J = J,
                                                    rnn = rnn_seed(901)),
      binomial = function() fit_treatment_models(Odat, K, arm = "binomial", J = J,
                                                 rnn = rnn_seed(902)),
      censoring = function() fit_censoring_models(Odat, K, J = J, rnn = rnn_seed(903)))
    g_fits <- if (cores > 1) {
      parallel::mclapply(g_tasks, function(f) f(), mc.cores = min(cores, 3),
                         mc.preschedule = FALSE)
    } else {
      lapply(g_tasks, function(f) f())
    }
    crashed <- sapply(g_fits, inherits, "try-error")
    if (any(crashed)) stop("LSTM nuisance fit crashed: ", g_fits[crashed][[1]])
    g_multi <- g_fits$multinomial
    g_bin <- g_fits$binomial
    g_C <- g_fits$censoring
  } else {
    g_multi <- fit_treatment_models(Odat, K, arm = "multinomial", use_sl = use.SL,
                                    n.folds = n.folds, J = J, cores = cores)
    g_bin <- fit_treatment_models(Odat, K, arm = "binomial", use_sl = use.SL,
                                  n.folds = n.folds, J = J, cores = cores)
    g_C <- fit_censoring_models(Odat, K, use_sl = use.SL, n.folds = n.folds, J = J,
                                cores = cores)
  }
  weights_multi <- setNames(lapply(rules, function(rule) {
    rule_weights(Odat, rule, g_multi$g, g_C$gC, K, gbound)
  }), rules)
  weights_bin <- setNames(lapply(rules, function(rule) {
    rule_weights(Odat, rule, g_bin$g, g_C$gC, K, gbound)
  }), rules)

  toc_g <- clock()
  # sequential regressions for every (target time, treatment model) pair; g-computation once
  Q_spec <- if (estimator == "tmle-lstm") rnn_spec else list(use_sl = use.SL, n.folds = n.folds)
  jobs <- expand.grid(t = rev(target.times), arm = c("multinomial", "binomial"),
                      stringsAsFactors = FALSE)
  run_job <- function(j) {
    arm <- jobs$arm[j]
    Q_spec_j <- modifyList(Q_spec, list(arm = arm, seed = r * 1000 + j))
    tryCatch(getTMLELong(Q_spec_j, rules, ltmle_design,
                         if (arm == "multinomial") g_multi$g else g_bin$g, g_C$gC, Odat,
                         if (arm == "multinomial") weights_multi else weights_bin,
                         gbound, ybound, jobs$t[j], gcomp = (arm == "multinomial")),
             error = function(e) {
               message("getTMLELong failed (t=", jobs$t[j], ", ", arm, "): ",
                       conditionMessage(e))
               NULL
             })
  }
  RNGkind("L'Ecuyer-CMRG")
  set.seed(r)
  fits <- if (cores > 1) {
    parallel::mclapply(seq_len(nrow(jobs)), run_job, mc.cores = cores, mc.preschedule = FALSE)
  } else {
    lapply(seq_len(nrow(jobs)), run_job)
  }
  fits <- lapply(fits, function(f) if (inherits(f, "try-error")) NULL else f)
  toc_q <- clock()
  timings <- rbind(nuisance_g_C = (toc_g - tic) / 60, outcome_recursions = (toc_q - toc_g) / 60)
  pick <- function(arm) {
    idx <- which(jobs$arm == arm)
    out <- fits[idx][order(jobs$t[idx])]
    names(out) <- paste0("t=", sort(jobs$t[idx]))
    out
  }
  contrasts_multi <- pick("multinomial")
  contrasts_bin <- pick("binomial")

  tmle_est_var <- TMLE_IC(contrasts_multi, Q_spec)
  tmle_est_var_bin <- TMLE_IC(contrasts_bin, Q_spec)
  iptw_est_var <- TMLE_IC(contrasts_multi, Q_spec, iptw = TRUE)
  iptw_est_var_bin <- TMLE_IC(contrasts_bin, Q_spec, iptw = TRUE)
  gcomp_est_var <- TMLE_IC(contrasts_multi, Q_spec, gcomp = TRUE)

  # performance metrics per estimator, rule and target time (event-probability scale)
  impl <- if (estimator == "tmle-lstm") "RNN" else if (use.SL) "SL" else "GLM"
  est_list <- list("LTMLE-%s (multi.)" = tmle_est_var, "LTMLE-%s (bin.)" = tmle_est_var_bin,
                   "IPTW-%s (multi.)" = iptw_est_var, "IPTW-%s (bin.)" = iptw_est_var_bin,
                   "G-Comp-%s" = gcomp_est_var)
  metrics <- do.call(rbind, lapply(names(est_list), function(nm) {
    ev <- est_list[[nm]]
    data.frame(r = r, estimator = sprintf(nm, impl),
               rule = rep(rules, each = length(target.times)),
               t = rep(target.times, length(rules)),
               est = as.vector(ev$est[paste0("t=", target.times), rules]),
               se = as.vector(ev$se[paste0("t=", target.times), rules]),
               truth = as.vector(truth$truth[target.times, rules]),
               stringsAsFactors = FALSE)
  }))
  metrics$lower <- metrics$est - qnorm(0.975) * metrics$se
  metrics$upper <- metrics$est + qnorm(0.975) * metrics$se
  metrics$bias <- metrics$est - metrics$truth
  metrics$abs_bias <- abs(metrics$bias)
  metrics$cover <- as.numeric(metrics$lower <= metrics$truth & metrics$truth <= metrics$upper)
  metrics$ciw <- metrics$upper - metrics$lower

  # positivity diagnostic: among uncensored subjects following the rule through t, share whose
  # (unbounded) cumulative probability of the rule path is < 0.025 (stochastic rule: prod g / g*)
  positivity <- do.call(rbind, lapply(c("multinomial", "binomial"), function(arm) {
    w <- if (arm == "multinomial") weights_multi else weights_bin
    do.call(rbind, lapply(rules, function(rule) {
      do.call(rbind, lapply(target.times, function(t) {
        p <- w[[rule]]$cum_prob[[t + 1]]
        data.frame(r = r, arm = arm, rule = rule, t = t,
                   n_at_risk = sum(ltmle_uncensored(Odat, t)), n_followers = sum(!is.na(p)),
                   prop_small = if (any(!is.na(p))) mean(p < 0.025, na.rm = TRUE) else NA_real_,
                   stringsAsFactors = FALSE)
      }))
    }))
  }))

  fit_failures <- sum(sapply(fits, function(f) {
    if (is.null(f)) 1 else sum(sapply(f, function(x) x$failures))
  }))
  lstm_epochs <- if (estimator == "tmle-lstm") {
    list(multinomial = g_multi$epochs, binomial = g_bin$epochs, censoring = g_C$epochs,
         hp = g_multi$hp)
  } else {
    NULL
  }
  failures <- c(g_multinomial = g_multi$failures, g_binomial = g_bin$failures,
                g_censoring = g_C$failures, ltmle = fit_failures,
                jobs_failed = sum(sapply(fits, is.null)))
  elapsed_time <- difftime(Sys.time(), start_time, units = "mins")
  iteration_results <- list("iteration" = r, "estimator" = estimator, "n" = n, "J" = J,
                            "t.end" = t.end, "n_folds" = n.folds, "use_SL" = use.SL,
                            "gbound" = gbound, "ybound" = ybound, "target_times" = target.times,
                            "metrics" = metrics, "positivity" = positivity, "failures" = failures,
                            "truth_mc_se" = truth$mc_se[target.times, rules],
                            "lstm_epochs" = lstm_epochs, "timings" = timings,
                            "elapsed_time" = elapsed_time)
  result_filename <- paste0(output_dir, "longitudinal_simulation_results_estimator_", estimator,
                            "_treatment_rule_", treatment.rule, "_r_", r, "_n_", n, "_J_", J,
                            "_n_folds_", n.folds, "_scale_continuous_", scale.continuous,
                            "_use_SL_", use.SL, ".rds")
  saveRDS(iteration_results, result_filename)
  print(paste0("Replicate ", r, " finished in ", round(elapsed_time, 1), " minutes; failures: ",
               paste(names(failures), failures, sep = "=", collapse = ", ")))
  return(iteration_results)
}
