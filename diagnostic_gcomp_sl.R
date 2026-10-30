############################################################################################
# Diagnostic of the bias of G-Comp-SL under the dynamic rule (paper, simulation results).  #
# Repeats the untargeted g-computation recursion for one rule and target time in one       #
# simulated dataset (replicate r), once with Super Learner and once with main-terms        #
# logistic regressions as the outcome models, using the simulation's own code              #
# (process_backward_sequential() in src/tmle_fns.R). At each backward step s it records    #
# the Super Learner ensemble weights and, for both learners, the mean change in the        #
# predicted risk among subjects at risk at s from setting A_s to the rule,                 #
# mean[Q_s(d_s) - Q_s(A_s)].                                                               #
#                                                                                          #
# Usage: Rscript diagnostic_gcomp_sl.R [r] [t] [rule]   (defaults: 1, 36, dynamic)         #
# Runs in one process (about 2 minutes for t = 36) and writes                              #
#   tables/gcomp_sl_diagnostic_steps.csv    per step: ensemble weights, mean pseudo-outcome #
#                                           and the change in predicted risk per learner   #
#   tables/gcomp_sl_diagnostic_summary.csv  estimates, truth, saved simulation estimate,   #
#                                           mean ensemble weights and the shrinkage ratio  #
############################################################################################

args <- commandArgs(trailingOnly = TRUE)
r <- if (length(args) >= 1) as.integer(args[1]) else 1L
t_target <- if (length(args) >= 2) as.integer(args[2]) else 36L
rule <- if (length(args) >= 3) args[3] else "dynamic"
if (!rule %in% c("static", "dynamic")) stop("rule must be 'static' or 'dynamic'")

# settings of the simulation runs (simulation.R)
n <- 10000; J <- 6; t.end <- 36; n.folds <- 3; ybound <- c(0.0001, 0.9999)

for (pkg in c("simcausal", "purrr", "origami", "sl3", "nnet", "ranger", "glmnet", "MASS",
              "progressr", "data.table", "gtools", "dplyr", "readr", "tidyr")) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}
options(simcausal.verbose = FALSE, sl3.verbose = FALSE)
source(file.path("src", "simcausal_fns.R"))
source(file.path("src", "tmle_fns.R"))
source(file.path("src", "SL3_fns.R"))
source(file.path("src", "simcausal_dgp.R"))
Dset <- set.DAG(D, vecfun = c("StochasticFun"))
truth <- compute_truth(Dset, t.end)

# replicate r's observed data, seeded as in simLong()
Odat <- sim(DAG = Dset, n = n, LTCF = "Y", rndseed = r, verbose = FALSE)
RNGkind("L'Ecuyer-CMRG")
set.seed(r)

# record each outcome regression fitted by process_backward_sequential(): ltmle_fit() is
# wrapped so that the fitted prediction function is kept
ltmle_fit_sim <- ltmle_fit
fitted_steps <- list()
ltmle_fit <- function(X, y, type, use_sl = TRUE, n.folds = 3, J = 6) {
  pred <- ltmle_fit_sim(X, y, type, use_sl, n.folds, J)
  fitted_steps[[length(fitted_steps) + 1]] <<- list(pred = pred, type = type)
  pred
}

# untargeted g-computation recursion (as in getTMLELong()) for one learner
gcomp_recursion <- function(use_sl) {
  fitted_steps <<- list()
  Q_next <- NULL
  steps <- vector("list", t_target + 1)
  for (s in t_target:0) {
    Y_s <- as.numeric(Odat[[paste0("Y_", s)]])
    Z <- if (s == t_target) Y_s else ifelse(Y_s == 1, 1, Q_next)
    step <- process_backward_sequential(Odat, s, rule, ltmle_design,
                                        list(use_sl = use_sl, n.folds = n.folds), ybound, Z)
    if (is.null(step)) stop("outcome regression failed at s = ", s)
    fs <- fitted_steps[[length(fitted_steps)]]
    # change in predicted risk from setting A_s to the rule, among subjects at risk at s
    at_risk <- step$at_risk
    a_obs <- as.integer(as.character(Odat[[paste0("A_", s)]]))
    d <- rule_assign(rule, s, Odat$V2_0, Odat[[paste0("L1_", s)]], Odat[[paste0("L2_", s)]],
                     Odat[[paste0("L3_", s)]])
    bound <- function(p) pmin(pmax(p, ybound[1]), ybound[2])
    Q_a <- bound(fs$pred(ltmle_design(Odat, s, A = a_obs)[at_risk, , drop = FALSE]))
    w <- if (use_sl) environment(fs$pred)$fit$coefficients else NULL
    steps[[s + 1]] <- list(s = s, type = fs$type, weights = w, n_at_risk = sum(at_risk),
                           follow = mean(a_obs[at_risk] == d[at_risk]),
                           mean_Z = mean(Z[step$in_R]),
                           delta = mean(step$Q_d[at_risk] - Q_a))
    Q_next <- step$Q_d
  }
  list(psi = mean(Q_next), steps = steps)
}

started <- Sys.time()
fit_sl <- gcomp_recursion(use_sl = TRUE)
fit_glm <- gcomp_recursion(use_sl = FALSE)
message("Recursions finished in ", round(difftime(Sys.time(), started, units = "mins"), 1),
        " minutes")

# per-step table; learners labelled as in Table S1
learner_label <- function(nm) {
  ifelse(grepl("^Lrnr_ranger", nm), "ranger",
  ifelse(grepl("^Lrnr_glmnet.*_1_10_", nm), "lasso",
  ifelse(grepl("^Lrnr_glmnet.*_0.5_10_", nm), "elastic_net",
  ifelse(grepl("^Lrnr_glm_", nm), "glm",
  ifelse(grepl("^Lrnr_mean", nm), "mean", nm)))))
}
learners <- c("ranger", "lasso", "elastic_net", "glm", "mean")
steps <- do.call(rbind, lapply(seq_along(fit_sl$steps), function(i) {
  a <- fit_sl$steps[[i]]; b <- fit_glm$steps[[i]]
  w <- setNames(rep(NA_real_, length(learners)), learners)
  if (!is.null(a$weights)) w[learner_label(names(a$weights))] <- unname(a$weights)
  data.frame(s = a$s, outcome = a$type, n_at_risk = a$n_at_risk, share_following = a$follow,
             mean_pseudo_outcome_sl = a$mean_Z, mean_pseudo_outcome_glm = b$mean_Z,
             delta_sl = a$delta, delta_glm = b$delta, t(w), check.names = FALSE)
}))
steps <- steps[order(-steps$s), ]

# summary: estimates, mean ensemble weights over the steps with continuous pseudo-outcomes
# (s < t, where the Super Learner uses the least-squares learners and the linear
# metalearner), and the shrinkage ratio sum_s delta_sl / sum_s delta_glm over those steps
saved <- readRDS(list.files(file.path("outputs", "sl_n10000_t12-24-36"),
                            pattern = paste0("_r_", r, "_n_"), full.names = TRUE))$metrics
saved_est <- saved$est[saved$estimator == "G-Comp-SL" & saved$rule == rule & saved$t == t_target]
cont <- steps$s < t_target
summary_tab <- data.frame(
  quantity = c("r", "t", "rule", "estimate_sl", "estimate_glm", "truth",
               "saved_simulation_estimate_sl", paste0("mean_weight_", learners),
               "shrinkage_ratio", "share_following_min", "share_following_max"),
  value = c(r, t_target, rule, fit_sl$psi, fit_glm$psi, truth$truth[t_target, rule],
            if (length(saved_est)) saved_est else NA,
            colMeans(steps[cont, learners], na.rm = TRUE),
            sum(steps$delta_sl[cont]) / sum(steps$delta_glm[cont]),
            min(steps$share_following), max(steps$share_following)))

dir.create("tables", showWarnings = FALSE)
write.csv(steps, file.path("tables", "gcomp_sl_diagnostic_steps.csv"), row.names = FALSE)
write.csv(summary_tab, file.path("tables", "gcomp_sl_diagnostic_summary.csv"), row.names = FALSE)
print(summary_tab, row.names = FALSE)
