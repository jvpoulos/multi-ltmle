###################################################################
# LTMLE core (rebuilt 2026-10)                                    #
# Sequential regression (iterated conditional expectations) for   #
# each target time t, targeting with the cumulative clever         #
# covariate, and influence-curve-based inference.                  #
# Nodes per time k: L1_k, L2_k, L3_k, A_k, C_k (1 = censored), Y_k #
###################################################################

# subjects still under observation at time k (uncensored through k-1, no event through k-1)
ltmle_at_risk <- function(dat, k) {
  if (k == 0) return(rep(TRUE, nrow(dat)))
  out <- dat[[paste0("C_", k - 1)]] == 0 & dat[[paste0("Y_", k - 1)]] == 0
  out[is.na(out)] <- FALSE
  out
}

# subjects at risk at k and uncensored at k (outcome Y_k observed)
ltmle_uncensored <- function(dat, k) {
  if (k == 0) return(rep(TRUE, nrow(dat)))
  out <- ltmle_at_risk(dat, k) & dat[[paste0("C_", k)]] == 0
  out[is.na(out)] <- FALSE
  out
}

# One-hot (dummy) columns <prefix>_2, ..., <prefix>_J of a treatment vector a in 1..J.
ltmle_onehot <- function(a, prefix, J = 6) {
  m <- sapply(2:J, function(j) as.numeric(a == j))
  if (!is.matrix(m)) m <- matrix(m, nrow = 1)
  colnames(m) <- paste0(prefix, "_", 2:J)
  m
}

# Design matrix at time k: baseline race (V1), diagnosis (V2), age (V3, optional), L at k and
# k-1, squared change in L1, indicators used by the rules, A_{k-1}, and optionally A_k.
# dat: wide simulated data; A: optional treatment vector at k (observed or set by a rule).
# Returns an n x p numeric matrix.
ltmle_design <- function(dat, k, A = NULL, include_V3 = TRUE, J = 6) {
  n <- nrow(dat)
  L1 <- as.numeric(dat[[paste0("L1_", k)]])
  L2 <- as.numeric(dat[[paste0("L2_", k)]])
  L3 <- as.numeric(dat[[paste0("L3_", k)]])
  lag <- function(v) if (k == 0) rep(0, n) else as.numeric(dat[[paste0(v, "_", k - 1)]])
  V1 <- as.integer(as.character(dat$V1_0))
  V2 <- as.integer(as.character(dat$V2_0))
  X <- cbind(V1_2 = as.numeric(V1 == 2), V1_3 = as.numeric(V1 == 3), V1_4 = as.numeric(V1 == 4),
             V2_2 = as.numeric(V2 == 2), V2_3 = as.numeric(V2 == 3),
             L1 = L1, L2 = L2, L3 = L3, L1_pos = as.numeric(L1 > 0),
             L_any = as.numeric(L1 > 0 | L2 > 0 | L3 > 0),
             L1_lag = lag("L1"), L2_lag = lag("L2"), L3_lag = lag("L3"),
             dL1_sq = (L1 - lag("L1"))^2)
  if (include_V3) X <- cbind(V3 = as.numeric(dat$V3_0), X)
  if (k > 0) {
    X <- cbind(X, ltmle_onehot(as.integer(as.character(dat[[paste0("A_", k - 1)]])), "A_lag", J))
  }
  if (!is.null(A)) X <- cbind(X, ltmle_onehot(A, "A", J))
  X
}

# lapply over time points, in forked processes when cores > 1 (a crashed worker raises an error)
ltmle_lapply <- function(X, FUN, cores = 1) {
  if (cores <= 1) return(lapply(X, FUN))
  out <- parallel::mclapply(X, FUN, mc.cores = cores, mc.preschedule = FALSE)
  crashed <- sapply(out, inherits, "try-error")
  if (any(crashed)) stop("ltmle_lapply: a worker failed: ", out[crashed][[1]])
  out
}

# Fit a regression of y on X with either sl3 Super Learner (use_sl=TRUE) or a parametric model.
# The SL's cross-validation folds are set on the sl3 task (V = n.folds); Lrnr_sl has no cv_folds
# argument.
# type: "binary" (0/1 outcome), "continuous" (outcome in [0,1]), "categorical" (factor 1..J).
# Returns a function predicting on new design matrices, or NULL on failure (logged, never imputed).
ltmle_fit <- function(X, y, type, use_sl = TRUE, n.folds = 3, J = 6) {
  keep <- apply(X, 2, function(x) length(unique(x)) > 1)
  X <- X[, keep, drop = FALSE]
  cols <- colnames(X)
  tryCatch({
    if (type == "categorical") {
      y <- factor(y, levels = 1:J)
      if (use_sl) {
        dt <- data.table::data.table(X); dt$y <- y
        task <- sl3_Task$new(dt, covariates = cols, outcome = "y", outcome_type = "categorical",
                             folds = n.folds)
        fit <- create_treatment_model_sl(n.folds)$train(task)
        return(function(Xnew) {
          dn <- data.table::data.table(Xnew[, cols, drop = FALSE])
          dn$y <- factor(rep(1, nrow(dn)), levels = 1:J)
          p <- unpack_predictions(fit$predict(sl3_Task$new(dn, covariates = cols, outcome = "y",
                                                           outcome_type = "categorical")))
          out <- matrix(0, nrow(Xnew), J); colnames(out) <- 1:J
          out[, colnames(p)] <- p
          out
        })
      }
      df <- data.frame(X); df$y <- droplevels(y)
      fit <- nnet::multinom(y ~ ., data = df, trace = FALSE, MaxNWts = 10000)
      return(function(Xnew) {
        p <- predict(fit, newdata = data.frame(Xnew[, cols, drop = FALSE]), type = "probs")
        if (!is.matrix(p)) {
          lv <- levels(df$y)
          p <- if (length(lv) == 2) cbind(1 - p, p) else matrix(p, nrow = nrow(Xnew))
          colnames(p) <- lv
        }
        out <- matrix(0, nrow(Xnew), J); colnames(out) <- 1:J
        out[, colnames(p)] <- p
        out
      })
    }
    if (length(unique(y)) == 1) {
      const <- unique(y)
      return(function(Xnew) rep(const, nrow(Xnew)))
    }
    if (use_sl) {
      dt <- data.table::data.table(X); dt$y <- y
      if (type == "binary") {
        task <- sl3_Task$new(dt, covariates = cols, outcome = "y", outcome_type = "binomial",
                             folds = n.folds)
        fit <- Lrnr_sl$new(learners = learner_stack_Y, metalearner = metalearner_Y)$train(task)
      } else {
        task <- sl3_Task$new(dt, covariates = cols, outcome = "y", outcome_type = "continuous",
                             folds = n.folds)
        fit <- Lrnr_sl$new(learners = learner_stack_Y_cont,
                           metalearner = metalearner_Y_cont)$train(task)
      }
      return(function(Xnew) {
        dn <- data.table::data.table(Xnew[, cols, drop = FALSE])
        dn$y <- rep_len(c(0, 1), nrow(dn)) # placeholder outcome
        out_type <- if (type == "binary") "binomial" else "continuous"
        as.numeric(fit$predict(sl3_Task$new(dn, covariates = cols, outcome = "y",
                                            outcome_type = out_type)))
      })
    }
    fit <- suppressWarnings(glm.fit(cbind(1, X), y, family = quasibinomial()))
    beta <- fit$coefficients
    beta[is.na(beta)] <- 0
    function(Xnew) as.numeric(plogis(cbind(1, Xnew[, cols, drop = FALSE]) %*% beta))
  }, error = function(e) {
    message("ltmle_fit (", type, ") failed: ", conditionMessage(e))
    NULL
  })
}

# Treatment models g_k(a | history) for k = 0..K among subjects at risk at k.
# arm = "multinomial": one multinomial model per time; "binomial": J separate one-vs-rest
# binary models per time (predictions are not renormalised). Returns a list of n x J matrices
# (NA rows outside the risk set) and the number of failed fits.
# rnn: NULL (Super Learner / GLM) or the LSTM spec built in simLong() (see src/lstm.R).
fit_treatment_models <- function(dat, K, arm = "multinomial", use_sl = TRUE, n.folds = 3, J = 6,
                                 cores = 1, rnn = NULL) {
  n <- nrow(dat)
  if (!is.null(rnn)) {
    res <- tryCatch(lstm_fit_treatment(dat, K, arm, J, rnn), error = function(e) {
      message("fit_treatment_models (LSTM, ", arm, ") failed: ", conditionMessage(e))
      NULL
    })
    if (is.null(res)) {
      empty <- lapply(0:K, function(k) matrix(NA_real_, n, J, dimnames = list(NULL, 1:J)))
      return(list(g = empty, failures = 1))
    }
    return(res)
  }
  g <- ltmle_lapply(0:K, function(k) {
    failures <- 0
    out <- matrix(NA_real_, n, J, dimnames = list(NULL, 1:J))
    rs <- ltmle_at_risk(dat, k)
    X <- ltmle_design(dat, k)[rs, , drop = FALSE]
    a <- as.integer(as.character(dat[[paste0("A_", k)]]))[rs]
    if (arm == "multinomial") {
      pred <- ltmle_fit(X, a, "categorical", use_sl, n.folds, J)
      if (is.null(pred)) return(list(g = out, failures = 1))
      out[rs, ] <- pred(X)
    } else {
      for (j in 1:J) {
        pred <- ltmle_fit(X, as.numeric(a == j), "binary", use_sl, n.folds, J)
        if (is.null(pred)) {
          failures <- failures + 1
          next
        }
        out[rs, j] <- pred(X)
      }
    }
    list(g = out, failures = failures)
  }, cores)
  list(g = lapply(g, `[[`, "g"), failures = sum(sapply(g, `[[`, "failures")))
}

# Censoring models P(C_k = 0 | history, A_k) for k = 1..K among subjects at risk at k.
# Age (V3) is excluded: it affects only censoring (age-out at 65) and no other node, so
# coarsening at random holds given (L, A) history alone, while conditioning on V3 would make
# P(C_k = 0) = 0 after age 65 (a structural positivity violation).
# Returns list(gC = list of length-n vectors for k = 1..K (NA outside the risk set), failures).
fit_censoring_models <- function(dat, K, use_sl = TRUE, n.folds = 3, J = 6, cores = 1,
                                 rnn = NULL) {
  n <- nrow(dat)
  if (!is.null(rnn)) {
    res <- tryCatch(lstm_fit_censoring(dat, K, J, rnn), error = function(e) {
      message("fit_censoring_models (LSTM) failed: ", conditionMessage(e))
      NULL
    })
    if (is.null(res)) {
      return(list(gC = lapply(seq_len(K), function(k) rep(NA_real_, n)), failures = 1))
    }
    return(res)
  }
  gC <- ltmle_lapply(seq_len(K), function(k) {
    out <- rep(NA_real_, n)
    rs <- ltmle_at_risk(dat, k)
    a_k <- as.integer(as.character(dat[[paste0("A_", k)]]))
    X <- ltmle_design(dat, k, A = a_k, include_V3 = FALSE)[rs, , drop = FALSE]
    pred <- ltmle_fit(X, as.numeric(dat[[paste0("C_", k)]][rs] == 0), "binary", use_sl, n.folds,
                      J)
    if (is.null(pred)) return(list(gC = out, failures = 1))
    out[rs] <- pred(X)
    list(gC = out, failures = 0)
  }, cores)
  list(gC = lapply(gC, `[[`, "gC"), failures = sum(sapply(gC, `[[`, "failures")))
}

# Cumulative clever-covariate weights for one rule.
# Deterministic rules: W_s = prod_{k<=s} I(A_k = d_k) / g_k(d_k) * prod_{k=1..s} 1 / P(C_k = 0).
# Stochastic rule: W_s = prod_{k=1..s} g*(A_k | A_{k-1}) / g_k(A_k) * prod_{k=1..s} 1 / P(C_k = 0).
# The per-time density ratio g*/g is bounded above at 1/gbound[1] (for deterministic rules
# g* = 1, i.e. g is bounded below at gbound[1]; bounding g itself would down-weight the
# stochastic rule's switches, whose probability is ~0.01). P(C = 0) is bounded below at
# gbound[1]. cum_prob is the unbounded cumulative probability of the rule path (deterministic)
# or g/g* ratio (stochastic), used for the positivity diagnostic.
# dat: wide data; rule: "static", "dynamic" or "stochastic"; g, gC: outputs of
# fit_treatment_models() and fit_censoring_models(); K: last time point.
# Returns list(W, cum_prob, d, rule): per time k = 0..K, the weights (0 for non-followers, NA
# outside the uncensored risk set), cumulative probabilities and rule assignments.
rule_weights <- function(dat, rule, g, gC, K, gbound) {
  n <- nrow(dat)
  V2 <- dat$V2_0
  cumw <- rep(1, n); follow <- rep(TRUE, n); cum_prob <- rep(1, n)
  W <- vector("list", K + 1); P <- vector("list", K + 1); D <- vector("list", K + 1)
  a_prev <- NULL
  for (k in 0:K) {
    a_obs <- as.integer(as.character(dat[[paste0("A_", k)]]))
    if (rule == "stochastic") {
      if (k > 0) {
        g_obs <- g[[k + 1]][cbind(seq_len(n), a_obs)]
        g_star <- rule_stochastic_density(a_obs, a_prev)
        cumw <- cumw * pmin(g_star / g_obs, 1 / gbound[1]) # density ratio bounded at 1/gbound[1]
        cum_prob <- cum_prob * g_obs / g_star
      }
    } else {
      d <- rule_assign(rule, k, V2, dat[[paste0("L1_", k)]], dat[[paste0("L2_", k)]],
                       dat[[paste0("L3_", k)]])
      D[[k + 1]] <- d
      g_d <- g[[k + 1]][cbind(seq_len(n), d)]
      follow <- follow & (a_obs == d)
      cumw <- cumw / pmax(g_d, gbound[1])
      cum_prob <- cum_prob * g_d
    }
    if (k > 0) cumw <- cumw / pmax(gC[[k]], gbound[1])
    in_R <- ltmle_uncensored(dat, k)
    W[[k + 1]] <- ifelse(in_R, ifelse(follow, cumw, 0), NA_real_)
    P[[k + 1]] <- ifelse(in_R & follow, cum_prob, NA_real_)
    a_prev <- a_obs
  }
  list(W = W, cum_prob = P, d = D, rule = rule)
}

###################################################################
# One step s of the backward recursion for one rule:              #
# regress the pseudo-outcome Z_s on history among subjects at     #
# risk and uncensored at s, then predict under observed A_s and   #
# under the rule for everyone at risk at s.                       #
# tmle_contrasts: pseudo-outcome vector Z_s (length n)            #
# essential_covars_Y: design function (dat, k, A) -> matrix       #
# initial_model_for_Y_sl_cont: list(use_sl, n.folds)              #
###################################################################
process_backward_sequential <- function(tmle_dat, t, tmle_rules, essential_covars_Y,
                                        initial_model_for_Y_sl_cont, ybound, tmle_contrasts,
                                        time.censored=NULL) {
  J <- 6
  s <- t
  rule <- tmle_rules
  n <- nrow(tmle_dat)
  in_R <- ltmle_uncensored(tmle_dat, s)
  at_risk <- ltmle_at_risk(tmle_dat, s)
  a_obs <- as.integer(as.character(tmle_dat[[paste0("A_", s)]]))
  Z <- tmle_contrasts
  bound <- function(p) pmin(pmax(p, ybound[1]), ybound[2])
  if (identical(initial_model_for_Y_sl_cont$learner, "rnn")) {
    # LSTM regression on the subject's last 12 months of history ending at s (src/lstm.R)
    rnn_pred <- tryCatch(
      lstm_outcome_step(tmle_dat, s, Z, in_R, rule, initial_model_for_Y_sl_cont, J),
      error = function(e) {
        message("LSTM outcome regression failed (s=", s, ", ", rule, "): ", conditionMessage(e))
        NULL
      })
    if (is.null(rnn_pred)) return(NULL)
    Q_obs <- rep(NA_real_, n); Q_d <- rep(NA_real_, n); Q_a <- NULL
    Q_obs[in_R] <- bound(rnn_pred(in_R))
    if (rule == "stochastic" && s > 0) {
      Q_a <- matrix(NA_real_, n, J)
      for (a in 1:J) Q_a[at_risk, a] <- bound(rnn_pred(at_risk, rep(a, n)))
    } else if (rule == "stochastic") {
      # A_0 keeps its natural distribution under the stochastic rule
      Q_d[at_risk] <- bound(rnn_pred(at_risk))
    } else {
      d <- rule_assign(rule, s, tmle_dat$V2_0, tmle_dat[[paste0("L1_", s)]],
                       tmle_dat[[paste0("L2_", s)]], tmle_dat[[paste0("L3_", s)]])
      Q_d[at_risk] <- bound(rnn_pred(at_risk, d))
    }
    return(list(Q_obs = Q_obs, Q_d = Q_d, Q_a = Q_a, in_R = in_R, at_risk = at_risk))
  }
  type <- if (all(Z[in_R] %in% c(0, 1))) "binary" else "continuous"
  pred <- ltmle_fit(essential_covars_Y(tmle_dat, s, A = a_obs)[in_R, , drop = FALSE], Z[in_R],
                    type, initial_model_for_Y_sl_cont$use_sl, initial_model_for_Y_sl_cont$n.folds,
                    J)
  if (is.null(pred)) return(NULL)
  Q_obs <- rep(NA_real_, n)
  Q_obs[in_R] <- bound(pred(essential_covars_Y(tmle_dat, s, A = a_obs)[in_R, , drop = FALSE]))
  Q_a <- NULL
  Q_d <- rep(NA_real_, n)
  if (rule == "stochastic") {
    if (s == 0) {
      # A_0 keeps its natural distribution under the stochastic rule
      X_s <- essential_covars_Y(tmle_dat, s, A = a_obs)[at_risk, , drop = FALSE]
      Q_d[at_risk] <- bound(pred(X_s))
    } else {
      a_prev <- as.integer(as.character(tmle_dat[[paste0("A_", s - 1)]]))
      Q_a <- matrix(NA_real_, n, J)
      for (a in 1:J) {
        X_a <- essential_covars_Y(tmle_dat, s, A = rep(a, n))[at_risk, , drop = FALSE]
        Q_a[at_risk, a] <- bound(pred(X_a))
      }
    }
  } else {
    d <- rule_assign(rule, s, tmle_dat$V2_0, tmle_dat[[paste0("L1_", s)]],
                     tmle_dat[[paste0("L2_", s)]], tmle_dat[[paste0("L3_", s)]])
    Q_d[at_risk] <- bound(pred(essential_covars_Y(tmle_dat, s, A = d)[at_risk, , drop = FALSE]))
  }
  list(Q_obs = Q_obs, Q_d = Q_d, Q_a = Q_a, in_R = in_R, at_risk = at_risk)
}

###################################################################
# LTMLE, IPTW (Hajek) and sequential g-computation of             #
# psi_t = E[Y_t^d] for one target time t (= t.end argument).      #
# initial_model_for_Y: list(use_sl, n.folds) for the Q regressions#
# tmle_rules: rule names; tmle_covars_Y: design function          #
# g_preds_bounded / C_preds_bounded: treatment / censoring fits   #
# obs.treatment: wide observed data; obs.rules: optional cached   #
# rule_weights() output (computed from g and C if NULL)           #
# Returns, per rule, point estimates and influence curves.        #
###################################################################
getTMLELong <- function(initial_model_for_Y, tmle_rules, tmle_covars_Y, g_preds_bounded,
                        C_preds_bounded, obs.treatment, obs.rules, gbound, ybound, t.end,
                        analysis = FALSE, debug = FALSE, gcomp = TRUE) {
  dat <- obs.treatment
  tstar <- t.end
  n <- nrow(dat)
  # recursion identifiers (used by the LSTM learner to warm-start within a recursion; ignored by
  # SL/GLM)
  spec_tmle <- modifyList(initial_model_for_Y, list(recursion = "tmle", target = tstar))
  spec_gcomp <- modifyList(initial_model_for_Y, list(recursion = "gcomp", target = tstar))
  if (is.null(tmle_covars_Y)) tmle_covars_Y <- ltmle_design
  out <- list()
  for (rule in tmle_rules) {
    rw <- if (!is.null(obs.rules) && !is.null(obs.rules[[rule]])) obs.rules[[rule]] else
      rule_weights(dat, rule, g_preds_bounded, C_preds_bounded, tstar, gbound)
    failures <- 0
    Y_t <- as.numeric(dat[[paste0("Y_", tstar)]])
    # targeted (LTMLE) and untargeted (g-computation) recursions
    Qstar_next <- NULL; Qg_next <- NULL
    D <- rep(0, n); D_g <- rep(0, n)
    eps <- rep(NA_real_, tstar + 1)
    ok <- TRUE
    for (s in tstar:0) {
      in_R <- ltmle_uncensored(dat, s)
      Y_s <- as.numeric(dat[[paste0("Y_", s)]])
      Z <- if (s == tstar) Y_t else ifelse(Y_s == 1, 1, Qstar_next)
      step <- process_backward_sequential(dat, s, rule, tmle_covars_Y, spec_tmle, ybound, Z)
      if (is.null(step)) { failures <- failures + 1; ok <- FALSE; break }
      # targeting: weighted logistic fluctuation with offset logit(Q_s(observed A_s)), weights W_s
      W_s <- rw$W[[s + 1]]
      w <- W_s[in_R]
      if (anyNA(w)) { # treatment/censoring fit failed
        failures <- failures + 1
        ok <- FALSE
        break
      }
      if (sum(w) > 0) {
        fl <- suppressWarnings(glm(Z[in_R] ~ 1, offset = qlogis(step$Q_obs[in_R]), weights = w,
                                   family = quasibinomial()))
        eps[s + 1] <- coef(fl)[1]
        if (!is.finite(eps[s + 1])) { failures <- failures + 1; ok <- FALSE; break }
      } else {
        eps[s + 1] <- 0
        failures <- failures + 1  # no rule-followers left to target on
      }
      upd <- function(q) plogis(qlogis(q) + eps[s + 1])
      Qstar_obs <- upd(step$Q_obs)
      Qstar_d <- if (!is.null(step$Q_a)) {
        a_prev <- as.integer(as.character(dat[[paste0("A_", s - 1)]]))
        rowSums(sapply(1:6, function(a) rule_stochastic_density(a, a_prev) * upd(step$Q_a[, a])))
      } else upd(step$Q_d)
      D[in_R] <- D[in_R] + w * (Z[in_R] - Qstar_obs[in_R])
      Qstar_next <- Qstar_d
      if (gcomp) {
        Zg <- if (s == tstar) Y_t else ifelse(Y_s == 1, 1, Qg_next)
        step_g <- if (s == tstar) step else {
          process_backward_sequential(dat, s, rule, tmle_covars_Y, spec_gcomp, ybound, Zg)
        }
        if (is.null(step_g)) { failures <- failures + 1; gcomp <- FALSE } else {
          # influence curve at the untargeted fits (approximate inference for g-computation)
          D_g[in_R] <- D_g[in_R] + w * (Zg[in_R] - step_g$Q_obs[in_R])
          Qg_next <- if (!is.null(step_g$Q_a)) {
            a_prev <- as.integer(as.character(dat[[paste0("A_", s - 1)]]))
            rowSums(sapply(1:6, function(a) rule_stochastic_density(a, a_prev) * step_g$Q_a[, a]))
          } else step_g$Q_d
        }
      }
    }
    if (ok) {
      psi <- mean(Qstar_next)
      ic <- D + Qstar_next - psi
    } else {
      psi <- NA_real_; ic <- rep(NA_real_, n)
    }
    # IPTW (Hajek): weight at min(t, event time) for subjects observed through then
    w_last <- rep(0, n); done <- rep(FALSE, n)
    for (s in 0:tstar) {
      in_R <- ltmle_uncensored(dat, s)
      stop_here <- in_R & !done & (s == tstar | dat[[paste0("Y_", s)]] == 1)
      stop_here[is.na(stop_here)] <- FALSE
      w_last[stop_here] <- rw$W[[s + 1]][stop_here]
      done <- done | stop_here
    }
    Yw <- ifelse(is.na(Y_t), 0, Y_t)
    if (anyNA(w_last)) w_last[] <- NA_real_
    psi_iptw <- if (isTRUE(sum(w_last) > 0)) sum(w_last * Yw) / sum(w_last) else NA_real_
    ic_iptw <- if (isTRUE(sum(w_last) > 0)) {
      w_last * (Yw - psi_iptw) / mean(w_last)
    } else {
      rep(NA_real_, n)
    }
    out[[rule]] <- list(
      psi = psi, ic = ic,
      psi_iptw = psi_iptw, ic_iptw = ic_iptw,
      psi_gcomp = if (gcomp && ok) mean(Qg_next) else NA_real_,
      ic_gcomp = if (gcomp && ok) D_g + Qg_next - mean(Qg_next) else rep(NA_real_, n),
      eps = eps, n_followers = sum(w_last > 0, na.rm = TRUE), failures = failures)
  }
  out
}
