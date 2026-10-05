###################################################################
# R bridge to the LSTM learners of the LTMLE core (rebuilt 2026-10)#
# Python side: src/utils.py, src/train_lstm.py, src/test_lstm.py.  #
# Arrays are passed in memory (no CSV files). Python/TensorFlow is #
# initialised lazily in the process that first needs it, so forked #
# workers (parallel::mclapply) each get their own TF runtime; the  #
# parent process must not initialise TF before forking.            #
###################################################################

lstm_env <- new.env()

# Select the project's Python interpreter without initialising Python.
lstm_python <- function(python = "./myenv/bin/python") {
  reticulate::use_python(python, required = TRUE)
  invisible(TRUE)
}

# Python modules for this process; (re)configures threads, seed and hyperparameters.
# spec: list(threads, seed, hp) as built in simLong().
lstm_modules <- function(spec) {
  if (is.null(lstm_env$mods) || !identical(lstm_env$pid, Sys.getpid())) {
    if (!is.null(lstm_env$pid) && !identical(lstm_env$pid, Sys.getpid())) {
      warning("lstm_modules: Python was initialised in the parent process before forking; TensorFlow is not fork-safe")
    }
    mods <- list(utils = reticulate::import_from_path("utils", path = "src"),
                 train = reticulate::import_from_path("train_lstm", path = "src"),
                 test = reticulate::import_from_path("test_lstm", path = "src"))
    if (!grepl("src", mods$utils$`__file__`)) stop("lstm_modules: imported a 'utils' module from outside src/")
    lstm_env$mods <- mods
    lstm_env$pid <- Sys.getpid()
    lstm_env$seed <- NULL
  }
  seed <- as.integer(if (is.null(spec$seed)) 1 else spec$seed)
  if (!identical(lstm_env$seed, seed)) {
    lstm_env$mods$utils$configure(threads = as.integer(if (is.null(spec$threads)) 1 else spec$threads),
                                  seed = seed, hp = spec$hp)
    lstm_env$seed <- seed
  }
  lstm_env$mods
}

# Per-subject arrays (n x T x F, T = K + 1) of the inputs used by all LSTM learners:
# V1 one-hot (3), V2 one-hot (2), V3 (median/IQR scaled), log1p(L1_t), L2_t, L3_t, t / t.end.
# Unobserved values (after censoring or an event) are 0 and are never used for fitting (risk-set masks).
lstm_arrays <- function(dat, K, t.end = 36) {
  n <- nrow(dat)
  T <- K + 1
  num <- function(v) { v <- as.numeric(as.character(v)); v[is.na(v)] <- 0; v }
  V1 <- as.integer(as.character(dat$V1_0)); V2 <- as.integer(as.character(dat$V2_0))
  V3 <- as.numeric(dat$V3_0)
  V3s <- (V3 - median(V3)) / IQR(V3)
  base <- array(0, dim = c(n, T, 10))
  for (k in 0:K) {
    base[, k + 1, ] <- cbind(V1 == 2, V1 == 3, V1 == 4, V2 == 2, V2 == 3, V3s,
                             log1p(num(dat[[paste0("L1_", k)]])), num(dat[[paste0("L2_", k)]]),
                             num(dat[[paste0("L3_", k)]]), k / t.end)
  }
  A <- sapply(0:K, function(k) num(dat[[paste0("A_", k)]]))
  C <- sapply(0:K, function(k) if (k == 0) rep(0, n) else num(dat[[paste0("C_", k)]]))
  at_risk <- sapply(0:K, function(k) as.numeric(ltmle_at_risk(dat, k)))
  list(base = base, A = matrix(A, n), C = matrix(C, n), at_risk = matrix(at_risk, n))
}

# Send one replicate's arrays to Python (once per process and data token).
lstm_set_data <- function(dat, spec, J = 6) {
  mods <- lstm_modules(spec)
  if (!identical(lstm_env$data_token, spec$data_token)) {
    arr <- lstm_arrays(dat, spec$K)
    mods$utils$set_data(spec$data_token, arr$base, arr$A, arr$C, arr$at_risk, as.integer(J))
    lstm_env$data_token <- spec$data_token
  }
  mods
}

# Treatment models g_k(a | history), k = 0..K, from one many-to-many LSTM (multinomial arm) or
# J separate many-to-many sigmoid LSTMs (binomial arm). Same output format as fit_treatment_models().
lstm_fit_treatment <- function(dat, K, arm, J, spec) {
  mods <- lstm_set_data(dat, spec, J)
  res <- mods$train$fit_treatment(arm)
  probs <- res[[1]]
  n <- nrow(dat)
  g <- lapply(0:K, function(k) {
    out <- matrix(NA_real_, n, J, dimnames = list(NULL, 1:J))
    rs <- ltmle_at_risk(dat, k)
    out[rs, ] <- probs[rs, k + 1, ]
    out
  })
  list(g = g, failures = 0, epochs = as.integer(res[[2]]), hp = mods$utils$HP)
}

# Censoring model P(C_k = 0 | history, A_k), k = 1..K, from one many-to-many LSTM (age excluded,
# as in fit_censoring_models()). Same output format as fit_censoring_models().
lstm_fit_censoring <- function(dat, K, J, spec) {
  mods <- lstm_set_data(dat, spec, J)
  res <- mods$train$fit_censoring()
  p <- res[[1]]
  gC <- lapply(seq_len(K), function(k) {
    out <- rep(NA_real_, nrow(dat))
    rs <- ltmle_at_risk(dat, k)
    out[rs] <- p[rs, k + 1]
    out
  })
  list(gC = gC, failures = 0, epochs = as.integer(res[[2]]))
}

# One ICE regression at step s for one recursion (key = arm | recursion | rule | target time):
# fits Z on the last 12 months of history ending at s among the subjects in fit_rows (logical),
# warm-started from the previous step of the same recursion. Returns a predictor
# function(rows (logical), a = NULL) giving Q_s with A_s observed (a = NULL) or set to a.
lstm_outcome_step <- function(dat, s, Z, fit_rows, rule, spec, J = 6) {
  mods <- lstm_set_data(dat, spec, J)
  key <- paste(spec$arm, spec$recursion, rule, spec$target, sep = "|")
  fallback <- paste(spec$arm, "tmle", rule, spec$target, sep = "|")
  idx <- which(fit_rows)
  mods$train$fit_outcome_step(key, fallback, as.integer(s), as.integer(idx - 1L), as.numeric(Z[idx]))
  function(rows, a = NULL) {
    ridx <- which(rows)
    if (length(ridx) == 0) return(numeric(0))
    a_py <- if (is.null(a)) NULL else as.integer(a[ridx])
    as.numeric(mods$test$predict_outcome_step(key, as.integer(s), as.integer(ridx - 1L), a_py))
  }
}

# Legacy interface (sliding windows across subjects, cached predictions); removed in the 2026-10 rebuild.
lstm <- function(data, outcome, covariates, t_end, window_size, out_activation, loss_fn, output_dir, J, ybound, gbound,
                 inference=FALSE, is_censoring=FALSE, debug=TRUE, batch_models=FALSE, batch_rules=NULL) {
  stop("lstm(): the legacy LSTM interface was removed; use simLong(estimator = 'tmle-lstm'), which fits the ",
       "treatment, censoring and outcome LSTMs through lstm_fit_treatment(), lstm_fit_censoring() and lstm_outcome_step()")
}
