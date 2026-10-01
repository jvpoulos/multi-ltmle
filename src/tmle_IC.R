###################################################################
# Influence-curve-based inference for the LTMLE and IPTW          #
# estimates (rebuilt 2026-10).                                    #
# Subjects are i.i.d., so the variance of each estimator is the   #
# empirical variance of its per-subject influence curve:          #
# SE = sd(IC_i) / sqrt(n); 95% CI = estimate +/- 1.96 SE.         #
# G-computation with data-adaptive regressions has no valid       #
# analytic IC; its SE uses the IC evaluated at the untargeted     #
# fits, which ignores plug-in bias (approximate, anti-conservative).#
###################################################################

# tmle_contrasts: list over target times (named "t=<t>") of getTMLELong() outputs, i.e. for each rule
#   list(psi, ic, psi_iptw, ic_iptw, psi_gcomp). Legacy inputs (elements with $Qstar matrices, as produced
#   by the tmle-lstm path) return point estimates only, with NA standard errors.
# Returns list(est, se = matrices [target time x rule], CI = list over times of 2 x rule matrices).
TMLE_IC <- function(tmle_contrasts, initial_model_for_Y, time.censored=NULL, iptw=FALSE, gcomp=FALSE,
                    estimator="tmle", basic_only=FALSE, variance_estimates=NULL, diagnostics=FALSE) {
  if (is.null(tmle_contrasts) || length(tmle_contrasts) == 0) {
    stop("TMLE_IC: tmle_contrasts is empty")
  }
  rules <- c("static", "dynamic", "stochastic")
  times <- names(tmle_contrasts)
  if (is.null(times)) times <- paste0("t=", seq_along(tmle_contrasts))
  est <- matrix(NA_real_, length(tmle_contrasts), length(rules), dimnames = list(times, rules))
  se <- est

  for (i in seq_along(tmle_contrasts)) {
    x <- tmle_contrasts[[i]]
    if (is.null(x)) next
    if (!is.null(x$Qstar) || !is.null(x$Qstar_iptw) || !is.null(x$Qstar_gcomp)) {
      # legacy (tmle-lstm) format: no influence curve available, so no valid standard error
      Q <- if (gcomp) x$Qstar_gcomp else if (iptw) x$Qstar_iptw else x$Qstar
      if (!is.null(Q)) est[i, seq_len(min(ncol(Q), 3))] <- colMeans(Q, na.rm = TRUE)[seq_len(min(ncol(Q), 3))]
      next
    }
    for (rule in intersect(rules, names(x))) {
      r <- x[[rule]]
      if (gcomp) {
        est[i, rule] <- r$psi_gcomp
        if (!is.null(r$ic_gcomp) && !anyNA(r$ic_gcomp)) se[i, rule] <- sd(r$ic_gcomp) / sqrt(length(r$ic_gcomp))
      } else if (iptw) {
        est[i, rule] <- r$psi_iptw
        if (!anyNA(r$ic_iptw)) se[i, rule] <- sd(r$ic_iptw) / sqrt(length(r$ic_iptw))
      } else {
        est[i, rule] <- r$psi
        if (!anyNA(r$ic)) se[i, rule] <- sd(r$ic) / sqrt(length(r$ic))
      }
    }
  }
  if (all(is.na(se)) && !gcomp) {
    warning("TMLE_IC: no influence curves supplied; standard errors are NA")
  }
  CI <- lapply(seq_len(nrow(est)), function(i) {
    rbind(lower = est[i, ] - qnorm(0.975) * se[i, ], upper = est[i, ] + qnorm(0.975) * se[i, ])
  })
  names(CI) <- times
  list(est = est, se = se, CI = CI)
}
