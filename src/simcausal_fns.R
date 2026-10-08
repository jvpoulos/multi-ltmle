#######################
# simcausal functions #
#######################

# definition: left-truncated normal distribution

RnormTrunc <- function(n, mean, sd, minval = 0, maxval = 10000,
                       min.low = 0, max.low = 50, min.high = 5000, max.high = 10000)
{                                                                                      
  out <- rnorm(n = n, mean = mean, sd = sd)
  minval <- minval[1]; min1 <- min.low[1]; max1 <- max.low[1]
  maxval <- maxval[1]; min2 <- min.high[1]; max2 <- max.high[1]
  leq.zero <- length(out[out <= minval])
  geq.max <- length(out[out >= maxval])
  out[out <= minval] <- runif(n = leq.zero, min = min1, max = max1)
  out[out >= maxval] <- runif(n = geq.max, min = min2, max = max2)
  out
}

# definition: negative binomial
NegBinom <- function(n, mu){                                                                                      
  rnbinom(n=n, size=1, mu=mu)
}

rbern <- function(n, prob) {
  rbinom(n=n, prob=prob, size=1)
}

# definition: multinomial
Multinom <- function(n,probs){
  # probs is vector or matrix
  # combination of:
  # rcat.b1 from simcausal
  # rmult from glmnet
  if (!is.na(n) && n == 0) {
    probs <- matrix(nrow = n, ncol = length(probs), byrow = TRUE)
  }
  if (is.vector(probs) && !is.na(n) && n > 0) {
    probs <- matrix(data = probs, nrow = n, ncol = length(probs), 
                    byrow = TRUE)
  }
  x <- t(apply(probs, 1, function(x) rmultinom(1, 1, x)))
  x <- x %*% seq(ncol(probs))
  return(as.factor(drop(x)))
}

# stochastic treatment rule function
StochasticFun <- function(condition, d, stay_prob=0.95) {
  condition[is.na(condition)] <- 0
  n <- length(condition)
  if (!is.matrix(d)) {
    d <- matrix(d, n, length(d), byrow = TRUE)
  }
  J <- ncol(d)
  
  # Create a matrix with higher stay probability and lower switch probability
  switch_prob <- (1 - stay_prob) / (J - 1)  # Adjusted switch probability
  
  # Create a matrix for the probabilities of staying vs. switching
  probs_matrix <- matrix(switch_prob, n, J, byrow = TRUE)
  for (j in 1:J) {
    probs_matrix[, j] <- ifelse(condition == j, stay_prob, switch_prob)
  }
  
  # Apply the stochastic adjustments d
  res <- probs_matrix + d
  
  # Ensure that no probabilities exceed 1 after the adjustment
  res[res > 1] <- 1
  
  # Replace any resulting NAs with a very small number
  if (any(is.na(res))) {
    res[is.na(res)] <- .Machine$double.eps
  }
  
  return(res)
}
###################################################################
# Shared treatment rules (static, dynamic, stochastic)            #
# used by the truth, the rule-following indicators, and the       #
# estimators. Treatments: 1 ari, 2 halo, 3 olan, 4 quet, 5 risp,  #
# 6 zipr. Diagnosis V2: 1 MDD, 2 bipolar, 3 schizophrenia.        #
###################################################################

# condition-specific drug: MDD -> ari, bipolar -> quet, schizophrenia -> halo
rule_condition_drug <- function(V2) {
  V2 <- as.integer(as.character(V2))
  ifelse(V2 == 3, 2L, ifelse(V2 == 2, 4L, 1L))
}

# treatment assigned by a deterministic rule at time t given (observed or counterfactual) history
rule_assign <- function(rule, t, V2, L1 = NULL, L2 = NULL, L3 = NULL) {
  if (rule == "static") {
    return(rule_condition_drug(V2))
  }
  if (rule == "dynamic") {
    if (t == 0) return(rep(5L, length(V2)))
    return(ifelse(L1 > 0 | L2 > 0 | L3 > 0, rule_condition_drug(V2), 5L))
  }
  stop("rule_assign: '", rule, "' is not a deterministic rule")
}

# intervention density g*(a_t | a_{t-1}) of the stochastic rule for t > 0
# (A_0 follows its natural distribution under this rule)
rule_stochastic_density <- function(a, a_prev, stay_prob = 0.95, J = 6) {
  ifelse(a == a_prev, stay_prob, (1 - stay_prob) / (J - 1))
}

# simcausal action nodes implementing the three rules under no censoring
rule_interventions <- function(t.end) {
  list(
    static = c(node("A", t = 0:t.end, distr = "rconst",
                    const = ifelse(V2[0]==3, 2, ifelse(V2[0]==1, 1, 4))),
               node("C", t = 1:t.end, distr = "rbern", prob = 0)),
    dynamic = c(node("A", t = 0, distr = "rconst", const = 5),
                node("A", t = 1:t.end, distr = "rconst",
                     const = ifelse((L1[t] > 0 | L2[t] > 0 | L3[t] > 0), ifelse(V2[0]==3, 2, ifelse(V2[0]==2, 4, 1)), 5)),
                node("C", t = 1:t.end, distr = "rbern", prob = 0)),
    stochastic = c(node("A", t = 1:t.end, distr = "Multinom",
                        probs = StochasticFun(A[(t-1)], d = c(0, 0, 0, 0, 0, 0))),
                   node("C", t = 1:t.end, distr = "rbern", prob = 0))
  )
}

# True counterfactual cumulative incidence E[Y_t^d], t = 1..t.end, for each rule, from a large
# Monte Carlo sample (n_truth subjects per seed) computed once and cached on disk.
compute_truth <- function(Dset, t.end, n_truth = 1e5, n_seeds = 5, cache_dir = "./data", seed_offset = 1e6,
                          dgp_file = "./src/simcausal_dgp.R") {
  cache_file <- file.path(cache_dir, paste0("truth_t_", t.end, "_n_", n_truth, "_seeds_", n_seeds, ".rds"))
  dgp_md5 <- unname(tools::md5sum(dgp_file))
  if (file.exists(cache_file)) {
    cached <- readRDS(cache_file)
    if (identical(cached$dgp_md5, dgp_md5)) return(cached)
    message("compute_truth: cached truth was computed under a different DGP; recomputing")
  }
  if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)
  interventions <- rule_interventions(t.end)
  rules <- names(interventions)
  draws <- array(NA_real_, dim = c(n_seeds, t.end, length(rules)), dimnames = list(NULL, paste0("t=", 1:t.end), rules))
  for (rule in rules) {
    D_rule <- Dset + action("rule", nodes = interventions[[rule]])
    D_rule <- set.targetE(D_rule, outcome = "Y", t = 1:t.end, param = "rule")
    for (s in seq_len(n_seeds)) {
      cf <- sim(DAG = D_rule, actions = "rule", n = n_truth, LTCF = "Y", rndseed = seed_offset + s, verbose = FALSE)
      draws[s, , rule] <- eval.target(D_rule, data = cf)$res
      if (s == 1 && rule != "stochastic") {
        # check that the simcausal intervention matches the shared rule implementation
        dat <- cf[["rule"]]
        for (k in 0:t.end) {
          at_risk <- if (k == 0) rep(TRUE, nrow(dat)) else dat[[paste0("Y_", k - 1)]] == 0
          d_k <- rule_assign(rule, k, dat$V2_0, dat[[paste0("L1_", k)]], dat[[paste0("L2_", k)]], dat[[paste0("L3_", k)]])
          mismatch <- sum(as.integer(as.character(dat[[paste0("A_", k)]]))[at_risk] != d_k[at_risk])
          if (mismatch > 0) stop("compute_truth: simcausal '", rule, "' intervention differs from rule_assign() at t=", k)
        }
      }
      rm(cf); gc()
    }
  }
  truth <- apply(draws, c(2, 3), mean)
  truth_mc_se <- apply(draws, c(2, 3), sd) / sqrt(n_seeds)
  out <- list(truth = truth, mc_se = truth_mc_se, n_truth = n_truth, n_seeds = n_seeds, dgp_md5 = dgp_md5)
  saveRDS(out, cache_file)
  out
}
