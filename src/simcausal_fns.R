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

  return(res)
}