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
StochasticFun <- function(condition, d) {
  condition[is.na(condition)] <- 0
  n <- length(condition)
  if(!is.matrix(d)){
    d <- matrix(d,n,length(d),byrow = TRUE)
  }
  J <- ncol(d)
 res <- as.matrix((matrix(condition,n,J)==1)+0)*matrix(c(0.95,0.01,0.01,0.01,0.01,0.01),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==2)+0)*matrix(c(0.01,0.95,0.01,0.01,0.01,0.01),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==3)+0)*matrix(c(0.01,0.01,0.95,0.01,0.01,0.01),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==4)+0)*matrix(c(0.01,0.01,0.01,0.95,0.01,0.01),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==5)+0)*matrix(c(0.01,0.01,0.01,0.01,0.95,0.01),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==6)+0)*matrix(c(0.01,0.01,0.01,0.01,0.01,0.95),n,J,byrow = TRUE) + d
  if(any(is.na(res))){
    res[is.na(res)] <- .Machine$double.eps # placeholder 
  }
  return(res)
}