######################
# simcausal functions#
######################

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

# # dynamic treatment rule function
DynamicFun <- function(condition1, condition2, cat, pstart, pswitch_a, pswitch_b){ 
  # condition 1: time-varying covariate that determines switch (e.g., antidiabetic prescription fill)
  # condition 2: baseline covariate that determines which drug to switch to (e.g., schizophrenia diagnosis)
  # cat: category level for condition 2 (e.g., 3= schizophrenia) 
  # pstart: starting multinomial probs.
  # pswitch_a, pswitch_b: switching multinomial probs. determined by condition2
  condition1[is.na(condition1)] <- 0
  condition2[is.na(condition2)] <- 0
  condition2 <- ifelse(condition2==cat,1,0)
  n <- length(condition1)
  J <- ncol(pstart)
  res <- (matrix(condition1,n,J)*matrix(condition2,n,J)*pswitch_a + matrix(condition1,n,J)*(1-matrix(condition2,n,J))*pswitch_a) + (1-matrix(condition1,n,J))*pstart
  return(res)
}

# stochastic treatment rule function
StochasticFun <- function(condition, d) { # Stochastic: reduce probability of olanzapine and increase probability of aripiprazole and haloperidol
  condition[is.na(condition)] <- 0
  n <- length(condition)
  if(!is.matrix(d)){
    d <- matrix(d,n,length(d),byrow = TRUE)
  }
  J <- ncol(d)
 res <- as.matrix((matrix(condition,n,J)==1)+0)*matrix(c(0.75,0.05,0.05,0.05,0.05,0.05),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==2)+0)*matrix(c(0.05,0.75,0.05,0.05,0.05,0.05),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==3)+0)*matrix(c(0.05,0.05,0.75,0.05,0.05,0.05),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==4)+0)*matrix(c(0.05,0.05,0.05,0.75,0.05,0.05),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==5)+0)*matrix(c(0.05,0.05,0.05,0.05,0.75,0.05),n,J,byrow = TRUE) +
    as.matrix((matrix(condition,n,J)==6)+0)*matrix(c(0.05,0.05,0.05,0.05,0.05,0.75),n,J,byrow = TRUE) + d
  if(any(is.na(res))){
    res[is.na(res)] <- .Machine$double.eps # placeholder 
  }
  return(res)
}