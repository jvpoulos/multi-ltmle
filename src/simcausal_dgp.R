#####################################
# Define data generating process #
#####################################

# Initialize the DAG
# Try to register vectorized functions if the version of simcausal supports it
D <- tryCatch({
  # First attempt with vecfun parameter (newer versions)
  DAG.empty(vecfun = c("StochasticFun", "SimpleTransition"))
}, error = function(e) {
  # Fallback for older versions that don't support vecfun parameter
  cat("Using alternative approach to register vectorized functions\n")
  # Create basic DAG
  dag <- DAG.empty()
  # Try to set the vectorized functions using options
  options(simcausal.vecfun = c("StochasticFun", "SimpleTransition"))
  # Return the DAG
  dag
})

# Ensure SimpleTransition is registered as a vectorized function
try({
  # Check if SimpleTransition is already registered
  curr_vecfun <- getOption("simcausal.vecfun")
  if (!("SimpleTransition" %in% curr_vecfun)) {
    # Add SimpleTransition to the registered vectorized functions
    new_vecfun <- unique(c(curr_vecfun, "SimpleTransition"))
    options(simcausal.vecfun = new_vecfun)
    cat("Updated vectorized functions:", paste(new_vecfun, collapse=","), "\n")
  }
})

# Store original t.end for use later if needed
original_t_end <- t.end

# baseline data (t = 0) and follow-up data (t = 1, . . . , T)  created using structural equations

# distributions at baseline (t=0): no intervention
D.base <- D +
  node("V1",                                      # race -> 1 = "white", 2  = "black", 3  = "latino", 4  = "other"
       t = 0,
       distr = "rcat.factor",
       probs = c(25349/38762,6649/38762,4187/38762)) +
  node("V2",                                      # smi_condition -> 1 = "mdd", 2 = "bipolar", 3 ="schiz" (varies by race)
       t = 0,
       distr = "rcat.factor",
       probs = c(ifelse(V1[0] == 1, (7740-1000)/38762, ifelse(V1[0] == 2, (7740+1000)/38762, (7740+500)/38762)),
                 ifelse(V1[0] == 1, (9721-2000)/38762, ifelse(V1[0] == 2, (9721+2000)/38762, (9721+1000)/38762)))) +
  node("V3",                                      # age (continuous) (varies by  smi condition)
       t = 0,
       distr = "RnormTrunc",
       mean = ifelse(V2[0]==3, 46.5, ifelse(V2[0]==2 ,44.5, 44)), 
       sd = ifelse(V2[0]==3, 9.2, ifelse(V2[0]==2, 10.2, 10.5)),
       minval = 19.9, maxval = 64.5,
       min.low = 19.9, max.low = 36.79, min.high = 53.27, max.high = 64.5) +
  node("L1",                                      # er_mhsa (count) (varies by smi condition)
       t = 0,
       distr = "NegBinom",
       mu = ifelse(V2[0] == 3, 0.09, ifelse(V2[0] == 2, 0.05, 0.03)))  +
  node("L2",                                      # ever_mt_gluc_or_lip (binary) (varies by smi condition)
       t = 0,
       distr = "rbern",
       prob = ifelse(V2[0] == 3, 0.075, ifelse(V2[0] == 2, 0.05, 0.025))) + 
  node("L3",                                      # ever_rx_antidiab (binary) (varies by smi condition)
       t = 0,
       distr = "rbern",
       prob = ifelse(V2[0] == 3, 0.085, ifelse(V2[0] == 2, 0.015, 0.035))) + 
  node("A",          # drug_group --> ARIPIPRAZOLE; HALOPERIDOL; OLANZAPINE; QUETIAPINE; RISPERIDONE; ZIPRASIDONE (varies by smi condition and antidiab rx)
       t = 0,
       distr = "Multinom",
       probs =  cbind(
         ifelse(V2[0]==1 & (L3[0]>0), 1/4, 1/8),
         ifelse(V2[0]==3 & (L1[0]>0), 1/4, 1/8),
         1/8,
         ifelse(V2[0]==2 & (L2[0]>0), 1/4, 1/8),
         ifelse(V2[0]==2 & (L1[0]>0 | L2[0]>0 | L3[0]>0), 1/4, 1/8),
         1/8
       )) +
  node("C",                                     # monthly_censored_indicator (no censoring at baseline)
       t = 0,
       distr = "rbern",
       prob = 0,
       EFU = TRUE) +
  node("Y",                                      # diabetes
       t = 0,
       distr = "rbern",
       prob= 0,
       EFU = TRUE) 

# distributions at later time-points (t = 1, . . . , T)
D <- D.base +
  node("L1",                                      # er_mhsa (count)
       t = 1:t.end,
       distr = "NegBinom",
       mu= plogis(.05 *L1[t-1]**2 + .05 * L2[t-1] + .1 * L3[t-1] + ifelse(A[(t-1)]==1 | A[(t-1)]==2 | A[(t-1)]==4, -1, ifelse(A[(t-1)]==5, -5, 0)))) + 
  node("L2",                                      # ever_mt_gluc_or_lip (binary)
       t = 1:t.end,
       distr = "rbern",
       prob= plogis(-2 + .05 * (L1[t] - L1[t-1])**2 + .1 * L3[t-1] + .1 * L2[t-1] + ifelse(A[(t-1)]==1 | A[(t-1)]==2 | A[(t-1)]==4, -4, ifelse(A[(t-1)]==5, -5, 0)))) +
  node("L3",                                      # ever_rx_antidiab (binary)
       t = 1:t.end,
       distr = "rbern",
       prob= plogis(-2 + .05 * (L1[t] - L1[t-1])**2 + 0.1 * L2[t-1] + 0.1 * L3[t-1] + ifelse(A[(t-1)]==1 | A[(t-1)]==2 | A[(t-1)]==4, -4, ifelse(A[(t-1)]==5, -5, 0)))) +
  node("A",          # drug_group --> ARIPIPRAZOLE; HALOPERIDOL; OLANZAPINE; QUETIAPINE; RISPERIDONE; ZIPRASIDONE
       t = 1:t.end, 
       distr = "Multinom",
       # Same probability model but with flatter structures to reduce stack usage
       # Preserves identical DGP probabilities from the original model
       probs = cbind(
         # Matrix representation is more efficient for stack usage
         # Each column represents one treatment probability:
         
         # 1. ARIPIPRAZOLE: base 0.01 + stay 0.94 + boost 0.01 if any L positive
         0.01 + 0.94 * (A[(t-1)]==1) + 0.01 * ((L1[t]>0) | (L2[t]>0) | (L3[t]>0)),
         
         # 2. HALOPERIDOL: base 0.01 + stay 0.94 + boost 0.01 if L1 positive  
         0.01 + 0.94 * (A[(t-1)]==2) + 0.01 * (L1[t]>0),
         
         # 3. OLANZAPINE: base 0.01 + stay 0.94 (no boost)
         0.01 + 0.94 * (A[(t-1)]==3),
         
         # 4. QUETIAPINE: base 0.01 + stay 0.94 + boost 0.01 if L2 positive
         0.01 + 0.94 * (A[(t-1)]==4) + 0.01 * (L2[t]>0),
         
         # 5. RISPERIDONE: base 0.01 + stay 0.94 + boost 0.01 if L3 positive
         0.01 + 0.94 * (A[(t-1)]==5) + 0.01 * (L3[t]>0),
         
         # 6. ZIPRASIDONE: base 0.01 + stay 0.94 (no boost)
         0.01 + 0.94 * (A[(t-1)]==6)
       )) +
  node("C",                                      # monthly_censored_indicator
       t = 1:t.end,
       distr = "rbern",
       prob =ifelse((V3[0]+(t/12))>65,1, plogis(-4 + .1 * (L1[t] - L1[t-1])**2 + 0.1 *L2[t-1] + 0.1 *L2[t]  + 0.1 * L3[t-1] + 0.1 * L3[t] + ifelse(A[(t-1)]==1 | A[(t-1)]==2 | A[(t-1)]==4, -0.5, ifelse(A[(t-1)]==5, -0.25, 0)) + ifelse(A[t]==1 | A[t]==2 | A[t]==4, -1, ifelse(A[(t-1)]==5, -2, 0)))), # deterministic: AGE out at 65 (medicaid -> medicare)
       EFU = TRUE) + # right-censoring (EFU) 
  node("Y",                                      # diabetes
       t = 1:t.end,
       distr = "rbern",
       prob = plogis(-4 + Y[t-1] + .1 * (L1[t] - L1[t-1])**2 + 0.1 *L2[t-1] + 0.1 *L2[t]  + 0.1 * L3[t-1] + 0.1 * L3[t] + ifelse(A[(t-1)]==1 | A[(t-1)]==2 | A[(t-1)]==4, -0.5, ifelse(A[(t-1)]==5, -0.25, 0)) + ifelse(A[t]==1 | A[t]==2 | A[t]==4, -1, ifelse(A[(t-1)]==5, -2, 0))),
       EFU = TRUE)
