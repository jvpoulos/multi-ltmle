#####################################
# Define data generating process #
#####################################

# initialize the DAG
D <- DAG.empty()

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
       probs =  c(ifelse(V2[0]==1 & (L3[0]>0), 1/4, 1/8), ifelse(V2[0]==3 & (L1[0]>0), 1/4, 1/8), 1/8, ifelse(V2[0]==2 & (L2[0]>0), 1/4, 1/8), ifelse(V2[0]==2 & (L1[0]>0 | L2[0]>0 | L3[0]>0), 1/4, 1/8), 1/8)) + 
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
       probs = StochasticFun(A[(t-1)], d=c(ifelse(L1[t]>0 | L2[t]>0 | L3[t]>0, 0.01, 0), ifelse(L1[t]>0, 0.01, 0), 0, ifelse(L2[t]>0, 0.01, 0), ifelse(L3[t]>0, 0.01, 0), 0), stay_prob=0.95)) +
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