######################
# lmtp functions#
######################

static_arip_on <- function(data, trt) {
  #  multinomial treatment is set to 1 (aripiprazole) at all time points for all observations
  return(factor(rep(1, length(data[[trt]])), levels=1:6))
}

static_halo_on <- function(data, trt) {
  #  multinomial treatment is set to 2 (haloperidol) at all time points for all observations
  return(factor(rep(2, length(data[[trt]])), levels=1:6))
}

static_olanz_on <- function(data, trt) {
  #  multinomial treatment is set to 3 (olanzapine) at all time points for all observations
  return(factor(rep(3, length(data[[trt]])), levels=1:6))
}

static_risp_on <- function(data, trt) {
  #  multinomial treatment is set to 5 (risperidone) at all time points for all observations
  return(factor(rep(5, length(data[[trt]])), levels=1:6))
}

static_quet_on <- function(data, trt) {
  #  multinomial treatment is set to 5 (risperidone) at all time points for all observations
  return(factor(rep(4, length(data[[trt]])), levels=1:6))
}

static_mtp <- function(data, trt) {
  # Static: Everyone gets quetiap (if bipolar=2), halo (if schizophrenia=3), ari (if MDD=1) and stays on it
  ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 3, # check if schizophrenia
         static_halo_on(data, trt), # switch to halo
         ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 1, # check if MDD
                static_arip_on(data, trt), # switch to ari
         static_quet_on(data, trt))) # else bipolar, and switch to quet
}

dynamic_mtp <- function(data, trt) {
  # Dynamic: Everyone starts with risp.
  # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to quetiap. (if bipolar), halo. (if schizophrenia), ari (if MDD); otherwise stay on risp.
  if (trt == "A_0") {
    static_risp_on(data, trt)
  } else {
    # otherwise check if the time varying covariate is positive and nonzero
    ifelse((!is.na(data[[sub("A", "L1", trt)]]) & !is.na(data[[sub("A", "L2", trt)]]) & !is.na(data[[sub("A", "L3", trt)]])) & 
             (data[[sub("A", "L1", trt)]] >0 | data[[sub("A", "L2", trt)]] >0 | data[[sub("A", "L3", trt)]] >0),
           ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 3, # check if schiz.
                  static_halo_on(data, trt), # switch to halo
                  ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 2, # check if bipolar
                  static_quet_on(data, trt), static_arip_on(data, trt))), # switch to quet, or ari if MDD
           static_risp_on(data, trt))  # otherwise stay on risp.
  }
}

stochastic_mtp <- function(data, trt) {
  # Stochastic: at each t>0, 95% chance of staying with treatment at t-1, 5% chance of randomly switching according to Multinomial distibution
  if (trt == "A_0") {
    data[[trt]] # do nothing
  }else{
    Multinom(length(data[[trt]]), StochasticFun(data[[trt]], d=c(0,0,0,0,0,0)))
  }
}