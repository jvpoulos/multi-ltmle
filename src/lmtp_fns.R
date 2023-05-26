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

static_mtp <- function(data, trt) {
  # Static: Everyone gets olanz. (if bipolar=2), haloperidol (if schizophrenia=3), risp. (if MDD=1) and stays on it
  ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 3, # check if schizophrenia
         static_halo_on(data, trt), # switch to halo
         ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 1,
                static_risp_on(data, trt),
         static_olanz_on(data, trt)))
}

dynamic_mtp <- function(data, trt) {
  # Dynamic: Start with Arip., then switch to olanz. (bipolar=2), haloperidol (schizophrenia=3), risp (MDD=1) if an antidiabetic drug is filled
  if (trt == "A_0") {
    static_arip_on(data, trt)
  } else {
    print(data[[sub("A", "L3", trt)]])
    # otherwise check if the time varying covariate equals 1
    ifelse(!is.na(data[[sub("A", "L3", trt)]]) & data[[sub("A", "L3", trt)]] == 1,
           ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 3, 
                  static_halo_on(data, trt), # switch to halo
                  ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 2, # switch to olanz
                  static_olanz_on(data, trt), static_risp_on(data, trt))),  # switch to risp
           static_arip_on(data, trt))  # otherwise stay on arip
  }
}

stochastic_mtp <- function(data, trt) {
  # Stochastic: t=0 is same as observed, reduce probability of Arip./Quet./Zipra and increase probability of halo., olanz, risp.
  if (trt == "A_0") {
    data[[trt]] # do nothing
  }else{
    Multinom(length(data[[trt]]), StochasticFun(data[[trt]], d=c(-0.01, 0.01,0.01,-0.01, 0.01,-0.01)))
  }
}