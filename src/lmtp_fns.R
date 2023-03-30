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

dynamic_mtp <- function(data, trt) {
  # Dynamic: Start with Arip., then switch to olanz. (bipolar/MDD) or haloperidol (schizophrenia) if an antidiabetic drug is filled
  if (trt == "A_1") {
    # if its the first time point, start with olanzapine
    static_arip_on(data, trt)
  } else {
    # otherwise check if the time varying covariate equals 1
    ifelse(!is.na(data[[sub("A", "L3", trt)]]) & data[[sub("A", "L3", trt)]] == 1,
           ifelse(!is.na(data[["V2_0"]]) & data[["V2_0"]] == 3, # check if schizophrenia
                  static_halo_on(data, trt), # switch to halo
                  static_olanz_on(data, trt)), # switch to olanz
           static_arip_on(data, trt))  # otherwise stay on arip
  }
}

stochastic_mtp <- function(data, trt) {
  # stochastic:  t=1 is same as observed, reduce probability of Arip./Quet./Risp./Zipra and increase probability of halo. and olanz.
  if (trt == "A_1") {
    data[[trt]] # do nothing
  } else if (trt %in% paste0("A_", seq(2,4))){ #t.end
    Multinom(length(data[[trt]]), StochasticFun(data[[trt]], d=c(-0.01, 0.01,0.01,-0.01,-0.01,-0.01)))
  }
}