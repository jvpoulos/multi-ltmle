######################
# ltmle functions#
######################

static_arip_on <- function(row) {
  #  binary treatment is set to aripiprazole at all time points for all observations
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A1",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  return(shifted)
}

static_halo_on <- function(row) {
  #  binary treatment is set to haloperidol at all time points for all observations
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A2",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  return(shifted)
}

static_olanz_on <- function(row) {
  #  binary treatment is set to olanzapine at all time points for all observations
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A3",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  return(shifted)
}

static_risp_on <- function(row) {
  #  binary treatment is set to risperidone at all time points for all observations
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A5",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  return(shifted)
}

static_mtp <- function(row){
  # Static: Everyone gets olanz. (if bipolar=2), haloperidol (if schizophrenia=3), risp. (if MDD=1) and stays on it
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A1_0",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  shifted <- sapply(0:36, function(t){ #t.end
    if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == 3){ # check if schizophrenia
      shifted[paste0("A2_",t)] <- 1 # switch to halo
    }else if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == 1){
      shifted[paste0("A5_",t)] <- 1 # switch to risp
    }else{
      shifted[paste0("A3_",t)] <- 1 # switch to olanz
    }
    return(shifted)})
  shifted <- rowSums(shifted)
  shifted[shifted>1] <- 1
  return(shifted)
}

dynamic_mtp <- function(row){
  # Dynamic: Start with Arip., then switch to olanz. (bipolar=2), haloperidol (schizophrenia=3), risp (MDD=1) if an antidiabetic drug is filled
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A1_0",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  shifted <- sapply(1:36, function(t){ #t.end
      if(!is.na(row[paste0("L2_","L3_",t)]) & row[paste0("L2_","L3_",t)] == 1){
      if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == 3){ # check if schizophrenia
        shifted[paste0("A2_",t)] <- 1 # switch to halo
      }else if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == q){
        shifted[paste0("A3_",t)] <- 1 # switch to olanz
      }else{
        shifted[paste0("A5_",t)] <- 1 # switch to risp.
      }}else if(!is.na(row[paste0("L3_",t)]) & row[paste0("L3_",t)] == 0){
        shifted[paste0("A1_",t)] <- 1  # otherwise stay on arip
      }
    return(shifted)})
  shifted <- rowSums(shifted)
  shifted[shifted>1] <- 1
  return(shifted)
}

stochastic_mtp <- function(row) {
  # Stochastic: t=0 is same as observed, reduce probability of Arip./Quet./Zipra and increase probability of halo., olanz, risp.
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- treats # do nothing first period
  names(shifted) <- names(treats)
  shifted <- sapply(1:36, function(t){ #t.end
    if(!is.na(row[paste0("C_",t)])){
      random_treat <- Multinom(1, StochasticFun(shifted[grep(paste0("_",t), names(shifted))], d=c(-0.01, 0.01,0.01,-0.01, 0.01,-0.01))[which(shifted[grep(paste0("_",t), names(shifted))]==1),])
      shifted[grep(paste0("_",t), names(shifted))] <- 0
      shifted[grep(paste0("_",t), names(shifted))][as.numeric(random_treat)] <- 1
    }else{
      shifted[grep(paste0("_",t), names(shifted))] <- 0 # skip if censored
    }
    shifted <- sapply(shifted,
                      function(x) as.numeric(as.character(x)))
    return(shifted)})
  shifted <- rowSums(shifted)
  shifted[shifted>1] <- 1
  return(shifted)
}