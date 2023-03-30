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

dynamic_mtp <- function(row){
  # Dynamic: Start with Arip., then switch to olanz. (bipolar/MDD) or haloperidol (schizophrenia) if an antidiabetic drug is filled
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A1_0",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  shifted <- sapply(2:4, function(t){ #t.end
      if(!is.na(row[paste0("L3_",t)]) & row[paste0("L3_",t)] == 1){
      if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == 3){ # check if schizophrenia
        shifted[paste0("A2_",t)] <- 1 # switch to halo
      }else{
        shifted[paste0("A3_",t)] <- 1 # switch to olanz
      }}else if(!is.na(row[paste0("L3_",t)]) & row[paste0("L3_",t)] == 0){
        shifted[paste0("A1_",t)] <- 1  # otherwise stay on arip
      }
    return(shifted)})
  shifted <- rowSums(shifted)
  shifted[shifted>1] <- 1
  return(shifted)
}

stochastic_mtp <- function(row) {
  # stochastic:  t=1 is same as observed, reduce probability of Arip./Quet./Risp./Zipra and increase probability of halo. and olanz.
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- treats # do nothing first period
  names(shifted) <- names(treats)
  shifted <- sapply(2:4, function(t){ #t.end
    if(!is.na(row[paste0("C_",t)])){
      random_treat <- Multinom(1, StochasticFun(shifted[grep(paste0("_",t), names(shifted))], d=c(-0.01, 0.01,0.01,-0.01,-0.01,-0.01))[which(shifted[grep(paste0("_",t), names(shifted))]==1),])
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