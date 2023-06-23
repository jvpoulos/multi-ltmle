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

static_quet_on <- function(row) {
  #  binary treatment is set to quetiapine at all time points for all observations
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A4",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  return(shifted)
}

static_mtp <- function(row){
  # Static: Everyone gets quetiap (if bipolar=2), halo (if schizophrenia=3), ari (if MDD=1) and stays on it
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A1_0",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  shifted <- sapply(0:36, function(t){ #t.end
    if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == 3){ # check if schizophrenia
      shifted[paste0("A2_",t)] <- 1 # switch to halo
    }else if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == 1){ # check if MDD
      shifted[paste0("A1_",t)] <- 1 # switch to ari
    }else{ # else bipolar
      shifted[paste0("A4_",t)] <- 1 # switch to quet
    }
    return(shifted)})
  shifted <- rowSums(shifted)
  shifted[shifted>1] <- 1
  return(shifted)
}

dynamic_mtp <- function(row){
  # Dynamic: Everyone starts with quetiap.
  # If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to risp (if bipolar), halo. (if schizophrenia), ari (if MDD)
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- ifelse(names(treats)%in%grep("A4_0",colnames(row), value=TRUE),1,0)
  names(shifted) <- names(treats)
  shifted <- sapply(1:36, function(t){ #t.end
      if(!is.na(row[paste0("L1_", "L2_","L3_",t)]) & any(row[paste0("L1_","L2_","L3_",t)] >0)){
      if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == 3){ # check if schizophrenia
        shifted[paste0("A2_",t)] <- 1 # switch to halo
      }else if(!is.na(row[["V2_0"]]) & row[["V2_0"]] == 2){ # check if bipolar
        shifted[paste0("A5_",t)] <- 1 # switch to risp
      }else{
        shifted[paste0("A1_",t)] <- 1 # switch to risp. # switch to ari if MDD
      }}else if(!is.na(row[paste0("L1_","L2_","L3_",t)]) & row[paste0("L1_","L2_","L3_",t)] == 0){
        shifted[paste0("A4_",t)] <- 1  # otherwise stay on quet.
      }
    return(shifted)})
  shifted <- rowSums(shifted)
  shifted[shifted>1] <- 1
  return(shifted)
}

stochastic_mtp <- function(row) {
  # Stochastic: at each t>0, 95% chance of staying with treatment at t-1, 5% chance of randomly switching according to Multinomial distibution
  treats <- row[grep("A[0-9]",colnames(row), value=TRUE)]
  shifted <- treats # do nothing first period
  names(shifted) <- names(treats)
  shifted <- sapply(1:36, function(t){ #t.end
    if(!is.na(row[paste0("C_",t)])){
      random_treat <- Multinom(1, StochasticFun(shifted[grep(paste0("_",t), names(shifted))], d=c(0,0,0,0,0,0))[which(shifted[grep(paste0("_",t), names(shifted))]==1),])
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