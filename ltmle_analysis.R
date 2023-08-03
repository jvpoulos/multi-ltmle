##################################
# Analysis with LTMLE            #
##################################

library(purrr)
library(SuperLearner)
library(origami)
library(weights)
library(dplyr)
library(reporttools)
library(ggplot2)
library(reshape2)
library(rpart)
library(ranger)
library(glmnet)
library(grid)
library(sl3)
options(sl3.verbose = FALSE)
library(car)
library(data.table) 
library(gtools)
library(tidyverse)

# set up parallel
library(parallel)
library(doParallel)
library(foreach)

######################
# Setup              #
######################

cores <- parallel::detectCores()
print(paste0("number of cores used: ", cores))

cl <- parallel::makeCluster(cores, outfile="")

doParallel::registerDoParallel(cl) # register cluster

# command line args
args <- commandArgs(trailingOnly = TRUE) # command line arguments 
# args <- c('tmle','all','none','TRUE','TRUE')
estimator <- as.character(args[1])
treatment.rule <- ifelse(estimator=="tmle", "all", as.character(args[2])) # tmle calculates calculates counterfactual means under all treatment rules
weights.loc <- as.character(args[3])
use.SL <- as.logical(args[4])  # When TRUE, use Super Learner for initial Y model and treatment model estimation; if FALSE, use GLM
use.simulated <- as.logical(args[5])  # When TRUE, use simulated data; if FALSE, use real data.

scale.continuous <- FALSE # standardize continuous covariates

gbound <- c(0.04,1) # define bounds to be used for the propensity score

ybound <- c(0.0001,0.9999) # define bounds to be used for the Y predictions

n.folds <- 5 # number of folds for SL

t.end <- 36 

J <- 6

# implement checks

if(J!=6){
  stop("J must be 6")
}

if(t.end<4 && t.end >36){
  stop("t.end must be at least 4 and no more than 36")
}

if(t.end!=36){
  stop("need to manually change  the number of lags in tmle_dat, and IC in tmle_fns")
}

if(n.folds<3){
  stop("n.folds needs to be greater than 3")
}

if(use.SL==FALSE){
  warning("not tested on use.SL=FALSE")
}

if(scale.continuous==TRUE){
  warning("real values of certain time-varying covariates needed to characterize treatment rules")
}

if(estimator=='tmle-lstm'){
  stop("tmle-lstm not functional")
}

# output directory
output_dir <- './outputs/'
simulation_version <- ifelse(weights.loc=='none', paste0(format(Sys.time(), "%Y%m%d"),"/"), weights.loc)
if(!dir.exists(output_dir)){
  print(paste0('create folder for outputs at: ', output_dir))
  
  dir.create(output_dir)
}
output_dir = paste0(output_dir, simulation_version)
if(!dir.exists(output_dir)){
  print(paste0('create folder for outputs at: ', output_dir))
  dir.create(output_dir)
}

filename <- paste0(output_dir, 
                   "longitudinal_results_",
                   "estimator_",estimator,
                   "_treatment_rule_",treatment.rule,
                   "_n_folds_",n.folds,
                   "_scale_continuous_",scale.continuous,
                   "_use_SL_", use.SL,".rds")

if(estimator%in%c("tmle", "tmle-lstm")){
  source('./src/misc_fns.R')
  source('./src/tmle_fns.R')
  source('./src/SL3_fns.R', local =TRUE)
}

if(estimator=='tmle-lstm'){
  library(tensorflow)
  library(keras)
  print(is_keras_available())
  print(tf_version())
}

# load utils
source('./src/simcausal_fns.R')

#####################################
# Load data #
#####################################

if(use.simulated){
  load("simdata_from_basevars.RData")
  source("add_tv_simulated.R") # add time-varying variables
  
  Odat <- simdata_from_basevars
  Odat <- Odat[Odat$hcp_patient_id%in%c(1:9000),] # select first 9k simulated IDs to improve run-time
  
  rm(simdata_from_basevars)
  data.label <- "(simulated CMS data)"
}else{
  load("/data/MedicaidAP_associate/poulos/fup3yr_episode_months_deid_fixmonthout.RData") # run on Argos
  paste0("Original data dimension: ", dim(fup3yr_episode_months_deid))
  
  Odat <- fup3yr_episode_months_deid
  
  rm(fup3yr_episode_months_deid)
  
  data.label <- "(CMS data)"
}

# make summary tables and plots

source("./ltmle_analysis_eda.R")

# make data balanced 38762*37 = 1434194

# Identify unique combinations of hcp_patient_id and month_number
unique_combinations <- Odat %>%
  select(hcp_patient_id, month_number) %>%
  distinct()

# Create a dataframe of all combinations of unique hcp_patient_id and month_number
complete_combinations <- expand.grid(hcp_patient_id = unique(unique_combinations$hcp_patient_id),
                                     month_number = unique(unique_combinations$month_number))

# Merge the complete_combinations dataframe with the original dataframe
balanced_panel <- merge(complete_combinations, Odat, 
                        by = c("hcp_patient_id", "month_number"), 
                        all.x = TRUE)

# Remove unwanted columns
balanced_panel <- balanced_panel %>%
  select(-c("death_36", "diabetes_36"))

# baseline covariates

L <- cbind(dummify(factor(Odat$state_character),show.na = FALSE),
           dummify(factor(Odat$race_defined_character),show.na = FALSE), 
           dummify(factor(Odat$smi_condition)), 
           "year"=Odat$year, 
           "female"=Odat$female, 
           "payer_index_mdcr"=dummify(factor(Odat$payer_index))[,2],
           "preperiod_ever_psych"=Odat$preperiod_ever_psych,
           "preperiod_ever_metabolic"=Odat$preperiod_ever_metabolic,
           "preperiod_ever_other"=Odat$preperiod_ever_other,
           "preperiod_drug_use_days"=Odat$preperiod_drug_use_days,
           "preperiod_ever_mt_gluc_or_lip"=Odat$preperiod_ever_mt_gluc_or_lip,
           "preperiod_ever_rx_antidiab"=Odat$preperiod_ever_rx_antidiab,
           "preperiod_ever_rx_cm_nondiab"=Odat$preperiod_ever_rx_cm_nondiab,
           "preperiod_ever_rx_other"=Odat$preperiod_ever_rx_other,
           "calculated_age"=Odat$calculated_age,
           "preperiod_olanzeq_dose_total"=Odat$preperiod_olanzeq_dose_total,
           "preperiod_er_mhsa"=Odat$preperiod_er_mhsa,
           "preperiod_er_nonmhsa"=Odat$preperiod_er_nonmhsa,
           "preperiod_er_injury"=Odat$preperiod_er_injury,
           "preperiod_cond_mhsa"=Odat$preperiod_cond_mhsa,
           "preperiod_cond_nonmhsa"=Odat$preperiod_cond_nonmhsa,
           "preperiod_cond_injury"=Odat$preperiod_cond_injury,
           "preperiod_los_mhsa"=Odat$preperiod_los_mhsa,
           "preperiod_los_nonmhsa"=Odat$preperiod_los_nonmhsa,
           "preperiod_los_injury"=Odat$preperiod_los_injury) 

colnames(L)[which(colnames(L)=="West Virginia")] <- "West_Virginia"
colnames(L)[which(colnames(L)=="South Dakota")] <- "South_Dakota"

if(scale.continuous){
  continuous.vars <- c("calculated_age","preperiod_olanzeq_dose_total","preperiod_drug_use_days","preperiod_er_mhsa","preperiod_er_nonmhsa","preperiod_er_injury","preperiod_cond_mhsa",
                       "preperiod_cond_nonmhsa","preperiod_cond_injury","preperiod_los_mhsa","preperiod_los_nonmhsa","preperiod_los_injury") 
  
  L[,continuous.vars] <- scale(L[,continuous.vars]) # scale continuous vars
}

if(use.SL==FALSE){ # exclude multicolinear variables for GLM (VIF)
  glm.exlude <- c("preperiod_olanzeq_dose_total","California")
  L <- L[,!colnames(L)%in%glm.exlude]
}

# TV covariates

tv <- Odat[,grep("monthly",colnames(Odat), value=TRUE)]

if(scale.continuous){
  continuous.tv.vars <- c(timevar_count_vars ,timevar_bincum_vars)
  
  tv[,continuous.tv.vars] <- scale(tv[,continuous.tv.vars]) # scale continuous vars
}

## treatment rule info

# store observed treatment assignment

obs.treatment <- lapply(0:t.end, function(t) factor(Odat$drug_group[Odat$month_number==t]))
names(obs.treatment) <- paste0("A_",seq(0,t.end))

obs.ID <- lapply(0:t.end, function(t) as.numeric(rownames(Odat[Odat$month_number==t,])))
names(obs.ID) <- paste0("ID_",seq(0,t.end))

treatments <- lapply(1:(t.end+1), function(t) as.data.frame(dummify(obs.treatment[[t]]))) # t-length list

# store drug fill, acute care, and preperiod conditions for dynamic rule (same lengths as obs.treatment) 

obs.fill <- lapply(0:t.end, function(t) ifelse((Odat$monthly_ever_rx_antidiab==1 | Odat$monthly_ever_rx_cm_nondiab==1) & Odat$month_number <=t,1,0)[Odat$month_number==t]) # any antidiabetic or non-diabetic cardiometabolic drug is filled
names(obs.fill) <- paste0("L_",seq(0,t.end))  

obs.test <- lapply(0:t.end, function(t) ifelse(Odat$monthly_ever_mt_gluc_or_lip==1 & Odat$month_number <=t,1,0)[Odat$month_number==t]) # lipid or glucose lab tests
names(obs.test) <- paste0("L_",seq(0,t.end))  

obs.care <- lapply(0:t.end, function(t) ifelse((Odat$monthly_los_mhsa>0 | Odat$monthly_er_mhsa>0) & Odat$month_number <=t,1,0)[Odat$month_number==t]) # any acute care (inpatient or ED visits) for MH is observed
names(obs.care) <- paste0("L_",seq(0,t.end))  

obs.bipolar <- lapply(0:t.end, function(t) ifelse(Odat$smi_condition=="bipolar" & Odat$month_number <=t,1,0)[Odat$month_number==t])
names(obs.bipolar) <- paste0("V_",seq(0,t.end))  

obs.schiz <- lapply(0:t.end, function(t) ifelse(Odat$smi_condition=="schiz" & Odat$month_number <=t,1,0)[Odat$month_number==t])
names(obs.schiz) <- paste0("V_",seq(0,t.end))  

# store first censored time

obs.censored <- sapply(0:t.end, function(t) ifelse(Odat$days_to_censored <= (1095/36)*t,1,0))
colnames(obs.censored) <- paste0("C_",seq(0,t.end))

time.censored <- data.frame("ID"=which(rowSums(obs.censored)>0),
                            "time_censored"=NA)

for(t in t.end:0){ # time first censored
  time.censored$time_censored[which(obs.censored[which(rowSums(obs.censored)>0),][,paste0("C_",t)]==1)] <- t
}

# store observed Y
# outcome is diabetes within 36 months

obs.Y <- sapply(0:t.end, function(t) ifelse((!is.na(Odat$days_to_diabetes) & Odat$days_to_diabetes <= (1095/36)*t), 1, 0))
colnames(obs.Y) <- paste0("Y_",seq(0,t.end))

# dummies for who followed treatment rule in observed data 

drug.levels <- c("ARIPIPRAZOLE","HALOPERIDOL","OLANZAPINE","QUETIAPINE","RISPERIDONE","ZIPRASIDONE")

static <- lapply(1:(t.end+1), function(t) factor(rep_len("OLANZAPINE",length.out=length(obs.treatment[[t]])), levels=drug.levels)) # Static: Everyone gets quetiap (if bipolar), halo (if schizophrenia), ari (if MDD) and stays on it
for(t in 1:(t.end+1)){
  static[[t]] <- ifelse(!is.na(obs.bipolar[[t]]) & obs.bipolar[[t]]==1, "QUETIAPINE", ifelse(!is.na(obs.schiz[[t]]) & obs.schiz[[t]]==1, "HALOPERIDOL", "ARIPIPRAZOLE"))
}

dynamic <- lapply(1:(t.end+1), function(t) factor(rep_len("RISPERIDONE",length.out=length(obs.treatment[[t]])), levels=drug.levels))   # Dynamic: Everyone starts with risp.
# If (i) any antidiabetic or non-diabetic cardiometabolic drug is filled OR metabolic testing is observed, or (ii) any acute care for MH is observed, then switch to quetiap. (if bipolar), halo. (if schizophrenia), ari (if MDD); otherwise, stay on risp.
for(t in 2:(t.end+1)){
  dynamic[[t]][(!is.na(obs.fill[[t]]) & obs.fill[[t]]>0) | (!is.na(obs.test[[t]]) & obs.test[[t]]>0) | (!is.na(obs.care[[t]]) & obs.care[[t]]>0)] <- ifelse(obs.bipolar[[t]][!is.na(obs.bipolar[[t]]) & ((!is.na(obs.fill[[t]]) & obs.fill[[t]]>0) | (!is.na(obs.test[[t]]) & obs.test[[t]]>0) | (!is.na(obs.care[[t]]) & obs.care[[t]]>0))]==1, "QUETIAPINE", ifelse(obs.schiz[[t]][!is.na(obs.schiz[[t]]) & ((!is.na(obs.fill[[t]]) & obs.fill[[t]]>0) | (!is.na(obs.test[[t]]) & obs.test[[t]]>0) | (!is.na(obs.care[[t]]) & obs.care[[t]]>0))]==1, "HALOPERIDOL", "ARIPIPRAZOLE"))
}

stochastic <- obs.treatment # Stochastic: at each t>0, 95% chance of staying with treatment at t-1, 5% chance of randomly switching according to Multinomial distibution.
for(t in 2:(t.end+1)){
  if(any(is.na(obs.treatment[[t]]))){
    obs.treatment[[t]][is.na(obs.treatment[[t]])] <- 0
  }
  stochastic.probs <- as.matrix((matrix(obs.treatment[[t]],length(obs.treatment[[t]]),J)=="ARIPIPRAZOLE")+0)*matrix(c(0.95,0.01,0.01,0.01,0.01,0.01),length(obs.treatment[[t]]),J,byrow = TRUE) +
    as.matrix((matrix(obs.treatment[[t]],length(obs.treatment[[t]]),J)=="HALOPERIDOL")+0)*matrix(c(0.01,0.95,0.01,0.01,0.01,0.01),length(obs.treatment[[t]]),J,byrow = TRUE) +
    as.matrix((matrix(obs.treatment[[t]],length(obs.treatment[[t]]),J)=="OLANZAPINE")+0)*matrix(c(0.01,0.01,0.95,0.01,0.01,0.01),length(obs.treatment[[t]]),J,byrow = TRUE) +
    as.matrix((matrix(obs.treatment[[t]],length(obs.treatment[[t]]),J)=="QUETIAPINE")+0)*matrix(c(0.01,0.01,0.01,0.95,0.01,0.01),length(obs.treatment[[t]]),J,byrow = TRUE) +
    as.matrix((matrix(obs.treatment[[t]],length(obs.treatment[[t]]),J)=="RISPERIDONE")+0)*matrix(c(0.01,0.01,0.01,0.01,0.95,0.01),length(obs.treatment[[t]]),J,byrow = TRUE) +
    as.matrix((matrix(obs.treatment[[t]],length(obs.treatment[[t]]),J)=="ZIPRASIDONE")+0)*matrix(c(0.01,0.01,0.01,0.01,0.01,0.95),length(obs.treatment[[t]]),J,byrow = TRUE)
  if(any(is.na(stochastic.probs))){
    stochastic.probs[is.na(stochastic.probs)]<- .Machine$double.eps # placeholder 
  }
  stochastic[[t]] <- Multinom(1, stochastic.probs)
  levels(stochastic[[t]]) <- drug.levels
}

obs.treatment.rule <- list()
obs.treatment.rule[["static"]] <- lapply(1:(t.end+1), function(t) (static[[t]] == obs.treatment[[t]])+0 )
obs.treatment.rule[["dynamic"]] <- lapply(1:(t.end+1), function(t) (dynamic[[t]] == obs.treatment[[t]])+0 )
obs.treatment.rule[["stochastic"]] <- lapply(1:(t.end+1), function(t) (stochastic[[t]] == obs.treatment[[t]])+0 )

# re-arrange so it is in same structure as QAW list
obs.rules <- lapply(1:(t.end+1), function(t) sapply(obs.treatment.rule, "[", t))
for(t in 2:(t.end+1)){ # cumulative sum across lists
  obs.rules[[t]] <- lapply(1:length(obs.rules[[t]]), function(i) obs.rules[[t]][[i]][intersect(obs.ID[[t]],obs.ID[[t-1]])] + obs.rules[[t-1]][[i]][intersect(obs.ID[[t]],obs.ID[[t-1]])])
}

for(t in 1:(t.end+1)){
  obs.rules[[t]] <- lapply(1:length(obs.rules[[t]]), function(i)(obs.rules[[t]][[i]]==t) +0)
}

obs.rules <- lapply(1:length(obs.rules), function(t) setNames(obs.rules[[t]], names(obs.treatment.rule) ))

# plot treatment adherence
png(paste0(output_dir,paste0("treatment_adherence_analysis_weights_loc_",weights.loc,"_use_simulated_", use.simulated,".png")))
plotSurvEst(surv = list("Static"=sapply(1:length(obs.rules), function(t) mean(obs.rules[[t]][[1]])), "Dynamic"=sapply(1:length(obs.rules), function(t) mean(obs.rules[[t]][[2]])), "Stochastic"=sapply(1:length(obs.rules), function(t) mean(obs.rules[[t]][[3]]))),
            ylab = "Share of patients who continued to follow each rule", 
            main = paste("Treatment rule adherence", data.label), 
            legend.xyloc = "topright", xaxt="n")
axis(1, at = seq(1, (t.end+1), by = 3))
dev.off() 

# store observed Ys
rules <- c("static","dynamic","stochastic")
Y.observed <- lapply(1:length(rules), function(i) sapply(1:t.end, function(t) mean(obs.Y[,paste0("Y_",t)][which(obs.rules[[t+1]][[i]]==1)]))) # skip Y in t=0
names(Y.observed) <- rules

Y.observed[["overall"]] <- sapply(1:(t.end+1), function(t) mean(obs.Y[,paste0("Y_",t-1)]))

png(paste0(output_dir,paste0("survival_plot_analysis_weights_loc_",weights.loc,"_use_simulated_", use.simulated,".png")))
plotSurvEst(surv = list("Static"=1-Y.observed[["static"]], "Dynamic"=1-Y.observed[["dynamic"]], "Stochastic"=1-Y.observed[["stochastic"]]),
            ylab = "Share of patients without diabetes diagnosis", 
            main = paste("Observed outcomes", data.label),
            ylim = c(0.7,1),
            legend.xyloc = "bottomleft", xaxt="n")
axis(1, at = seq(1, t.end, by = 5))
lines(1:(t.end+1), 1-Y.observed[["overall"]], type = "l", lty = 2)
dev.off()

## Manual TMLE (ours)

initial_model_for_A <- list()
initial_model_for_A_bin <- list()
initial_model_for_C  <- list() 
initial_model_for_Y  <- list()
initial_model_for_Y_bin  <- list()
tmle_contrasts  <- list() 
tmle_contrasts_bin  <- list()

tmle_estimates <- list()
iptw_estimates <- list()
iptw_bin_estimates <- list()
gcomp_estimates <- list()
Ahat_tmle <- list()
prob_share <- list()
Chat_tmle <- list()

tmle_bin_estimates <- list()
Ahat_tmle_bin <- list()
prob_share_bin <- list()
Chat_tmle_bin <- list()

if(estimator=="tmle"){ 
  
  tmle_dat <- data.frame("month_number"=Odat$month_number,L,tv, "days_to_censored"=Odat$days_to_censored, "drug_group"=Odat$drug_group, "days_to_death"=Odat$days_to_death, "days_to_diabetes"=Odat$days_to_diabetes)

  tmle_dat$L1 <- unlist(sapply(0:t.end, function(t) ifelse((tmle_dat$monthly_ever_rx_antidiab==1 | tmle_dat$monthly_ever_rx_cm_nondiab==1) & tmle_dat$month_number <=t,1,0)[tmle_dat$month_number==t]))

  tmle_dat$L2 <- unlist(sapply(0:t.end, function(t) ifelse(tmle_dat$monthly_ever_mt_gluc_or_lip==1 & tmle_dat$month_number <=t,1,0)[tmle_dat$month_number==t]))

  tmle_dat$L3 <- unlist(sapply(0:t.end, function(t) ifelse((tmle_dat$monthly_los_mhsa>0 | tmle_dat$monthly_er_mhsa>0)& tmle_dat$month_number <=t,1,0)[tmle_dat$month_number==t]))

  tv <- cbind(tv, "L1"=tmle_dat$L1, "L2"=tmle_dat$L2, "L3"=tmle_dat$L3)

  tmle_dat$Y <- unlist(sapply(0:t.end, function(t) obs.Y[,paste0("Y_", t)][which(tmle_dat$month_number ==t)]))

  tmle_dat$A <- as.numeric(unlist(sapply(0:t.end, function(t) obs.treatment[[paste0("A_", t)]])))

  tmle_dat$ID <- unlist(sapply(0:t.end, function(t) obs.ID[[paste0("ID_", t)]]))

  tmle_dat$C <- unlist(sapply(0:t.end, function(t) obs.censored[,paste0("C_", t)][which(tmle_dat$month_number ==t)]))
  
  tmle_dat <- 
    tmle_dat %>%
    arrange(ID, month_number) %>%  # Ensure data is ordered by hcp_patient_id and month_number
    group_by(ID) %>% # create first, second, and third-order lags
    mutate(Y.lag = dplyr::lag(Y, n = 1, default = NA),
           Y.lag2 = dplyr::lag(Y, n = 2, default = NA),
           Y.lag3 = dplyr::lag(Y, n = 3, default = NA),
           L1.lag = dplyr::lag(L1, n = 1, default = NA),
           L1.lag2 = dplyr::lag(L1, n = 2, default = NA),
           L1.lag3 = dplyr::lag(L1, n = 3, default = NA),
           L2.lag = dplyr::lag(L2, n = 1, default = NA), 
           L2.lag2 = dplyr::lag(L2, n = 2, default = NA),
           L2.lag3 = dplyr::lag(L2, n = 3, default = NA),
           L3.lag = dplyr::lag(L3, n = 1, default = NA), 
           L3.lag2 = dplyr::lag(L3, n = 2, default = NA),
           L3.lag3 = dplyr::lag(L3, n = 3, default = NA),
           A.lag = dplyr::lag(A, n = 1, default = NA), 
           A.lag2 = dplyr::lag(A, n = 2, default = NA),
           A.lag3 = dplyr::lag(A, n = 3, default = NA))
  
  tmle_dat <- cbind(tmle_dat, dummify(factor(tmle_dat$A)), dummify(factor(tmle_dat$A.lag)), dummify(factor(tmle_dat$A.lag2)), dummify(factor(tmle_dat$A.lag3))) # binarize categorical variables

  if(scale.continuous){
    tmle_dat[c("L1","L1.lag","L1.lag2","L1.lag3")] <- scale(tmle_dat[c("L1","L1.lag","L1.lag2","L1.lag3")]) # scale continuous variables
  }
  
  treat.names <-  c("A1","A2","A3","A4","A5","A6","A1.lag","A2.lag","A3.lag","A4.lag","A5.lag","A6.lag","A1.lag2","A2.lag2","A3.lag2","A4.lag2","A5.lag2","A6.lag2","A1.lag3","A2.lag3","A3.lag3","A4.lag3","A5.lag3","A6.lag3")
  
  colnames(tmle_dat)[colnames(tmle_dat) %in% colnames(tmle_dat)[(length(colnames(tmle_dat))-length(treat.names)+1):length(colnames(tmle_dat))]] <- treat.names
  
  tmle_covars_Y <- tmle_covars_A <- tmle_covars_C <- c()
  tmle_covars_Y <- c(colnames(L), colnames(tv), treat.names, "Y.lag", "Y.lag2", "Y.lag3") #incl lagged Y
  tmle_covars_A <- tmle_covars_Y[!tmle_covars_Y%in%c("Y.lag","Y.lag2","Y.lag3","A1", "A2", "A3", "A4", "A5", "A6")] # incl lagged A
  tmle_covars_C <- c(tmle_covars_A, "A1", "A2", "A3", "A4", "A5", "A6")
  
  tmle_dat[,c("Y.lag","Y.lag2","Y.lag3","L1.lag","L1.lag2", "L1.lag3","L2.lag", "L2.lag2", "L2.lag3","L3.lag","L3.lag2","L3.lag3")][is.na(tmle_dat[,c("Y.lag","Y.lag2","Y.lag3","L1.lag","L1.lag2", "L1.lag3","L2.lag", "L2.lag2", "L2.lag3","L3.lag","L3.lag2","L3.lag3")])] <- 0 # make lagged NAs zero
  
  tmle_dat <- tmle_dat[,!colnames(tmle_dat)%in%c("A.lag","A.lag2","A.lag3")] # clean up
  tmle_dat$A <- factor(tmle_dat$A)
  
  tmle_dat$t <- tmle_dat$month_number

  ##  fit initial treatment model
  
  # multinomial
  if(weights.loc!='none'){
    initial_model_for_A <-  readRDS(paste0(output_dir, 
                                           "initial_model_for_A_",
                                           "estimator_",estimator,
                                           "_treatment_rule_",treatment.rule,
                                           "_n_folds_",n.folds,
                                           "_scale_continuous_",scale.continuous,
                                           "_use_SL_", use.SL,".rds"))
  }else{
    initial_model_for_A_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                           learners = if(use.SL) learner_stack_A else make_learner(Lrnr_glm),
                                           metalearner = metalearner_A,
                                           keep_extra=FALSE)
    
    initial_model_for_A <- lapply(0:t.end, function(t){ # going forward in time
      
      tmle_dat_sub <- tmle_dat[tmle_dat$t==t,][!colnames(tmle_dat)%in%c("Y","C")]
      
      # define cross-validation folds
      folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
      
      # define task and candidate learners
      
      initial_model_for_A_task <- make_sl3_Task(tmle_dat_sub, 
                                                covariates = tmle_covars_A,
                                                outcome = "A", 
                                                outcome_type="categorical", 
                                                folds = folds) 
      
      # train
      initial_model_for_A_sl_fit <- initial_model_for_A_sl$train(initial_model_for_A_task)
      
      return(list("preds"=initial_model_for_A_sl_fit$predict(initial_model_for_A_task),
                  "folds"= folds,
                  "task"=initial_model_for_A_task,
                  "fit"=initial_model_for_A_sl_fit,
                  "data"=tmle_dat_sub)) 
    })
    
    saveRDS(initial_model_for_A, paste0(output_dir, 
                                        "initial_model_for_A_",
                                        "estimator_",estimator,
                                        "_treatment_rule_",treatment.rule,
                                        "_n_folds_",n.folds,
                                        "_scale_continuous_",scale.continuous,
                                        "_use_SL_", use.SL,".rds"))
  }
  
  g_preds <- lapply(1:length(initial_model_for_A), function(i) data.frame(matrix(unlist(lapply(initial_model_for_A[[i]]$preds, unlist)), nrow=length(lapply(initial_model_for_A[[i]]$preds, unlist)), byrow=TRUE)) ) # t length list of estimated propensity scores 
  g_preds <- lapply(1:length(initial_model_for_A), function(x) setNames(g_preds[[x]], grep("A[0-9]$",colnames(tmle_dat), value=TRUE)) )
  
  g_preds_ID <- lapply(1:length(initial_model_for_A), function(i) unlist(lapply(initial_model_for_A[[i]]$data$ID, unlist))) 
  
  g_preds_cuml <- vector("list", length(g_preds))
  
  g_preds_cuml[[1]] <- g_preds[[1]]
  
  for (i in 2:length(g_preds)) {
    g_preds_cuml[[i]] <- g_preds[[i]][which(g_preds_ID[[i-1]]%in%g_preds_ID[[i]]),] * g_preds_cuml[[i-1]][which(g_preds_ID[[i-1]]%in%g_preds_ID[[i]]),]
  }
  
  g_preds_cuml_bounded <- lapply(1:length(initial_model_for_A), function(x) boundProbs(g_preds_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores                             
  
  # binomial
  
  initial_model_for_A_sl_bin <- make_learner(Lrnr_sl, # cross-validates base models
                                             learners = if(use.SL) learner_stack_A_bin else make_learner(Lrnr_glm),
                                             metalearner = metalearner_A_bin,
                                             keep_extra=FALSE)
  
  if(weights.loc!='none'){
    initial_model_for_A_bin <-  readRDS(paste0(output_dir, 
                                           "initial_model_for_A_bin_",
                                           "estimator_",estimator,
                                           "_treatment_rule_",treatment.rule,
                                           "_n_folds_",n.folds,
                                           "_scale_continuous_",scale.continuous,
                                           "_use_SL_", use.SL,".rds"))
  }else{
    
    initial_model_for_A_bin <- lapply(0:t.end, function(t){
      
      tmle_dat_sub <- tmle_dat[tmle_dat$t==t,][!colnames(tmle_dat)%in%c("C","Y")]
      
      # define cross-validation
      folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
      
      # define task and candidate learners
      
      initial_model_for_A_task_bin <- lapply(1:J, function(j) make_sl3_Task(cbind("A"=dummify(tmle_dat_sub$A)[,j],tmle_dat_sub[tmle_covars_A]), 
                                                                            covariates = tmle_covars_A, 
                                                                            outcome = "A",
                                                                            outcome_type="binomial",
                                                                            folds = folds)) 
      # train
      
      initial_model_for_A_sl_fit_bin <- lapply(1:J, function(j) initial_model_for_A_sl_bin$train(initial_model_for_A_task_bin[[j]]))
      
      return(list("preds"=sapply(1:J, function(j) initial_model_for_A_sl_fit_bin[[j]]$predict(initial_model_for_A_task_bin[[j]])), 
                  "folds"= folds,
                  "task"=initial_model_for_A_task_bin,
                  "fit"=initial_model_for_A_sl_fit_bin,
                  "data"=tmle_dat_sub)) 
    })
    
    saveRDS(initial_model_for_A_bin, paste0(output_dir, 
                                            "initial_model_for_A_bin_",
                                            "estimator_",estimator,
                                            "_treatment_rule_",treatment.rule,
                                            "_n_folds_",n.folds,
                                            "_scale_continuous_",scale.continuous,
                                            "_use_SL_", use.SL,".rds"))
  }
  
  g_preds_bin <- lapply(1:length(initial_model_for_A_bin), function(i) data.frame(initial_model_for_A_bin[[i]]$preds) ) # t length list of estimated propensity scores 
  g_preds_bin <- lapply(1:length(initial_model_for_A_bin), function(x) setNames(g_preds_bin[[x]], grep("A[0-9]$",colnames(tmle_dat), value=TRUE)) )
  
  g_preds_bin_ID <- lapply(1:length(initial_model_for_A_bin), function(i) unlist(lapply(initial_model_for_A_bin[[i]]$data$ID, unlist)))
  
  g_preds_bin_cuml <- vector("list", length(g_preds_bin))
  
  g_preds_bin_cuml[[1]] <- g_preds_bin[[1]]
  
  for (i in 2:length(g_preds_bin)) {
    g_preds_bin_cuml[[i]] <- g_preds_bin[[i]][which(g_preds_bin_ID[[i-1]]%in%g_preds_bin_ID[[i]]),] * g_preds_bin_cuml[[i-1]][which(g_preds_bin_ID[[i-1]]%in%g_preds_bin_ID[[i]]),]
  }  
  
  g_preds_bin_cuml_bounded <- lapply(1:length(initial_model_for_A_bin), function(x) boundProbs(g_preds_bin_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores                             
  
  ##  fit initial censoring model
  ## implicitly fit on those that are uncensored until t-1
  
  initial_model_for_C_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                         learners = if(use.SL) learner_stack_A_bin else make_learner(Lrnr_glm),
                                         metalearner = metalearner_A_bin,
                                         keep_extra=FALSE)
  
  if(weights.loc!='none'){
    initial_model_for_C <-  readRDS(paste0(output_dir, 
                                           "initial_model_for_C_",
                                           "estimator_",estimator,
                                           "_treatment_rule_",treatment.rule,
                                           "_n_folds_",n.folds,
                                           "_scale_continuous_",scale.continuous,
                                           "_use_SL_", use.SL,".rds"))
  }else{
    
    initial_model_for_C <- lapply(0:t.end, function(t){
      
      tmle_dat_sub <- tmle_dat[tmle_dat$t==t,][!colnames(tmle_dat)%in%c("A","Y")]
      
      # define cross-validation appropriatte for dependent data
      folds <- origami::make_folds(tmle_dat_sub, fold_fun = folds_vfold, V = n.folds)
      
      # define task and candidate learners
      initial_model_for_C_task <- make_sl3_Task(data=tmle_dat_sub,
                                                covariates = tmle_covars_C, 
                                                outcome = "C",
                                                outcome_type="binomial", 
                                                folds = folds) 
      
      # train
      
      initial_model_for_C_sl_fit <- initial_model_for_C_sl$train(initial_model_for_C_task)
      
      return(list("preds"=initial_model_for_C_sl_fit$predict(initial_model_for_C_task),
                  "folds"= folds,
                  "task"=initial_model_for_C_task,
                  "fit"=initial_model_for_C_sl_fit,
                  "data"=tmle_dat_sub)) 
    })
    
    saveRDS(initial_model_for_C, paste0(output_dir, 
                                        "initial_model_for_C_",
                                        "estimator_",estimator,
                                        "_treatment_rule_",treatment.rule,
                                        "_n_folds_",n.folds,
                                        "_scale_continuous_",scale.continuous,
                                        "_use_SL_", use.SL,".rds"))
  }
  
  C_preds <- lapply(1:length(initial_model_for_C), function(i) 1-initial_model_for_C[[i]]$preds) # t length list # C=1 if uncensored; C=0 if censored  
  
  C_preds_ID <- lapply(1:length(initial_model_for_C), function(i) unlist(lapply(initial_model_for_C[[i]]$data$ID, unlist))) 
  
  C_preds_cuml <- vector("list", length(C_preds))
  
  C_preds_cuml[[1]] <- C_preds[[1]]
  
  for (i in 2:length(C_preds)) {
    C_preds_cuml[[i]] <- C_preds[[i]][which(C_preds_ID[[i-1]]%in%C_preds_ID[[i]])] * C_preds_cuml[[i-1]][which(C_preds_ID[[i-1]]%in%C_preds_ID[[i]])]
  }    
  C_preds_cuml_bounded <- lapply(1:length(initial_model_for_C), function(x) boundProbs(C_preds_cuml[[x]],bounds=gbound))  # winsorized cumulative propensity scores                             
  
  ## sequential g-formula
  ## model is fit on all uncensored and alive (until t-1)
  ## the outcome is the observed Y for t=T and updated Y if t<T
  
  initial_model_for_Y_sl <- make_learner(Lrnr_sl, # cross-validates base models
                                         learners = if(use.SL) learner_stack_Y else make_learner(Lrnr_glm),
                                         metalearner = metalearner_Y,
                                         keep_extra=FALSE)
  
  initial_model_for_Y_sl_cont <- make_learner(Lrnr_sl, # cross-validates base models
                                              learners = if(use.SL) learner_stack_Y_cont else make_learner(Lrnr_glm),
                                              metalearner = metalearner_Y_cont,
                                              keep_extra=FALSE)
  
  initial_model_for_Y <- list()
  initial_model_for_Y_bin <- list() # updated Y's are used as outcomes for t<T
  initial_model_for_Y[[t.end]] <- sequential_g(t=t.end, tmle_dat=tmle_dat[!tmle_dat$ID%in%time.censored$ID[which(time.censored$time_censored<t.end)],], n.folds=n.folds, tmle_covars_Y=tmle_covars_Y, initial_model_for_Y_sl, ybound) # for t=T fit on measured Y
  initial_model_for_Y_bin[[t.end]] <- initial_model_for_Y[[t.end]] # same
  
  # Update equations, calculate point estimate and variance
  # backward in time: T.end, ..., 1
  # fit on thoese uncesnored until t-1
  
  tmle_rules <- list("static"=static_mtp,
                     "dynamic"=dynamic_mtp,
                     "stochastic"=stochastic_mtp)
  
  tmle_contrasts <-list()
  tmle_contrasts_bin <- list()
  tmle_contrasts[[t.end]] <- getTMLELong(initial_model_for_Y=initial_model_for_Y[[t.end]], tmle_rules=tmle_rules, tmle_covars_Y=tmle_covars_Y, g_preds_bounded=g_preds_cuml_bounded[[t.end+1]], C_preds_bounded=C_preds_cuml_bounded[[t.end+1]], obs.treatment=treatments[[t.end+1]], obs.rules=obs.rules[[t.end+1]], gbound=gbound, ybound=ybound, t.end=t.end)
  tmle_contrasts_bin[[t.end]] <- getTMLELong(initial_model_for_Y=initial_model_for_Y_bin[[t.end]], tmle_rules=tmle_rules, tmle_covars_Y=tmle_covars_Y, g_preds_bounded=g_preds_bin_cuml_bounded[[t.end+1]], C_preds_bounded=C_preds_cuml_bounded[[t.end+1]], obs.treatment=treatments[[t.end+1]], obs.rules=obs.rules[[t.end+1]], gbound=gbound, ybound=ybound, t.end=t.end)
  
  for(t in (t.end-1):1){
    initial_model_for_Y[[(t)]] <- sapply(1:length(tmle_rules), function(i) sequential_g(t=t, tmle_dat=tmle_dat[!tmle_dat$ID%in%time.censored$ID[which(time.censored$time_censored<t)],], n.folds=n.folds, tmle_covars_Y=tmle_covars_Y, initial_model_for_Y_sl=initial_model_for_Y_sl_cont, ybound=ybound, Y_pred = tmle_contrasts[[t+1]]$Qstar[[i]]))
    initial_model_for_Y_bin[[(t)]] <- sapply(1:length(tmle_rules), function(i) sequential_g(t=t, tmle_dat=tmle_dat[!tmle_dat$ID%in%time.censored$ID[which(time.censored$time_censored<t)],], n.folds=n.folds, tmle_covars_Y=tmle_covars_Y,  initial_model_for_Y_sl=initial_model_for_Y_sl_cont, ybound=ybound, Y_pred =tmle_contrasts_bin[[t+1]]$Qstar[[i]]))
    
    tmle_contrasts[[t]] <- sapply(1:length(tmle_rules), function(i) getTMLELong(initial_model_for_Y=initial_model_for_Y[[t]][,i], tmle_rules=tmle_rules, tmle_covars_Y=tmle_covars_Y, g_preds_bounded=g_preds_cuml_bounded[[t+1]], C_preds_bounded=C_preds_cuml_bounded[[t+1]], obs.treatment=treatments[[t+1]], obs.rules=obs.rules[[t+1]], gbound=gbound, ybound=ybound, t.end=t.end))
    tmle_contrasts_bin[[t]] <- sapply(1:length(tmle_rules), function(i) getTMLELong(initial_model_for_Y=initial_model_for_Y_bin[[t]][,i], tmle_rules=tmle_rules, tmle_covars_Y=tmle_covars_Y, g_preds_bounded=g_preds_bin_cuml_bounded[[t+1]], C_preds_bounded=C_preds_cuml_bounded[[t+1]], obs.treatment=treatments[[t+1]], obs.rules=obs.rules[[t+1]], gbound=gbound, ybound=ybound, t.end=t.end))
  }

  saveRDS(initial_model_for_Y, paste0(output_dir, 
                                      "initial_model_for_Y_",
                                      "estimator_",estimator,
                                      "_treatment_rule_",treatment.rule,
                                      "_n_folds_",n.folds,
                                      "_scale_continuous_",scale.continuous,
                                      "_use_SL_", use.SL,".rds"))
  
  saveRDS(initial_model_for_Y_bin, paste0(output_dir, 
                                      "initial_model_for_Y_bin_",
                                      "estimator_",estimator,
                                      "_treatment_rule_",treatment.rule,
                                      "_n_folds_",n.folds,
                                      "_scale_continuous_",scale.continuous,
                                      "_use_SL_", use.SL,".rds"))
  # plot estimated survival curves
  
  tmle_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar[[x]]))) # static, dynamic, stochastic
  tmle_bin_estimates <-  cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t]][,x]$Qstar[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t.end]]$Qstar[[x]]))) 
  
  png(paste0(output_dir,paste0("survival_plot_tmle_estimates_weights_loc_",weights.loc,"_use_simulated_", use.simulated,".png")))
  plotSurvEst(surv = list("Static"= tmle_estimates[1,], "Dynamic"= tmle_estimates[2,], "Stochastic"= tmle_estimates[3,]),  
              ylab = "Estimated share of patients without diabetes diagnosis", 
              main = "LTMLE (ours, multinomial) outcomes (CMS data)",
              ylim = c(0.5,1))
  dev.off()
  
  png(paste0(output_dir,paste0("survival_plot_tmle_estimates_bin_weights_loc_",weights.loc,"_use_simulated_", use.simulated,".png")))
  plotSurvEst(surv = list("Static"= tmle_bin_estimates[1,], "Dynamic"= tmle_bin_estimates[2,], "Stochastic"= tmle_bin_estimates[3,]),  
              ylab = "Estimated outcomes (CMS data)", 
              main = "Estimated share of patients without diabetes diagnosis",
              ylim = c(0.5,1))
  dev.off()
  
  iptw_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar_iptw[[x]]))) # static, dynamic, stochastic
  iptw_bin_estimates <-  cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t]][,x]$Qstar_iptw[[x]], na.rm=TRUE))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts_bin[[t.end]]$Qstar_iptw[[x]], na.rm=TRUE))) 
  
  if(r==1){
    png(paste0(output_dir,paste0("survival_plot_iptw_estimates_",n, ".png")))
    plotSurvEst(surv = list("Static"= iptw_estimates[1,], "Dynamic"= iptw_estimates[2,], "Stochastic"= iptw_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "IPTW (ours, multinomial) estimated counterfactuals",
                xlab = "Month",
                ylim = c(0.5,1),
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
    
    png(paste0(output_dir,paste0("survival_plot_iptw_bin_estimates_",n, ".png")))
    plotSurvEst(surv = list("Static"= iptw_bin_estimates[1,], "Dynamic"= iptw_bin_estimates[2,], "Stochastic"= iptw_bin_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "IPTW (ours, binomial) estimated counterfactuals",
                xlab = "Month",
                ylim = c(0.5,1))
    dev.off()
  }
  
  gcomp_estimates <- cbind(sapply(1:(t.end-1), function(t) sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t]][,x]$Qstar_gcomp[[x]]))), sapply(1:(ncol(obs.rules[[t+1]])), function(x) 1-mean(tmle_contrasts[[t.end]]$Qstar_gcomp[[x]]))) # static, dynamic, stochastic
  
  if(r==1){
    png(paste0(output_dir,paste0("survival_plot_gcomp_estimates_",n, ".png")))
    plotSurvEst(surv = list("Static"= gcomp_estimates[1,], "Dynamic"= gcomp_estimates[2,], "Stochastic"= gcomp_estimates[3,]),  
                ylab = "Estimated share of patients without diabetes diagnosis", 
                main = "G-comp. (ours) estimated counterfactuals",
                xlab = "Month",
                ylim = c(0.5,1),
                legend.xyloc = "bottomleft", xindx = 1:t.end, xaxt="n")
    axis(1, at = seq(1, t.end, by = 5))
    dev.off()
  }
  
  # calc share of cumulative probabilities of continuing to receive treatment according to the assigned treatment rule which are smaller than 0.025
  
  prob_share <- lapply(1:(t.end+1), function(t) sapply(1:ncol(obs.rules[[(t)]]), function(i) round(colMeans(g_preds_cuml_bounded[[(t)]][which(obs.rules[[(t)]][na.omit(tmle_dat[tmle_dat$t==(t),])$ID,][,i]==1),]<0.025, na.rm=TRUE),2)))
  for(t in 1:(t.end+1)){
    colnames(prob_share[[t]]) <- colnames(obs.rules[[(t.end)]])
  }
  
  names(prob_share) <- paste0("t=",seq(0,t.end))
  
  prob_share.bin <- lapply(1:(t.end+1), function(t) sapply(1:ncol(obs.rules[[(t)]]), function(i) colMeans(g_preds_bin_cuml_bounded[[(t)]][which(obs.rules[[(t)]][na.omit(tmle_dat[tmle_dat$t==(t),])$ID,][,i]==1),]<0.025, na.rm=TRUE)))
  for(t in 1:(t.end+1)){
    colnames(prob_share.bin[[t]]) <- colnames(obs.rules[[(t.end)]])
  }
  
  names(prob_share.bin) <- paste0("t=",seq(0,t.end))
  
  # calc CIs 
  
  tmle_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored)
  tmle_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y_bin, time.censored)
  
  iptw_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, iptw=TRUE)
  iptw_est_var_bin <- TMLE_IC(tmle_contrasts_bin, initial_model_for_Y_bin, time.censored, iptw=TRUE)
  
  gcomp_est_var <- TMLE_IC(tmle_contrasts, initial_model_for_Y, time.censored, gcomp=TRUE)
  
  # store results

  Ahat_tmle  <- g_preds_cuml_bounded
  Ahat_tmle_bin  <- g_preds_bin_cuml_bounded
  
  Chat_tmle  <- C_preds_cuml_bounded
}

# save results
results <- list("tmle_contrasts"=tmle_contrasts, "tmle_contrasts_bin"=tmle_contrasts_bin, 
            "Ahat_tmle"=Ahat_tmle, "Chat_tmle"=Chat_tmle, "yhat_tmle"= tmle_estimates, "prob_share_tmle"= prob_share,
            "Ahat_tmle_bin"=Ahat_tmle_bin,"yhat_tmle_bin"= tmle_bin_estimates, "prob_share_tmle_bin"= prob_share_bin,
            "yhat_gcomp"= gcomp_estimates,"gcomp_est_var"=gcomp_est_var,
            "yhat_iptw"= iptw_estimates,"iptw_est_var"=iptw_est_var,
            "yhat_iptw_bin"= iptw_bin_estimates,"iptw_est_var_bin"=iptw_est_var_bin)

saveRDS(results, filename)

stopCluster(cl)