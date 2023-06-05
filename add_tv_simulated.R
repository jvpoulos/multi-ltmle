########################################################################################
# Quickly add time-varying covariates to simulated dataset (FOR ILLUSTRATIVE PURPOSES) #
########################################################################################

# make censoring variable name consistent with CMS dataset
names(simdata_from_basevars)[names(simdata_from_basevars) == 'days_to_censored_or_outcome'] <- 'days_to_censored'

# generate time-varying variables 
timevar_binary_vars <- c("monthly_ever_mt_gluc_or_lip", "monthly_ever_rx_antidiab", 
                         "monthly_ever_rx_cm_nondiab", "monthly_ever_rx_other")

nontimevar_binary_vars <- c("preperiod_ever_mt_gluc_or_lip", "preperiod_ever_rx_antidiab",
                            "preperiod_ever_rx_cm_nondiab", "preperiod_ever_rx_other")

timevar_bincum_vars <- c("monthly_evcum_metabolic","monthly_evcum_psych","monthly_evcum_other")

nontimevar_bin_vars <- c("preperiod_ever_metabolic","preperiod_ever_psych","preperiod_ever_other")

timevar_count_vars <- c("monthly_er_mhsa", "monthly_er_nonmhsa", "monthly_er_injury", "monthly_cond_mhsa", "monthly_cond_nonmhsa", 
                        "monthly_cond_injury", "monthly_los_mhsa", "monthly_los_nonmhsa", "monthly_los_injury")

nontimevar_count_vars <- c("preperiod_er_mhsa", "preperiod_er_nonmhsa", "preperiod_er_injury", "preperiod_cond_mhsa", 
                           "preperiod_cond_nonmhsa", "preperiod_cond_injury",
                           "preperiod_los_mhsa", "preperiod_los_nonmhsa", "preperiod_los_injury")


# Set the seed for reproducibility
set.seed(123)

# continuous variable ("monthly_olanzeq_dose_total")
simdata_from_basevars$monthly_olanzeq_dose_total <- simdata_from_basevars$preperiod_olanzeq_dose_total + RnormTrunc(nrow(simdata_from_basevars), mean = mean(simdata_from_basevars$preperiod_olanzeq_dose_total)-1303.43051967477, sd = sd(simdata_from_basevars$preperiod_olanzeq_dose_total)-1467.6408871971, minval = 0, maxval = max(simdata_from_basevars$preperiod_olanzeq_dose_total)-20329.8514516875)
simdata_from_basevars$monthly_olanzeq_dose_total <- RnormTrunc(nrow(simdata_from_basevars), mean = mean(simdata_from_basevars$preperiod_olanzeq_dose_total), sd = sd(simdata_from_basevars$preperiod_olanzeq_dose_total), minval = 0, maxval = max(simdata_from_basevars$preperiod_olanzeq_dose_total), 
                                                               min.low = 0, max.low = summary(simdata_from_basevars$preperiod_olanzeq_dose_total)[[2]], min.high = summary(simdata_from_basevars$preperiod_olanzeq_dose_total)[[3]], max.high = max(simdata_from_basevars$preperiod_olanzeq_dose_total))

# count variables
for(i in 1:length(timevar_count_vars)){
  simdata_from_basevars[,timevar_count_vars[i]] <- NegBinom(nrow(simdata_from_basevars), ifelse(simdata_from_basevars[,nontimevar_count_vars[i]]>0,0.5, 0.1))
}

# binary variables
for(i in 1:length(timevar_binary_vars)){
  simdata_from_basevars[,timevar_binary_vars[i]] <- rbern(nrow(simdata_from_basevars), ifelse(simdata_from_basevars[,nontimevar_binary_vars[i]]==1, 0.5, .1))
}

# cumulative binary variables
for(i in 1:length(timevar_bincum_vars)){
  simdata_from_basevars[,timevar_bincum_vars[i]] <- rbern(nrow(simdata_from_basevars), ifelse(simdata_from_basevars[,nontimevar_bin_vars[i]]==1,1,0.1))
}