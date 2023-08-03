#################################################################
# Descriptive tables and plots for analysis                 #
#################################################################

print(paste0("Summarizing results from output directory: ", output_dir))

## Summary statistics table

print(tableNominal(data.frame(Odat, dummify(factor(Odat$state_character),show.na = FALSE),
                              dummify(factor(Odat$race_defined_character),show.na = FALSE), 
                              dummify(factor(Odat$year),show.na = FALSE), 
                              dummify(factor(Odat$payer_index),show.na = FALSE), 
                              dummify(factor(Odat$smi_condition)))[c("California","Georgia","Iowa",                         
                                                                             "Mississippi","Oklahoma","South.Dakota","West.Virginia","black","latino","white","other","mdd","schiz","bipolar","female", "X2008", "X2009", "X2010",
                                                                             "dual","mdcr",nontimevar_bin_vars ,nontimevar_binary_vars, timevar_binary_vars)], group=factor(Odat$drug_group), prec = 3, cumsum=FALSE, longtable = FALSE))

print(tableContinuous(data.frame(Odat)[c("calculated_age","preperiod_drug_use_days","preperiod_er_mhsa","preperiod_er_nonmhsa","preperiod_er_injury","preperiod_cond_mhsa",
                                                 "preperiod_cond_nonmhsa","preperiod_cond_injury","preperiod_los_mhsa","preperiod_los_nonmhsa","preperiod_los_injury", nontimevar_count_vars, timevar_bincum_vars, timevar_count_vars)], group=Odat$drug_group, prec = 3, cumsum=FALSE, stats= c("n", "min", "mean", "max", "s"), longtable = FALSE))