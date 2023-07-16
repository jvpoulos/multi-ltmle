############################################################################################
# Combine plots from static setting (T=1) simulations                                       #
############################################################################################

library(tidyverse)
library(ggplot2)
library(data.table)
library(latex2exp)
library(dplyr)
library(grid)
library(gtable)

# Define parameters
J <- 6
n <- 10000
R <- 20
t.end <- 36

treatment.rules <- c("static","dynamic","stochastic")

estimators <- c("tmle","tmle_bin") #lmtp

n.estimators <-as.numeric(length(estimators))
n.rules <-as.numeric(length(treatment.rules))

# Load results data

options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE) # args <- c("outputs/20230712")
output.path <- as.character(args[1])

filenames <- list.files(path=output.path, pattern = ".rds", full.names = TRUE)
  
filenames <- filenames[grep(paste0("J_",J),filenames)]
filenames <- filenames[grep(paste0("R_",R),filenames)]
filenames <- filenames[grep(paste0("n_",n,"_"),filenames)]

if(any( duplicated(substring(filenames, 18)))){
  print("removing duplicate filenames")
  filenames <- filenames[-which(duplicated(substring(filenames, 18)))]
}

results <- list() # structure is: [[filename]][[metric]]
for(f in filenames){
  print(f)
  result.matrix <- readRDS(f)
  result.matrix <- result.matrix
  # if(isTRUE(grep("lmtp", f)==1)){
  #   estimator <- "lmtp"
  # }
  # if(isTRUE(grep("ltmle", f)==1)){
  #   estimator <- "ltmle"
  # }
  if(isTRUE(grep("tmle", f)==1)){
    estimator <- c("tmle","tmle_bin")
  }
  bias.tmle <- matrix(NA, R, n.rules)
  colnames(bias.tmle) <- paste0("bias_", treatment.rules)
  bias.tmle.bin <- matrix(NA, R, n.rules)
  colnames(bias.tmle.bin) <- paste0("bias_", treatment.rules)
  
  CP.tmle <- matrix(NA, R, n.rules)
  colnames(CP.tmle) <- paste0("CP_",treatment.rules)
  CP.tmle.bin <- matrix(NA, R, n.rules)
  colnames(CP.tmle.bin) <- paste0("CP_",treatment.rules)
  
  CIW.tmle <- matrix(NA, R, n.rules)
  colnames(CIW.tmle) <- paste0("CIW_",treatment.rules)
  CIW.tmle.bin <- matrix(NA, R, n.rules)
  colnames(CIW.tmle.bin) <- paste0("CIW_",treatment.rules)
  
  bias.tmle <- mapply(cbind, result.matrix["bias_tmle",])
  bias.tmle.bin <- mapply(cbind, result.matrix["bias_tmle_bin",])
  
  CP.tmle <- mapply(cbind, result.matrix["CP_tmle",])
  CP.tmle.bin <- mapply(cbind, result.matrix["CP_tmle_bin",])
  
  CIW.tmle <- mapply(cbind, result.matrix["CIW_tmle",])
  CIW.tmle.bin <-  mapply(cbind, result.matrix["CIW_tmle_bin",])
  
  results[[f]] <- list("bias_tmle"=bias.tmle,"bias_tmle_bin"=bias.tmle.bin,"CP_tmle"=CP.tmle,"CP_tmle_bin"=CP.tmle.bin,"CIW_tmle"=CIW.tmle,"CIW_tmle_bin"=CIW.tmle.bin,"R"=R)
}

# Create New lists
# structure is: [[estimator]][[filename]]

bias <- list()
bias[["tmle"]] <- lapply(filenames, function(f) results[[f]]$bias_tmle)
bias[["tmle_bin"]] <- lapply(filenames, function(f) results[[f]]$bias_tmle_bin)

CP <- list()
CP[["tmle"]] <- lapply(filenames, function(f) results[[f]]$CP_tmle)
CP[["tmle_bin"]] <- lapply(filenames, function(f) results[[f]]$CP_tmle_bin)

CIW <- list()
CIW[["tmle"]] <- lapply(filenames, function(f) results[[f]]$CIW_tmle)
CIW[["tmle_bin"]] <- lapply(filenames, function(f) results[[f]]$CIW_tmle_bin)

# Create dataframe for plot (SHOULD VARY BY T)
results.df <- data.frame("abs.bias"=abs(unlist(bias)), # sapply(1:length(1:(t.end-1)), function(t) abs(unlist(bias[["tmle"]][[1]][1,])))
                         "Coverage"=unlist(CP),
                         "CIW"=unlist(CIW))
                     #   "t"= 2:t.end) # skip first period
                        # "filename"=c(rep(unlist(sapply(1:length(filenames), function(i) rep(filenames[i], length.out=R))), n.estimators)))

#results.df$Estimator <- c(rep("LMTP (super learner)",length.out=length(c(unlist(CP[[1]])))))

results.df$Estimator <- c(rep("Multinomial (super learner)",length.out=length(c(unlist(CP[[1]])))), rep("Binomial (super learner)",length.out=length(c(unlist(CP[[2]])))))

results.df$rule <- c(rep(treatment.rules,length.out=length(c(unlist(CP[[1]])))), rep(treatment.rules,length.out=length(c(unlist(CP[[2]])))))
  
proper <- function(x) paste0(toupper(substr(x, 1, 1)), tolower(substring(x, 2)))
results.df$rule <- proper(results.df$rule)
# create coverage rate variable

results.df <- results.df %>% 
  group_by(Estimator,rule) %>% 
  mutate(CP = mean(Coverage)) 

# get summary stats by treatment rule

setDT(results.df)[, as.list(summary(CP)), by = rule] # CP
setDT(results.df)[, as.list(summary(CIW)), by = rule]
setDT(results.df)[, as.list(summary(abs.bias)), by = rule]

# reshape and plot
#results.df$id <- with(results.df, paste(rule, J, sep = "_"))
results.df <- as.data.frame(results.df)
#colnames(results.df)[1:3] <- c("abs.bias", "CP", "CIW")
results_long <- reshape2::melt(results.df, id.vars=c("Estimator","rule"))  # convert to long format #,"filename"

# bias 
sim.results.bias <- ggplot(data=results_long[results_long$variable=="abs.bias",],
                           aes(x=rule, y=value, fill=forcats::fct_rev(Estimator)))  + geom_boxplot(outlier.alpha = 0.3,outlier.size = 1, outlier.stroke = 0.1, lwd=0.25) +
 # facet_grid(overlap.setting ~  gamma.setting, scales = "free", labeller=labeller3)  +  
  xlab("Treatment rule") + ylab("Abs. diff. btwn. true and estimated target quantity") +  ggtitle(paste0("Absolute bias")) +
  scale_fill_discrete(name = "Estimator:") +
  #scale_x_discrete(labels=x.labels,
   #                limits = rev) +
  theme(legend.position="bottom") +   theme(plot.title = element_text(hjust = 0.5, family="serif", size=20)) +
  theme(axis.title=element_text(family="serif", size=14)) +
  theme(axis.text.y=element_text(family="serif", size=12)) +
  theme(axis.text.x=element_text(family="serif", size=12, angle = 0, vjust = 0.5, hjust=0.25)) +
  theme(legend.text=element_text(family="serif", size=12)) +
  theme(legend.title=element_text(family="serif", size=12)) +
  theme(strip.text.x = element_text(family="serif", size=14)) +
  theme(strip.text.y = element_text(family="serif", size=14)) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l =0))) +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l =0))) +
  theme(panel.spacing = unit(1, "lines")) 

ggsave(paste0("sim_results/long_simulation_bias_estimand","_J_",J,"_n_",n,"_R_",R,".png"),plot = sim.results.bias)

# coverage
sim.results.coverage <- ggplot(data=results_long[results_long$variable=="CP",],
                           aes(x=rule, y=value, colour=forcats::fct_rev(Estimator), group=forcats::fct_rev(Estimator)))  +   geom_line()  +
  #facet_grid(overlap.setting ~  gamma.setting, scales = "fixed", labeller=labeller3)  +  
  xlab("Treatment rule") + ylab("Share of estimated CIs containing true target quantity") + ggtitle(paste0("Coverage probability")) + 
  scale_colour_discrete(name = "Estimator:") +
  # scale_x_discrete(labels=x.labels,
  #                  limits = rev) +
  geom_hline(yintercept = 0.95, linetype="dotted")+
  theme(legend.position="bottom") +   theme(plot.title = element_text(hjust = 0.5, family="serif", size=20)) +
  theme(axis.title=element_text(family="serif", size=14)) +
  theme(axis.text.y=element_text(family="serif", size=12)) +
  theme(axis.text.x=element_text(family="serif", size=12, angle = 0, vjust = 0.5, hjust=0.25)) +
  theme(legend.text=element_text(family="serif", size=12)) +
  theme(legend.title=element_text(family="serif", size=12)) +
  theme(strip.text.x = element_text(family="serif", size=14)) +
  theme(strip.text.y = element_text(family="serif", size=14)) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l =0))) +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l =0))) +
  theme(panel.spacing = unit(1, "lines"))

ggsave(paste0("sim_results/long_simulation_coverage_estimand","_J_",J,"_n_",n,"_R_",R,".png"),plot = sim.results.coverage)

# CI width
sim.results.CI.width <- ggplot(data=results_long[results_long$variable=="CIW",],
                               aes(x=rule, y=value, fill=forcats::fct_rev(Estimator)))  + geom_boxplot(outlier.alpha = 0.3,outlier.size = 1, outlier.stroke = 0.1, lwd=0.25) +
 # facet_grid(overlap.setting ~  gamma.setting, scales = "free", labeller=labeller3)  + 
  xlab("Treatment rule") + ylab("Difference btwn. upper & lower bounds of estimated CIs") + ggtitle(paste0("Confidence interval width")) +
  scale_fill_discrete(name = "Estimator:") +
  # scale_x_discrete(labels=x.labels,
  #                  limits = rev) +
  theme(legend.position="bottom") +   theme(plot.title = element_text(hjust = 0.5, family="serif", size=20)) +
  theme(axis.title=element_text(family="serif", size=14)) +
  theme(axis.text.y=element_text(family="serif", size=12)) +
  theme(axis.text.x=element_text(family="serif", size=12, angle = 0, vjust = 0.5, hjust=0.25)) +
  theme(legend.text=element_text(family="serif", size=12)) +
  theme(legend.title=element_text(family="serif", size=12)) +
  theme(strip.text.x = element_text(family="serif", size=14)) +
  theme(strip.text.y = element_text(family="serif", size=14)) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l =0))) +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l =0))) +
  theme(panel.spacing = unit(1, "lines"))

ggsave(paste0("sim_results/long_simulation_ci_width_estimand","_J_",J,"_n_",n,"_R_",R,".png"),plot = sim.results.CI.width)