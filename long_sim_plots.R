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
n <- 10000 #15000
R <- 275 #325
t.end <- 36

treatment.rules <- c("static","dynamic","stochastic")

estimator <- "tmle"
estimators <- c("tmle","tmle_bin","gcomp","iptw","iptw_bin")

n.estimators <-as.numeric(length(estimators))
n.rules <-as.numeric(length(treatment.rules))

# Load results data

options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE) # args <- c("outputs/20240130")
output.path <- as.character(args[1])

filenames <- list.files(path=output.path, pattern = ".rds", full.names = TRUE)
  
filenames <- filenames[grep(paste0("J_",J),filenames)]
filenames <- filenames[grep(paste0("R_",R),filenames)]
filenames <- filenames[grep(paste0("n_",n,"_"),filenames)]
filenames <- filenames[grep(paste0("estimator_",estimator,"_"),filenames)]

if(any( duplicated(substring(filenames, 18)))){
  print("removing duplicate filenames")
  filenames <- filenames[-which(duplicated(substring(filenames, 18)))]
}

results.obj <-readRDS(filenames)
omit.result <- names(which(apply(results.obj,2,function(x) class(x$Ahat_tmle))=="character"))
R <- R-length(omit.result)

results <- list() # structure is: [[filename]][[metric]]
for(f in filenames){
  print(f)
  result.matrix <- readRDS(f)
  result.matrix <- result.matrix[,!colnames(result.matrix) %in% omit.result]
  if(isTRUE(grep("lmtp", f)==1)){
    estimator <- "lmtp"
  }
  if(isTRUE(grep("ltmle", f)==1)){
    estimator <- "ltmle"
  }
  if(isTRUE(grep("tmle", f)==1)){
    estimator <- c("tmle","tmle_bin","gcomp","iptw","iptw_bin")
  }
  #tmle
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
  
  #iptw
  bias.iptw <- matrix(NA, R, n.rules)
  colnames(bias.iptw) <- paste0("bias_", treatment.rules)
  
  bias.iptw.bin <- matrix(NA, R, n.rules)
  colnames(bias.iptw.bin) <- paste0("bias_", treatment.rules)
  
  CP.iptw <- matrix(NA, R, n.rules)
  colnames(CP.iptw) <- paste0("CP_",treatment.rules)
  
  CP.iptw.bin <- matrix(NA, R, n.rules)
  colnames(CP.iptw.bin) <- paste0("CP_",treatment.rules)
  
  CIW.iptw <- matrix(NA, R, n.rules)
  colnames(CIW.iptw) <- paste0("CIW_",treatment.rules)
  
  CIW.iptw.bin <- matrix(NA, R, n.rules)
  colnames(CIW.iptw.bin) <- paste0("CIW_",treatment.rules)
  
  #gcomp
  bias.gcomp <- matrix(NA, R, n.rules)
  colnames(bias.gcomp) <- paste0("bias_", treatment.rules)

  CP.gcomp <- matrix(NA, R, n.rules)
  colnames(CP.gcomp) <- paste0("CP_",treatment.rules)
  
  CIW.gcomp <- matrix(NA, R, n.rules)
  colnames(CIW.gcomp) <- paste0("CIW_",treatment.rules)
  
  # tmle
  bias.tmle <- mapply(cbind, result.matrix["bias_tmle",])
  bias.tmle.bin <- mapply(cbind, result.matrix["bias_tmle_bin",])
  
  CP.tmle <- mapply(cbind, result.matrix["CP_tmle",])
  CP.tmle.bin <- mapply(cbind, result.matrix["CP_tmle_bin",])
  
  CIW.tmle <- mapply(cbind, result.matrix["CIW_tmle",])
  CIW.tmle.bin <-  mapply(cbind, result.matrix["CIW_tmle_bin",])
  
  # iptw
  bias.iptw <- mapply(cbind, result.matrix["bias_iptw",])
  bias.iptw.bin <- mapply(cbind, result.matrix["bias_iptw_bin",])
  
  CP.iptw <- mapply(cbind, result.matrix["CP_iptw",])
  CP.iptw.bin <- mapply(cbind, result.matrix["CP_iptw_bin",])
  
  CIW.iptw <- mapply(cbind, result.matrix["CIW_iptw",])
  CIW.iptw.bin <-  mapply(cbind, result.matrix["CIW_iptw_bin",])
  
  # gcomp
  bias.gcomp <- mapply(cbind, result.matrix["bias_gcomp",])
  
  CP.gcomp <- mapply(cbind, result.matrix["CP_gcomp",])
  
  CIW.gcomp <- mapply(cbind, result.matrix["CIW_gcomp",])
  
  results[[f]] <- list("bias_tmle"=bias.tmle,"bias_tmle_bin"=bias.tmle.bin,"CP_tmle"=CP.tmle,"CP_tmle_bin"=CP.tmle.bin,"CIW_tmle"=CIW.tmle,"CIW_tmle_bin"=CIW.tmle.bin,
                       "bias_iptw"=bias.iptw,"bias_iptw_bin"=bias.iptw.bin,"CP_iptw"=CP.iptw,"CP_iptw_bin"=CP.iptw.bin,"CIW_iptw"=CIW.iptw,"CIW_iptw_bin"=CIW.iptw.bin,
                       "bias_gcomp"=bias.gcomp,"CP_gcomp"=CP.gcomp,"CIW_gcomp"=CIW.gcomp,"R"=R)
}

# Create New lists
# structure is: [[estimator]][[filename]]

bias <- list()
bias[["tmle"]] <- lapply(filenames, function(f) results[[f]]$bias_tmle)
bias[["tmle_bin"]] <- lapply(filenames, function(f) results[[f]]$bias_tmle_bin)
bias[["iptw"]] <- lapply(filenames, function(f) results[[f]]$bias_iptw)
bias[["iptw_bin"]] <- lapply(filenames, function(f) results[[f]]$bias_iptw_bin)
bias[["gcomp"]] <- lapply(filenames, function(f) results[[f]]$bias_gcomp)

CP <- list()
CP[["tmle"]] <- lapply(filenames, function(f) results[[f]]$CP_tmle)
CP[["tmle_bin"]] <- lapply(filenames, function(f) results[[f]]$CP_tmle_bin)
CP[["iptw"]] <- lapply(filenames, function(f) results[[f]]$CP_iptw)
CP[["iptw_bin"]] <- lapply(filenames, function(f) results[[f]]$CP_iptw_bin)
CP[["gcomp"]] <- lapply(filenames, function(f) results[[f]]$CP_gcomp)

CIW <- list()
CIW[["tmle"]] <- lapply(filenames, function(f) results[[f]]$CIW_tmle)
CIW[["tmle_bin"]] <- lapply(filenames, function(f) results[[f]]$CIW_tmle_bin)
CIW[["iptw"]] <- lapply(filenames, function(f) results[[f]]$CIW_iptw)
CIW[["iptw_bin"]] <- lapply(filenames, function(f) results[[f]]$CIW_iptw_bin)
CIW[["gcomp"]] <- lapply(filenames, function(f) results[[f]]$CIW_gcomp)

# Create dataframe for plot: estimators ---> rules ---> R ---> T
results.df <- data.frame("abs.bias"=c(sapply(estimators, function(x) sapply(1:length(1:(t.end-1)), function(t) abs(unlist(bias[[x]][[1]][t,]))))), 
                         "coverage"=c(sapply(estimators, function(x) sapply(1:length(1:(t.end-1)), function(t) unlist(CP[[x]][[1]][t,])))), 
                         "CIW"=c(sapply(estimators, function(x) sapply(1:length(1:(t.end-1)), function(t) unlist(CIW[[x]][[1]][t,])))), 
                         "Estimator" = rep(c(" TMLE (Multi.) "," TMLE (Bin.) ", " G-Comp. "," IPTW (Multi.) "," IPTW (Bin.) "), each=length(treatment.rules)*R*length(1:(t.end-1))), 
                         "Rule" = rep(rep(rep(treatment.rules, R), length(1:(t.end-1))), n.estimators), 
                          "t"= rep(rep(2:t.end, each=R*length(treatment.rules)), n.estimators))

proper <- function(x) paste0(toupper(substr(x, 1, 1)), tolower(substring(x, 2)))
results.df$Rule <- proper(results.df$Rule)

# create coverage rate variable

results.df <- results.df %>% 
  group_by(Estimator,Rule) %>% 
  mutate(CP = mean(coverage)) 

# get summary stats by estimator

setDT(results.df)[, as.list(summary(CP)), by = Estimator] # CP
setDT(results.df)[, as.list(summary(CIW)), by = Estimator]
setDT(results.df)[, as.list(summary(abs.bias)), by = Estimator]

# get summary stats by rule

setDT(results.df)[, as.list(summary(CP)), by = Rule] 
setDT(results.df)[, as.list(summary(CIW)), by = Rule]
setDT(results.df)[, as.list(summary(abs.bias)), by = Rule]

# get summary stats by estimator and rule

setDT(results.df)[, as.list(summary(CP)), by = .(Estimator,Rule)] 
setDT(results.df)[, as.list(summary(CIW)), by = .(Estimator,Rule)]
setDT(results.df)[, as.list(summary(abs.bias)), by = .(Estimator,Rule)]

# reshape and plot
results.df <- as.data.frame(results.df)
results_long <- reshape2::melt(results.df, id.vars=c("Estimator","Rule", "t"))  # convert to long format

# bias 
sim.results.bias <- ggplot(data=results_long[results_long$variable=="abs.bias" & results_long$value <0.25,], # rm extreme outliers
                           aes(x=factor(t), y=value, fill=forcats::fct_rev(Estimator)))  + geom_boxplot(outlier.alpha = 0.3,outlier.size = 1, outlier.stroke = 0.1, lwd=0.25) +
  facet_grid(Rule ~  ., scales = "free")  +  
  xlab("Time") + ylab("Abs. diff. btwn. true and estimated counterfactual outcomes") +  ggtitle(paste0("Absolute bias")) +
  scale_fill_discrete(name = "Estimator:  ") +
  theme(legend.position="bottom") +   theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
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

# Get the ggplot grob
z.bias <- ggplotGrob(sim.results.bias)

# Labels 
labelR <- "Treatment rule"

# Get the positions of the strips in the gtable: t = top, l = left, ...
posR <- subset(z.bias$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips, 
width <- z.bias$widths[max(posR$r)]    # width of current right strips

z.bias <- gtable_add_cols(z.bias, width, max(posR$r))  

# Construct the new strip grobs
stripR <- gTree(name = "Strip_right", children = gList(
  rectGrob(gp = gpar(col = NA, fill = "grey85")),
  textGrob(labelR, rot = -90, gp = gpar(fontsize=16, col = "grey10"))))

# Position the grobs in the gtable
z.bias <- gtable_add_grob(z.bias, stripR, t = min(posR$t)+0.1, l = max(posR$r) + 1, b = max(posR$b)+1, name = "strip-right")

# Add small gaps between strips
z.bias <- gtable_add_cols(z.bias, unit(1/5, "line"), max(posR$r))

# Draw it
grid.newpage()
grid.draw(z.bias)

ggsave(paste0("sim_results/long_simulation_bias_estimand","_J_",J,"_n_",n,"_R_",R,".png"), plot = z.bias,scale=1.75)

# coverage
sim.results.coverage <- ggplot(data=results_long[results_long$variable=="CP",],
                           aes(x=factor(t), y=value, colour=forcats::fct_rev(Estimator), group=forcats::fct_rev(Estimator)))  +   geom_line()   +
  facet_grid(Rule ~  ., scales = "free")  +  
  xlab("Time") + ylab("Share of estimated CIs containing true target quantity") + ggtitle(paste0("Coverage probability")) + 
  scale_colour_discrete(name = "Estimator:  ") +
  geom_hline(yintercept = 0.95, linetype="dotted")+
  theme(legend.position="bottom") +   theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
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

# Get the ggplot grob
z.coverage <- ggplotGrob(sim.results.coverage)

# Get the positions of the strips in the gtable: t = top, l = left, ...
posR <- subset(z.coverage$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips, 
width <- z.coverage$widths[max(posR$r)]    # width of current right strips

z.coverage <- gtable_add_cols(z.coverage, width, max(posR$r))  

# Position the grobs in the gtable
z.coverage <- gtable_add_grob(z.coverage, stripR, t = min(posR$t)+0.1, l = max(posR$r) + 1, b = max(posR$b)+1, name = "strip-right")

# Add small gaps between strips
z.coverage <- gtable_add_cols(z.coverage, unit(1/5, "line"), max(posR$r))

# Draw it
grid.newpage()
grid.draw(z.coverage)

ggsave(paste0("sim_results/long_simulation_coverage_estimand","_J_",J,"_n_",n,"_R_",R,".png"),plot = z.coverage, scale=1.75)

# CI width
sim.results.CI.width <- ggplot(data=results_long[results_long$variable=="CIW",],
                               aes(x=factor(t), y=value, fill=forcats::fct_rev(Estimator)))  + geom_boxplot(outlier.alpha = 0.3,outlier.size = 1, outlier.stroke = 0.1, lwd=0.25) +
  facet_grid(Rule ~  ., scales = "free")  +  
  xlab("Time") + ylab("Difference btwn. upper & lower bounds of estimated CIs") + ggtitle(paste0("Confidence interval width")) +
  scale_fill_discrete(name = "Estimator:  ") +
  theme(legend.position="bottom") +   theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
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

# Get the ggplot grob
z.width <- ggplotGrob(sim.results.CI.width)

# Get the positions of the strips in the gtable: t = top, l = left, ...
posR <- subset(z.width$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips, 
width <- z.width$widths[max(posR$r)]    # width of current right strips

z.width <- gtable_add_cols(z.width, width, max(posR$r))  

# Position the grobs in the gtable
z.width <- gtable_add_grob(z.width, stripR, t = min(posR$t)+0.1, l = max(posR$r) + 1, b = max(posR$b)+1, name = "strip-right")

# Add small gaps between strips
z.width <- gtable_add_cols(z.width, unit(1/5, "line"), max(posR$r))

# Draw it
grid.newpage()
grid.draw(z.width)

ggsave(paste0("sim_results/long_simulation_ci_width_estimand","_J_",J,"_n_",n,"_R_",R,".png"), plot = z.width, scale=1.75)