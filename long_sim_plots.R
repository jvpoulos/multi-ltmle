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
R <- 325
t.end <- 36

treatment.rules <- c("static","dynamic","stochastic")

# Add flag for combining results (TRUE) or LSTM only (FALSE)
combine_results <- FALSE  # Set to TRUE to include both SL and LSTM results

# Load results based on flag
results_lstm <- readRDS("intermediate_results_lstm.rds")
if(combine_results) {
  results_sl <- readRDS("final_results.rds")
  n.estimators <- 10  # 5 estimators * 2 (SL and LSTM)
  estimator_names <- c(
    "LTMLE-SL (multi.)", "LTMLE-SL (bin.)", "G-Comp-SL",
    "IPTW-SL (multi.)", "IPTW-SL (bin.)",
    "LTMLE-RNN (multi.)", "LTMLE-RNN (bin.)", "G-Comp-RNN",
    "IPTW-RNN (multi.)", "IPTW-RNN (bin.)"
  )
} else {
  n.estimators <- 5  # 5 LSTM estimators only
  estimator_names <- c(
    "LTMLE-RNN (multi.)", "LTMLE-RNN (bin.)", "G-Comp-RNN",
    "IPTW-RNN (multi.)", "IPTW-RNN (bin.)"
  )
}

n.rules <- as.numeric(length(treatment.rules))

# Function to reshape matrix from wide to long format
reshape_matrix <- function(mat) {
  # Reshape the matrix to vector by columns
  as.vector(mat)
}

# Create LSTM dataframe
create_lstm_df <- function(results) {
  # Reshape all matrices to vectors
  abs.bias <- c(
    reshape_matrix(abs(results$bias_tmle)),
    reshape_matrix(abs(results$bias_tmle_bin)),
    reshape_matrix(abs(results$bias_gcomp)),
    reshape_matrix(abs(results$bias_iptw)),
    reshape_matrix(abs(results$bias_iptw_bin))
  )
  
  coverage <- c(
    reshape_matrix(results$CP_tmle),
    reshape_matrix(results$CP_tmle_bin),
    reshape_matrix(results$CP_gcomp),
    reshape_matrix(results$CP_iptw),
    reshape_matrix(results$CP_iptw_bin)
  )
  
  ciw <- c(
    reshape_matrix(results$CIW_tmle),
    reshape_matrix(results$CIW_tmle_bin),
    reshape_matrix(results$CIW_gcomp),
    reshape_matrix(results$CIW_iptw),
    reshape_matrix(results$CIW_iptw_bin)
  )
  
  n_rows <- length(abs.bias)
  n_timesteps <- nrow(results$bias_tmle)
  n_treatments <- ncol(results$bias_tmle)
  
  df <- data.frame(
    "abs.bias" = abs.bias,
    "coverage" = coverage,
    "CIW" = ciw,
    "Estimator" = rep(estimator_names[if(combine_results) 6:10 else 1:5], 
                      each = n_timesteps * n_treatments),
    "Rule" = rep(rep(treatment.rules, each = n_timesteps), n.estimators),
    "t" = rep(2:t.end, n.estimators * n_treatments)
  )
  
  return(df)
}

# Create the dataframe(s)
if(combine_results) {
  # Create and combine both dataframes
  lstm_df <- create_lstm_df(results_lstm)
  sl_df <- create_sl_df(results_sl)  # You'll need to implement this similarly
  results.df <- rbind(sl_df, lstm_df)
} else {
  # LSTM results only
  results.df <- create_lstm_df(results_lstm)
}

# Format rule names
proper <- function(x) paste0(toupper(substr(x, 1, 1)), tolower(substring(x, 2)))
results.df$Rule <- proper(results.df$Rule)

# Create coverage rate variable
results.df <- results.df %>% 
  group_by(Estimator, Rule) %>% 
  mutate(CP = mean(coverage)) 

# Get summary stats
setDT(results.df)[, as.list(summary(CP)), by = Estimator]
setDT(results.df)[, as.list(summary(CIW)), by = Estimator]
setDT(results.df)[, as.list(summary(abs.bias)), by = Estimator]

setDT(results.df)[, as.list(summary(CP)), by = Rule]
setDT(results.df)[, as.list(summary(CIW)), by = Rule]
setDT(results.df)[, as.list(summary(abs.bias)), by = Rule]

setDT(results.df)[, as.list(summary(CP)), by = .(Estimator,Rule)]
setDT(results.df)[, as.list(summary(CIW)), by = .(Estimator,Rule)]
setDT(results.df)[, as.list(summary(abs.bias)), by = .(Estimator,Rule)]

# Print out dimensions to verify
print("Data frame dimensions:")
print(dim(results.df))
print("First few rows:")
print(head(results.df))

# Reshape and plot
results.df <- as.data.frame(results.df)
results_long <- reshape2::melt(results.df, id.vars=c("Estimator","Rule", "t"))

# Bias plot
sim.results.bias <- ggplot(
  data=results_long[results_long$variable=="abs.bias" & results_long$value <0.25,],
  aes(x=factor(t), y=value, fill=forcats::fct_rev(Estimator))
) + 
  geom_boxplot(outlier.alpha = 0.3, outlier.size = 1, outlier.stroke = 0.1, lwd=0.25) +
  facet_grid(Rule ~  ., scales = "free") +  
  xlab("Time") + 
  ylab("Abs. diff. btwn. true and estimated counterfactual outcomes") +
  ggtitle(paste0("Absolute bias")) +
  scale_fill_discrete(name = "Estimator:  ") +
  theme(legend.position="bottom") +
  theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
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

# Get the positions of the strips in the gtable
posR <- subset(z.bias$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips
width <- z.bias$widths[max(posR$r)]
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

ggsave(paste0("sim_results/long_simulation_bias_estimand","_J_",J,"_n_",n,"_R_",R,".png"), plot = z.bias, scale=1.75)

# Coverage plot
sim.results.coverage <- ggplot(
  data=results_long[results_long$variable=="CP",],
  aes(x=factor(t), y=value, colour=forcats::fct_rev(Estimator), group=forcats::fct_rev(Estimator))
) +
  geom_line() +
  facet_grid(Rule ~  ., scales = "free") +  
  xlab("Time") + 
  ylab("Share of estimated CIs containing true target quantity") + 
  ggtitle(paste0("Coverage probability")) + 
  scale_colour_discrete(name = "Estimator:  ") +
  geom_hline(yintercept = 0.95, linetype="dotted") +
  theme(legend.position="bottom") +
  theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
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

# Get the positions of the strips in the gtable
posR <- subset(z.coverage$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips
width <- z.coverage$widths[max(posR$r)]
z.coverage <- gtable_add_cols(z.coverage, width, max(posR$r))  

# Position the grobs in the gtable
z.coverage <- gtable_add_grob(z.coverage, stripR, t = min(posR$t)+0.1, l = max(posR$r) + 1, b = max(posR$b)+1, name = "strip-right")

# Add small gaps between strips
z.coverage <- gtable_add_cols(z.coverage, unit(1/5, "line"), max(posR$r))

# Draw it
grid.newpage()
grid.draw(z.coverage)

ggsave(paste0("sim_results/long_simulation_coverage_estimand","_J_",J,"_n_",n,"_R_",R,".png"), plot = z.coverage, scale=1.75)

# CI width plot
sim.results.CI.width <- ggplot(
  data=results_long[results_long$variable=="CIW",],
  aes(x=factor(t), y=value, fill=forcats::fct_rev(Estimator))
) + 
  geom_boxplot(outlier.alpha = 0.3, outlier.size = 1, outlier.stroke = 0.1, lwd=0.25) +
  facet_grid(Rule ~  ., scales = "free") +  
  xlab("Time") + 
  ylab("Difference btwn. upper & lower bounds of estimated CIs") + 
  ggtitle(paste0("Confidence interval width")) +
  scale_fill_discrete(name = "Estimator:  ") +
  theme(legend.position="bottom") +
  theme(plot.title = element_text(hjust = 0.5, family="serif", size=16)) +
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

# Get the positions of the strips in the gtable
posR <- subset(z.width$layout, grepl("strip-r", name), select = t:r)

# Add a new column to the right of current right strips
width <- z.width$widths[max(posR$r)]
z.width <- gtable_add_cols(z.width, width, max(posR$r))  

# Position the grobs in the gtable
z.width <- gtable_add_grob(z.width, stripR, t = min(posR$t)+0.1, l = max(posR$r) + 1, b = max(posR$b)+1, name = "strip-right")

# Add small gaps between strips
z.width <- gtable_add_cols(z.width, unit(1/5, "line"), max(posR$r))

# Draw it
grid.newpage()
grid.draw(z.width)

ggsave(paste0("sim_results/long_simulation_ci_width_estimand","_J_",J,"_n_",n,"_R_",R,".png"), plot = z.width, scale=1.75)