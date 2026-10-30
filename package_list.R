############################################################################################
# Installs the R packages used by the simulation code (run once: Rscript package_list.R).  #
# The results in the paper were produced with R 4.3.3 and the package versions listed in   #
# session_info.txt; the three packages installed from GitHub are pinned to the commits     #
# used.                                                                                     #
############################################################################################

repos <- "https://cloud.r-project.org"

# CRAN packages: simulation (simulation.R, src/) and tables/figures (long_sim_plots.R)
cran <- c("data.table", "dplyr", "ggplot2", "glmnet", "gtools", "igraph", "latex2exp", "MASS",
          "nnet", "origami", "progressr", "purrr", "ranger", "readr", "remotes", "reticulate",
          "tidyr")
install.packages(setdiff(cran, rownames(installed.packages())), repos = repos)

# GitHub packages, pinned to the commits used for the paper
remotes::install_github("jvpoulos/Rsolnp@16049fb5c426766177f8b442bc9619033deefd2f",
                        upgrade = "never") # Rsolnp 1.16 (fork)
remotes::install_github("osofr/simcausal@bce5ad71eb90c0cc7d5ac37c4065ffd0ef1f8e15",
                        upgrade = "never", build_vignettes = FALSE) # simcausal 0.5.5
remotes::install_github("tlverse/sl3@0e8f2365bcbe54010b8120c04a7a2dcfc8119227",
                        upgrade = "never") # sl3 1.4.5
