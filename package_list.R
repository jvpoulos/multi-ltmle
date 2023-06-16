######################
# Install packages   #
######################

packages <- c("devtools","ggplot2","nnet","tmle","MASS","tidyverse","data.table","SuperLearner","Rsolnp","reshape2","origami","tictoc","weights","grid","car","latex2exp","progressr","future","lmtle","gtools")

super.learner <- TRUE
dependencies <- FALSE # data.table, stringi, HMisc dependencies might be needed for SuperLearner libraries
if(super.learner){
  packages <- c(packages, c("glmnet","ranger","rpart","nnls","xgboost"))
}
if(dependencies){
  remove.packages("data.table")                         # First remove the current version
  install.packages("data.table", type = "source", repos = "https://Rdatatable.gitlab.io/data.table")
  install.packages("stringi", dependencies=TRUE, INSTALL_opts = c('--no-lock'))

  install.packages(c("HMisc",packages),repos = "http://cran.us.r-project.org")
} else{
  install.packages(packages,repos = "http://cran.us.r-project.org")
}

# development packages
devtools::install_github("nt-williams/lmtp@sl3")
remotes::install_github("tlverse/sl3")

devtools::install_github('osofr/gridisl', build_vignettes = FALSE)
devtools::install_github('osofr/stremr')
devtools::install_github('osofr/simcausal', build_vignettes = FALSE)

# doMPI
doMPI <- TRUE
if(doMPI){
  install.packages("Rmpi", repos = "http://cran.us.r-project.org")
  install.packages("doMPI", dependencies=TRUE, repos = "http://cran.us.r-project.org")
}

# keras CPU/GPU
use.GPU <- FALSE
if(use.GPU){
  install.packages("keras", repos = "http://cran.us.r-project.org")
  install_keras(tensorflow = "gpu") # requires tensorflow-gpu installation # e.g, https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1605009731/Tensorflow+on+O2
  remotes::install_github("jvpoulos/sl3") # v1.4.4  <<-- changes to keras
}else{
  install.packages("keras", repos = "http://cran.us.r-project.org")
}