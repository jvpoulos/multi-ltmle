######################
# Install packages   #
######################

packages <- c("ggplot2","nnet","tmle","MASS","data.table","SuperLearner","reshape2","origami","tictoc","weights","grid","car","latex2exp","progressr","future","ltmle","gtools","readr","reporttools")

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
install.packages("https://cran.r-project.org/src/contrib/Archive/imputeMissings/imputeMissings_0.0.3.tar.gz")
remotes::install_github("tlverse/sl3")
remotes::install_github("jvpoulos/Rsolnp") # v1.16

devtools::install_github('osofr/simcausal', build_vignettes = FALSE)

# doMPI
doMPI <- FALSE
if(doMPI){
  install.packages("Rmpi", repos = "http://cran.us.r-project.org")
  install.packages("doMPI", dependencies=TRUE, repos = "http://cran.us.r-project.org")
}else{
  install.packages("parallel", repos = "http://cran.us.r-project.org")
  install.packages("doParallel", repos = "http://cran.us.r-project.org")
  install.packages("foreach", repos = "http://cran.us.r-project.org")
}

# keras/TensorFlow
keras <- TRUE
if(keras){
  install.packages("tensorflow", repos = "http://cran.us.r-project.org")
  install.packages("keras", repos = "http://cran.us.r-project.org")
  install.packages("reticulate", repos = "http://cran.us.r-project.org")
}