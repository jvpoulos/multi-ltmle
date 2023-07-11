###############################################
# Super Learner functions for ltmle package   #
###############################################

SL.ranger.100 = function(...) {
  SL.ranger(..., num.trees = 100)
}

SL.ranger.500 = function(...) {
  SL.ranger(..., num.trees = 100)
}

SL.xgboost.1000 = function(...) {
  SL.xgboost(..., nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective = "reg:logistic")
}

SL.glmnet.lasso = function(...) {
  SL.glmnet(..., alpha = 1)
}

SL.glmnet.ridge = function(...) {
  SL.glmnet(..., alpha = 0)
}

SL.glmnet.25 = function(...) {
  SL.glmnet(..., alpha = 0.25)
}

SL.glmnet.50 = function(...) {
  SL.glmnet(..., alpha = 0.50)
}

SL.glmnet.75 = function(...) {
  SL.glmnet(..., alpha = 0.75)
}

SL.library <- list("Q"=c("SL.xgboost.1000","SL.ranger.100","SL.glmnet.lasso","SL.glmnet.50"), 
                   "g"=c("SL.xgboost.1000","SL.ranger.100","SL.glmnet.lasso","SL.glmnet.50"))