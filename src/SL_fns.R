###############################################
# Super Learner functions for ltmle package   #
###############################################

SL.ranger.100 = function(...) {
  SL.ranger(..., num.trees = 100)
}

SL.ranger.500 = function(...) {
  SL.ranger(..., num.trees = 100)
}

SL.xgboost.20 = function(...) {
  SL.xgboost(..., nrounds=20, objective = "reg:logistic")
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

SL.library <- list("Q"=c("SL.xgboost.20","SL.ranger.100","SL.ranger.500","SL.glmnet.lasso","SL.glmnet.25","SL.glmnet.50","SL.glmnet.75"), 
                   "g"=c("SL.xgboost.20","SL.ranger.100","SL.ranger.500","SL.glmnet.lasso","SL.glmnet.25","SL.glmnet.50","SL.glmnet.75"))