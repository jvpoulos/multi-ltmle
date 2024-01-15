###########################################################################
# Super Learner functions for sl3                                         #
###########################################################################

# stack learners into a model

learner_stack_A <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective="multi:softprob", eval_metric="mlogloss",num_class=J), list("Lrnr_ranger",num.trees=100),list("Lrnr_glmnet",nfolds = n.folds,alpha = 1, family = "multinomial"), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5, family = "multinomial"))  
learner_stack_A_bin <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective = "reg:logistic"), list("Lrnr_ranger",num.trees=100), list("Lrnr_glmnet",nfolds = n.folds,alpha = 1, family = "binomial"), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5, family = "binomial"))  
learner_stack_Y <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective = "reg:logistic"), list("Lrnr_ranger",num.trees=100), list("Lrnr_glmnet",nfolds = n.folds,alpha = 1, family = "binomial"), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5, family = "binomial")) 
learner_stack_Y_cont <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective = "reg:squarederror"), list("Lrnr_ranger",num.trees=100), list("Lrnr_glmnet",nfolds = n.folds,alpha = 1, family = "gaussian"), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5, family = "gaussian"))

# metalearner defaults (https://tlverse.org/sl3/reference/default_metalearner.html)
metalearner_Y <- make_learner(Lrnr_solnp,learner_function=metalearner_logistic_binomial, eval_function=loss_loglik_binomial)
metalearner_Y_cont <- make_learner(Lrnr_solnp,learner_function=metalearner_linear, eval_function=loss_squared_error)
metalearner_A <- make_learner(Lrnr_solnp,learner_function=metalearner_linear_multinomial, eval_function=loss_loglik_multinomial)
metalearner_A_bin <- make_learner(Lrnr_solnp,learner_function=metalearner_logistic_binomial, eval_function=loss_loglik_binomial)
