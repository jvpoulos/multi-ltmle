###########################################################################
# Super Learner functions for sl3 (lmtp package and our implementation)   #
###########################################################################

# stack learners into a model

if(estimator%in%c("tmle")){
  learner_stack_A <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective="multi:softprob", eval_metric="mlogloss",num_class=J), list("Lrnr_ranger",num.trees=100),list("Lrnr_glmnet",nfolds = n.folds,alpha = 1, family = "multinomial"), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5, family = "multinomial"))  
  learner_stack_A_bin <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective = "reg:logistic"), list("Lrnr_ranger",num.trees=100), list("Lrnr_glmnet",nfolds = n.folds,alpha = 1, family = "binomial"), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5, family = "binomial"))  
  learner_stack_Y <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective = "reg:logistic"), list("Lrnr_ranger",num.trees=100), list("Lrnr_glmnet",nfolds = n.folds,alpha = 1, family = "binomial"), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5, family = "binomial")) 
  learner_stack_Y_cont <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2, objective = "reg:squarederror"), list("Lrnr_ranger",num.trees=100), list("Lrnr_glmnet",nfolds = n.folds,alpha = 1, family = "gaussian"), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5, family = "gaussian"))
}

if(estimator=="tmle-lstm"){
  learner_stack_A <- make_learner_stack(list("Lrnr_lstm_keras",batch_size=32, units=32, dropout=0.5, recurrent_dropout=0.5, activation='tanh', recurrent_activation='sigmoid', recurrent_out='softmax', epochs=100,  layers=2, callbacks = list(keras::callback_early_stopping(patience = 10, restore_best_weights=TRUE)), validation_split=0.2)) #loss="categorical_crossentropy"
  learner_stack_A_bin <- make_learner_stack(list("Lrnr_lstm_keras",batch_size=32, units=32, dropout=0.5, recurrent_dropout=0.5, activation='tanh', recurrent_activation='sigmoid', recurrent_out='sigmoid', epochs=100,  layers=2, callbacks = list(keras::callback_early_stopping(patience = 10, restore_best_weights=TRUE)), validation_split=0.2))# loss="binary_crossentropy
  learner_stack_Y <-  make_learner_stack(list("Lrnr_lstm_keras",batch_size=32, units=32, dropout=0.5, recurrent_dropout=0.5, activation='tanh', recurrent_activation='sigmoid', recurrent_out='sigmoid', epochs=100,  layers=2, callbacks = list(keras::callback_early_stopping(patience = 10, restore_best_weights=TRUE)), validation_split=0.2))# loss="binary_crossentropy"
  learner_stack_Y_cont <-  make_learner_stack(list("Lrnr_lstm_keras",batch_size=32, units=32, dropout=0.5, recurrent_dropout=0.5, activation='tanh', recurrent_activation='sigmoid', recurrent_out='linear', epochs=100,  layers=2, callbacks = list(keras::callback_early_stopping(patience = 10, restore_best_weights=TRUE)), validation_split=0.2))# loss="mse"
}

if(estimator%in%c("tmle","tmle-lstm")){
  # metalearner defaults (https://tlverse.org/sl3/reference/default_metalearner.html)
  metalearner_Y <- make_learner(Lrnr_solnp,learner_function=metalearner_logistic_binomial, eval_function=loss_loglik_binomial)
  metalearner_Y_cont <- make_learner(Lrnr_solnp,learner_function=metalearner_linear, eval_function=loss_squared_error)
  metalearner_A <- make_learner(Lrnr_solnp,learner_function=metalearner_linear_multinomial, eval_function=loss_loglik_multinomial)
  metalearner_A_bin <- make_learner(Lrnr_solnp,learner_function=metalearner_logistic_binomial, eval_function=loss_loglik_binomial)
}

if(estimator%in% c("lmtp-tmle","lmtp-iptw","lmtp-gcomp","lmtp-sdr")){
  learner_stack_A <- learner_stack_Y <- make_learner_stack(list("Lrnr_xgboost",nrounds=1000, early_stopping_rounds=3, max_depth = 2, eta = 1, verbose = 0, nthread = 2), list("Lrnr_ranger",num.trees=100),list("Lrnr_glmnet",nfolds = n.folds,alpha = 1), list("Lrnr_glmnet",nfolds = n.folds,alpha = 0.5)) 
}