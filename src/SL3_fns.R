###########################################################################
# Super Learner functions for sl3                                         #
###########################################################################

# Function to create pre-configured treatment model SuperLearner
# This avoids the non-function error by creating everything in one place
create_treatment_model_sl <- function(n.folds = 5) {
  # Create ranger learners - safe for categorical outcomes
  ranger1 <- Lrnr_ranger$new(
    num.trees = 100, 
    min.node.size = 5, 
    respect.unordered.factors = "partition", 
    num.threads = 1
  )
  
  ranger2 <- Lrnr_ranger$new(
    num.trees = 50, 
    min.node.size = 10, 
    respect.unordered.factors = "partition", 
    num.threads = 1
  )
  
  # Simple mean learner
  mean_lrnr <- Lrnr_mean$new()
  
  # Define stack
  stack <- Stack$new(ranger1, ranger2, mean_lrnr)
  
  # Create Super Learner with mean metalearner
  sl <- Lrnr_sl$new(
    learners = stack,
    metalearner = mean_lrnr
  )
  
  return(sl)
}

# Standard learner stacks for binary outcomes
learner_stack_A_bin <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, alpha = 1, family = "binomial")
)

learner_stack_Y <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, alpha = 1, family = "binomial")
)

learner_stack_Y_cont <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, alpha = 1, family = "gaussian")
)

# Standard metalearners
metalearner_Y <- make_learner(Lrnr_solnp, learner_function = metalearner_logistic_binomial, 
                              eval_function = loss_loglik_binomial)
metalearner_Y_cont <- make_learner(Lrnr_solnp, learner_function = metalearner_linear, 
                                  eval_function = loss_squared_error)
metalearner_A_bin <- make_learner(Lrnr_solnp, learner_function = metalearner_logistic_binomial, 
                                 eval_function = loss_loglik_binomial)