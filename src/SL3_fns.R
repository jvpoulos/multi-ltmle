###########################################################################
# Super Learner functions for sl3                                         #
###########################################################################

create_treatment_model_sl <- function(n.folds = 3) {
  # Create a more robust learner stack for multinomial outcomes
  # Direct instantiation of learners to avoid string reference issues
  learner_stack <- Stack$new(
    # Ranger with explicit settings for categorical outcomes
    Lrnr_ranger$new(
      num.trees = 100,
      min.node.size = 5,
      respect.unordered.factors = "partition",
      probability = TRUE
    ),
    
    # GLMnet with L1 regularization (alpha=1)
    Lrnr_glmnet$new(
      nfolds = n.folds,
      nlambda = 10,
      alpha = 1, 
      family = "multinomial"
    ),
    
    # GLMnet with L1 and L2 regularization (alpha=0.5)
    Lrnr_glmnet$new(
      nfolds = n.folds,
      nlambda = 10,
      alpha = 0.5, 
      family = "multinomial"
    )
  )
  
  # Use appropriate metalearner for multinomial outcomes
  # The solnp solver with linear_multinomial learner function
  metalearner <- make_learner(
    Lrnr_solnp,
    learner_function = metalearner_linear_multinomial, 
    eval_function = loss_loglik_multinomial
  )
  
  # Create SuperLearner with proper configuration
  sl <- Lrnr_sl$new(
    learners = learner_stack,
    metalearner = metalearner,
    keep_extra = FALSE,
    cv_folds = n.folds
  )
  
  return(sl)
}

# Standard learner stacks for binary outcomes
learner_stack_A_bin <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 1, family = "binomial"),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 0.5, family = "binomial"),
  list("Lrnr_glm", family = binomial()), # Use binomial() function, not strin
  list("Lrnr_mean")
)

learner_stack_Y <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 1, family = "binomial"),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 0.5, family = "binomial"),
  list("Lrnr_glm", family = binomial()), # Use binomial() function, not string
  list("Lrnr_mean")
)

learner_stack_Y_cont <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 1, family = "gaussian"),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 0.5, family = "gaussian"),
  list("Lrnr_glm", family = gaussian()), # Use gaussian() function
  list("Lrnr_mean")
)

# Standard metalearners
metalearner_Y <- make_learner(Lrnr_solnp, learner_function = metalearner_logistic_binomial, 
                              eval_function = loss_loglik_binomial)
metalearner_Y_cont <- make_learner(Lrnr_solnp, learner_function = metalearner_linear, 
                                  eval_function = loss_squared_error)
metalearner_A_bin <- make_learner(Lrnr_solnp, learner_function = metalearner_logistic_binomial, 
                                 eval_function = loss_loglik_binomial)