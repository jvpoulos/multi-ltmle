###########################################################################
# Super Learner functions for sl3                                         #
###########################################################################

create_treatment_model_sl <- function(n.folds = 5) {
  # Create ranger learner - keep this as it's more stable
  ranger1 <- Lrnr_ranger$new(
    num.trees = 100, 
    min.node.size = 5, 
    respect.unordered.factors = "partition", 
    num.threads = 1
  )
  
  # Add multinomial glmnet (similar to binomial glmnet in learner_stack_A_bin)
  multinomial_glmnet <- tryCatch({
    Lrnr_glmnet$new(
      family = "multinomial",
      alpha = 1,        # Same alpha as in learner_stack_A_bin
      nfolds = 3,       # Use 3 CV folds for faster fitting
      nlambda = 10      # Fewer lambda values for faster computation
    )
  }, error = function(e) {
    message("Cannot create multinomial glmnet: ", e$message)
    NULL
  })
  
  # Simple mean learner as safe fallback
  mean_lrnr <- Lrnr_mean$new()
  
  # Build stack with available learners
  learners <- list(ranger1, mean_lrnr)
  if (!is.null(multinomial_glmnet)) learners <- c(learners, multinomial_glmnet)
  
  stack <- Stack$new(learners)
  
  # Create Super Learner with mean metalearner
  # The mean metalearner is more robust when only some learners succeed
  sl <- Lrnr_sl$new(
    learners = stack,
    metalearner = mean_lrnr,
    keep_extra = FALSE   # Don't keep unnecessary data to reduce memory usage
  )
  
  return(sl)
}

# Standard learner stacks for binary outcomes
learner_stack_A_bin <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 1, family = "binomial"),
  list("Lrnr_glm", family = binomial()), # Use binomial() function, not strin
  list("Lrnr_mean")
)

learner_stack_Y <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 1, family = "binomial"),
  list("Lrnr_glm", family = binomial()), # Use binomial() function, not string
  list("Lrnr_mean")
)

learner_stack_Y_cont <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 1, family = "gaussian"),
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