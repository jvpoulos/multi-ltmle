###########################################################################
# Super Learner functions for sl3                                         #
###########################################################################

# Super Learner for the multinomial treatment model: ranger, lasso and elastic-net multinomial
# regressions combined by a multinomial log-likelihood metalearner.
# n.folds: cross-validation folds. Returns an untrained sl3 Lrnr_sl object.
create_treatment_model_sl <- function(n.folds = 3) {
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

# Learner stacks for binary outcomes (outcome regressions and the one-vs-rest treatment and
# censoring models) and for continuous outcomes in [0, 1] (pseudo-outcomes of the recursion)
learner_stack_Y <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 1, family = "binomial"),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 0.5, family = "binomial"),
  list("Lrnr_glm", family = binomial()),
  list("Lrnr_mean")
)

learner_stack_Y_cont <- make_learner_stack(
  list("Lrnr_ranger", num.trees = 100),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 1, family = "gaussian"),
  list("Lrnr_glmnet", nfolds = 3, nlambda = 10, alpha = 0.5, family = "gaussian"),
  list("Lrnr_glm", family = gaussian()),
  list("Lrnr_mean")
)

# Metalearners: logistic (binary outcomes) and linear (continuous outcomes) combinations
metalearner_Y <- make_learner(Lrnr_solnp, learner_function = metalearner_logistic_binomial,
                              eval_function = loss_loglik_binomial)
metalearner_Y_cont <- make_learner(Lrnr_solnp, learner_function = metalearner_linear,
                                   eval_function = loss_squared_error)
