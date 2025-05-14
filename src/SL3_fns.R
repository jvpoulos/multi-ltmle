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

# Add the original_subset_covariates function that is referenced in Stack class
# This function will be used instead of the default sl3 implementation
original_subset_covariates <- function(task) {
  # This is a replacement for Stack$subset_covariates method
  # which is called when predicting with the Stack
  
  # Get covariates from the task
  all_covariates <- task$nodes$covariates
  
  # Check if all covariates exist in the task
  X_dt <- task$X
  task_covariates <- colnames(X_dt)
  
  # Find missing covariates
  missing_covariates <- setdiff(all_covariates, task_covariates)
  
  # If there are missing covariates, add them to the task data
  if (length(missing_covariates) > 0) {
    # Create a new data.table with the missing covariates
    missing_cols <- data.table::data.table(matrix(0, nrow = task$nrow, ncol = length(missing_covariates)))
    data.table::setnames(missing_cols, missing_covariates)
    
    # Add the missing columns to X_dt
    X_dt <- cbind(X_dt, missing_cols)
  }
  
  # Return the task with all covariates
  return(X_dt[, all_covariates, with = FALSE])
}

# Fix the subset_covariates error directly by adding it to global environment
# so it can be found when needed
attach_subset_covariates <- function() {
  # Make sure original_subset_covariates is in the global environment
  assign("original_subset_covariates", original_subset_covariates, envir = .GlobalEnv)
  message("original_subset_covariates function attached to global environment")
}

# Attach the function to the global environment immediately
tryCatch({
  attach_subset_covariates()
}, error = function(e) {
  message("Error attaching original_subset_covariates: ", e$message)
})
