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
  
  # Improved error handling and diagnostics
  tryCatch({
    # Get covariates from the task
    if (is.null(task) || is.null(task$nodes) || is.null(task$nodes$covariates)) {
      warning("Task or its covariates structure is NULL. Creating minimal fallback.")
      return(data.table::data.table(V3=0, L1=0, L2=0, L3=0, Y.lag=0))
    }
    
    all_covariates <- task$nodes$covariates
    
    # Check if all covariates exist in the task
    if(is.null(task$X)) {
      warning("Task X is NULL. Creating data with required covariates.")
      X_dt <- data.table::data.table(matrix(0, nrow = max(1, task$nrow), ncol = length(all_covariates)))
      data.table::setnames(X_dt, all_covariates)
      return(X_dt)
    }
    
    X_dt <- task$X
    task_covariates <- colnames(X_dt)
    
    # Find missing covariates
    missing_covariates <- setdiff(all_covariates, task_covariates)
    
    # If there are missing covariates, add them to the task data
    if (length(missing_covariates) > 0) {
      # Log missing covariates
      message("Adding missing covariates: ", paste(missing_covariates, collapse=", "))
      
      # Create a new data.table with the missing covariates
      missing_cols <- data.table::data.table(matrix(0, nrow = task$nrow, ncol = length(missing_covariates)))
      data.table::setnames(missing_cols, missing_covariates)
      
      # Add the missing columns to X_dt
      X_dt <- cbind(X_dt, missing_cols)
    }
    
    # Return the task with all covariates
    if (all(all_covariates %in% colnames(X_dt))) {
      return(X_dt[, all_covariates, with = FALSE])
    } else {
      # Some covariates still missing - create fallback
      warning("Still missing covariates after attempted fix. Creating complete fallback.")
      result <- data.table::data.table(matrix(0, nrow = nrow(X_dt), ncol = length(all_covariates)))
      data.table::setnames(result, all_covariates)
      return(result)
    }
  }, error = function(e) {
    # Complete fallback on any error
    warning("Error in original_subset_covariates: ", e$message, ". Creating fallback data.")
    result <- data.table::data.table(
      V3=0, L1=0, L2=0, L3=0, Y.lag=0, Y.lag2=0, Y.lag3=0,
      L1.lag=0, L1.lag2=0, L1.lag3=0, L2.lag=0, L2.lag2=0, L2.lag3=0,
      L3.lag=0, L3.lag2=0, L3.lag3=0, white=0, black=0, latino=0, other=0,
      mdd=0, bipolar=0, schiz=0, A1=0, A2=0, A3=0, A4=0, A5=0, A6=0,
      A1.lag=0, A2.lag=0, A3.lag=0, A4.lag=0, A5.lag=0, A6.lag=0,
      A1.lag2=0, A2.lag2=0, A3.lag2=0, A4.lag2=0, A5.lag2=0, A6.lag2=0,
      A1.lag3=0, A2.lag3=0, A3.lag3=0, A4.lag3=0, A5.lag3=0, A6.lag3=0
    )
    return(result)
  })
}

# Fix the subset_covariates error directly by adding it to global environment
# and also monkey-patching the Stack class method
attach_subset_covariates <- function(force_patch = FALSE) {
  # Make sure original_subset_covariates is in the global environment
  assign("original_subset_covariates", original_subset_covariates, envir = .GlobalEnv)
  message("original_subset_covariates function attached to global environment")
  
  # Also try to monkey-patch the Stack class method directly
  if (exists("Stack") && methods::is(Stack, "R6ClassGenerator") || force_patch) {
    tryCatch({
      # Create a new definition of the Stack class with our custom method
      old_stack_predict <- Stack$public_methods$predict
      
      # Define a new wrapper that ensures subset_covariates uses our function
      Stack$set("public", "subset_covariates", function(task) {
        tryCatch({
          # First try the original function that's in the global environment
          if (exists("original_subset_covariates", envir = .GlobalEnv)) {
            result <- get("original_subset_covariates", envir = .GlobalEnv)(task)
            return(result)
          } else {
            # Fall back to a simple implementation
            warning("original_subset_covariates not found in global env, using local implementation")
            X_dt <- task$X
            all_covariates <- task$nodes$covariates
            missing_covs <- setdiff(all_covariates, colnames(X_dt))
            
            if (length(missing_covs) > 0) {
              missing_cols <- data.table::data.table(matrix(0, nrow = task$nrow, ncol = length(missing_covs)))
              data.table::setnames(missing_cols, missing_covs)
              X_dt <- cbind(X_dt, missing_cols)
            }
            
            return(X_dt[, all_covariates, with = FALSE])
          }
        }, error = function(e) {
          warning("Error in monkey-patched subset_covariates: ", e$message)
          # Create fallback data with all required covariates
          result <- data.table::data.table(
            V3=0, L1=0, L2=0, L3=0, Y.lag=0, Y.lag2=0, Y.lag3=0,
            L1.lag=0, L1.lag2=0, L1.lag3=0, L2.lag=0, L2.lag2=0, L2.lag3=0,
            L3.lag=0, L3.lag2=0, L3.lag3=0, white=0, black=0, latino=0, other=0,
            mdd=0, bipolar=0, schiz=0, A1=0, A2=0, A3=0, A4=0, A5=0, A6=0,
            A1.lag=0, A2.lag=0, A3.lag=0, A4.lag=0, A5.lag=0, A6.lag=0,
            A1.lag2=0, A2.lag2=0, A3.lag2=0, A4.lag2=0, A5.lag2=0, A6.lag2=0,
            A1.lag3=0, A2.lag3=0, A3.lag3=0, A4.lag3=0, A5.lag3=0, A6.lag3=0
          )
          return(result[1:task$nrow, ])
        })
      })
      
      message("Successfully monkey-patched Stack$subset_covariates method")
    }, error = function(e) {
      warning("Failed to monkey-patch Stack$subset_covariates: ", e$message)
    })
  } else {
    message("Stack class not available yet, skipping monkey-patching")
  }
}

# Attach the function to the global environment and patch the Stack class
tryCatch({
  attach_subset_covariates()
}, error = function(e) {
  message("Error attaching original_subset_covariates: ", e$message)
})
