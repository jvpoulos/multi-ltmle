######################
# Misc. functions    #
######################

# Function to fill NA values in estimate matrices
fill_na_estimates <- function(est_matrix, use_interpolation=TRUE) {
  # Check if the matrix has any NA values
  if (!any(is.na(est_matrix))) {
    return(est_matrix)  # No NAs, return as is
  }
  
  # Process each row (treatment rule)
  for (i in 1:nrow(est_matrix)) {
    row_vals <- est_matrix[i,]
    na_indices <- which(is.na(row_vals))
    
    # Skip if no NAs in this row
    if (length(na_indices) == 0) next
    
    # If all values are NA, use default value
    if (length(na_indices) == length(row_vals)) {
      est_matrix[i,] <- 0.5  # Default value
      next
    }
    
    # Handle partial NAs with interpolation as fallback
    if (use_interpolation) {
      non_na_indices <- which(!is.na(row_vals))
      
      for (j in na_indices) {
        # Find nearest non-NA values
        left_idx <- max(non_na_indices[non_na_indices < j], -Inf)
        right_idx <- min(non_na_indices[non_na_indices > j], Inf)
        
        if (left_idx == -Inf && right_idx == Inf) {
          # No non-NA values found, use default
          est_matrix[i,j] <- 0.5
        } else if (left_idx == -Inf) {
          # Only right value available, use nearest neighbor
          est_matrix[i,j] <- est_matrix[i,right_idx]
        } else if (right_idx == Inf) {
          # Only left value available, use nearest neighbor
          est_matrix[i,j] <- est_matrix[i,left_idx]
        } else {
          # Both left and right values available, interpolate
          # Calculate weights based on distance
          left_weight <- 1 - (j - left_idx) / (right_idx - left_idx)
          right_weight <- 1 - (right_idx - j) / (right_idx - left_idx)
          
          # Linear interpolation
          est_matrix[i,j] <- left_weight * est_matrix[i,left_idx] + 
            right_weight * est_matrix[i,right_idx]
        }
      }
    }
  }
  
  return(est_matrix)
}

# Add this function to ensure proper matrix dimensions
ensure_matrix_dimensions <- function(mat, nrow=3, ncol=36) {
  if(is.null(mat)) {
    # Create a properly dimensioned matrix with NA values
    result <- matrix(NA, nrow=nrow, ncol=ncol)
    rownames(result) <- c("static", "dynamic", "stochastic")[1:nrow]
    return(result)
  }
  
  if(!is.matrix(mat)) {
    # Convert vector to matrix
    if(is.vector(mat)) {
      if(length(mat) == nrow*ncol) {
        # Reshape to matrix with the right dimensions
        result <- matrix(mat, nrow=nrow, ncol=ncol)
      } else if(length(mat) == nrow) {
        # Replicate vector across columns
        result <- matrix(rep(mat, each=ncol), nrow=nrow, ncol=ncol)
      } else if(length(mat) == ncol) {
        # Replicate vector across rows
        result <- matrix(rep(mat, times=nrow), nrow=nrow, ncol=ncol, byrow=TRUE)
      } else {
        # Create uniform matrix
        result <- matrix(NA, nrow=nrow, ncol=ncol)
      }
    } else {
      # Create uniform matrix for non-vector, non-matrix
      result <- matrix(NA, nrow=nrow, ncol=ncol)
    }
    rownames(result) <- c("static", "dynamic", "stochastic")[1:nrow]
    return(result)
  }
  
  # Handle matrix with wrong dimensions
  if(nrow(mat) != nrow || ncol(mat) != ncol) {
    result <- matrix(NA, nrow=nrow, ncol=ncol)
    # Copy as much data as possible
    r_copy <- min(nrow, nrow(mat))
    c_copy <- min(ncol, ncol(mat))
    result[1:r_copy, 1:c_copy] <- mat[1:r_copy, 1:c_copy]
    rownames(result) <- c("static", "dynamic", "stochastic")[1:nrow]
    return(result)
  }
  
  # Matrix already has correct dimensions
  if(is.null(rownames(mat))) {
    rownames(mat) <- c("static", "dynamic", "stochastic")[1:nrow]
  }
  return(mat)
}

# Safe reshaping function
safe_reshape_data <- function(data, t_end, debug) {
  if(debug){
    # Debug output
    print("Data column names at start:")
    print(names(data))
  }
  
  # Get unique IDs
  unique_ids <- unique(data$ID)
  n_ids <- length(unique_ids)
  
  # Ensure A column exists and is properly formatted
  if(!"A" %in% names(data)) {
    print("Looking for alternative treatment columns...")
    A_cols <- grep("^A[0-9]$", names(data), value=TRUE)
    if(length(A_cols) > 0) {
      print(paste("Found treatment columns:", paste(A_cols, collapse=", ")))
      # Use first treatment column if multiple exist
      data$A <- data[[A_cols[1]]]
    } else {
      stop("No treatment column found in data")
    }
  }
  
  # Convert treatment to factor if not already
  data$A <- factor(data$A)
  
  # Get unique IDs
  unique_ids <- unique(data$ID)
  n_ids <- length(unique_ids)
  
  # Create time sequence
  time_seq <- 0:t_end
  
  # Create expanded grid of IDs and times
  wide_base <- expand.grid(
    ID = unique_ids,
    t = time_seq,
    stringsAsFactors = FALSE
  )
  
  # Sort by ID and time
  wide_base <- wide_base[order(wide_base$ID, wide_base$t), ]
  
  # Merge with original data
  wide_data <- merge(wide_base, data, by = c("ID", "t"), all.x = TRUE)
  
  # Get time-varying columns
  time_varying_cols <- c("L1", "L2", "L3", "A", "C", "Y")
  
  # Reshape each time-varying variable
  wide_reshaped <- wide_data[!duplicated(wide_data$ID), setdiff(names(wide_data), c("t", time_varying_cols))]
  
  for(col in time_varying_cols) {
    if(col %in% names(wide_data)) {
      temp_data <- wide_data[, c("ID", "t", col)]
      temp_wide <- reshape(temp_data,
                           timevar = "t",
                           idvar = "ID",
                           direction = "wide",
                           sep = ".")
      
      # Add reshaped columns
      wide_cols <- grep(paste0("^", col, "\\."), names(temp_wide), value = TRUE)
      if(length(wide_cols) > 0) {
        wide_reshaped[wide_cols] <- temp_wide[, wide_cols]
      }
    }
  }
  
  # Handle missing values
  na_cols <- c(
    grep("^Y\\.", names(wide_reshaped), value = TRUE),
    grep("^C\\.", names(wide_reshaped), value = TRUE),
    grep("^L[1-3]\\.", names(wide_reshaped), value = TRUE),
    grep("^V3\\.", names(wide_reshaped), value = TRUE)
  )
  
  wide_reshaped[na_cols][is.na(wide_reshaped[na_cols])] <- -1
  
  # Handle treatment columns
  A_cols <- grep("^A\\.", names(wide_reshaped), value = TRUE)
  if(length(A_cols) > 0) {
    wide_reshaped[A_cols] <- lapply(wide_reshaped[A_cols], function(x) {
      if(is.factor(x)) {
        x <- addNA(x)
        levels(x) <- c("0", levels(x)[-length(levels(x))])
      } else {
        x <- factor(x)
        x <- addNA(x)
        levels(x) <- c("0", levels(x)[-length(levels(x))])
      }
      as.numeric(as.character(x))
    })
  }
  
  return(wide_reshaped)
}

create_standard_matrix <- function(data, expected_rows, expected_cols) {
  # Handle NULL or empty data
  if(is.null(data) || length(data) == 0) {
    mat <- matrix(1/expected_cols, nrow=expected_rows, ncol=expected_cols)
    colnames(mat) <- paste0("A", 1:expected_cols)
    return(mat)
  }
  
  # Convert list to matrix if needed with better error handling
  if(is.list(data) && !is.data.frame(data)) {
    tryCatch({
      data <- do.call(rbind, data)
    }, error = function(e) {
      # Fall back to uniform matrix on conversion error
      data <- matrix(1/expected_cols, nrow=expected_rows, ncol=expected_cols)
    })
  }
  
  # Ensure data is a matrix with correct dimensions
  if(!is.matrix(data)) {
    tryCatch({
      # First convert to numeric to avoid complex conversions
      data <- as.numeric(data)
      # If data is a vector, reshape it properly
      if(length(data) > 0) {
        if(length(data) >= expected_rows * expected_cols) {
          # Reshape to matrix with proper dimensions
          data <- matrix(data[1:(expected_rows * expected_cols)], 
                         nrow=expected_rows, ncol=expected_cols)
        } else if(length(data) >= expected_rows) {
          # Fill with the data we have
          temp_matrix <- matrix(1/expected_cols, nrow=expected_rows, ncol=expected_cols)
          # Fill first column with data
          temp_matrix[1:min(length(data), expected_rows), 1] <- data[1:min(length(data), expected_rows)]
          data <- temp_matrix
        } else {
          # Repeat data to fill matrix
          temp_matrix <- matrix(1/expected_cols, nrow=expected_rows, ncol=expected_cols)
          temp_matrix[1:expected_rows, 1] <- rep(data, length.out=expected_rows)
          data <- temp_matrix
        }
      } else {
        # Empty numeric vector - create default matrix
        data <- matrix(1/expected_cols, nrow=expected_rows, ncol=expected_cols)
      }
    }, error = function(e) {
      # Fall back to uniform matrix if conversion fails
      data <- matrix(1/expected_cols, nrow=expected_rows, ncol=expected_cols)
    })
  }
  
  # Fix dimensions if needed
  if(nrow(data) != expected_rows || ncol(data) != expected_cols) {
    # Simply create new matrix with correct dimensions
    mat <- matrix(1/expected_cols, nrow=expected_rows, ncol=expected_cols)
    
    # If data has some content, try to copy parts of it
    if(nrow(data) > 0 && ncol(data) > 0) {
      # Copy as much as possible from original data
      max_rows <- min(nrow(data), expected_rows)
      max_cols <- min(ncol(data), expected_cols)
      mat[1:max_rows, 1:max_cols] <- data[1:max_rows, 1:max_cols]
    }
    data <- mat
  }
  
  # Check if data is a matrix before setting column names
  if(!is.matrix(data)) {
    # If somehow still not a matrix, create one
    data <- matrix(1/expected_cols, nrow=expected_rows, ncol=expected_cols)
  }
  
  # Ensure column names
  colnames(data) <- paste0("A", 1:expected_cols)
  
  return(data)
}

# from weights package
dummify <- function(x, show.na=FALSE, keep.na=FALSE){
  if(!is.factor(x)){
    stop("variable needs to be a factor")
  }
  levels(x) <- c(levels(x), "NAFACTOR")
  x[is.na(x)] <- "NAFACTOR"
  levs <- levels(x)
  out <- model.matrix(~x-1)
  colnames(out) <- levs
  attributes(out)$assign <- NULL
  attributes(out)$contrasts <- NULL
  if(show.na==FALSE)
    out <- out[,(colnames(out)=="NAFACTOR")==FALSE]
  if(keep.na==TRUE)
    out[x=="NAFACTOR",] <- matrix(NA, sum(x=="NAFACTOR"), dim(out)[2])
  out
}

# function to bound probabilities to be used when making predictions
boundProbs <- function(x, bounds=c(0.025,1)){
  tryCatch({
    # Create a safe matrix with guaranteed structure
    if (is.null(x) || !is.numeric(x)) {
      # Return a 1x1 matrix with the lower bound if input is NULL or non-numeric
      result <- matrix(min(bounds), nrow=1, ncol=1)
      colnames(result) <- "A1"
      return(result)
    }
    
    # Handle vectors
    if (is.vector(x)) {
      x <- matrix(x, ncol=1)
    }
    
    # Handle non-matrix inputs that could be coerced
    if (!is.matrix(x)) {
      x <- try(as.matrix(x), silent=TRUE)
      if (inherits(x, "try-error")) {
        # If conversion fails, return a matrix of lower bounds
        result <- matrix(min(bounds), nrow=1, ncol=1)
        colnames(result) <- "A1"
        return(result)
      }
    }
    
    # Get dimensions explicitly
    nr <- nrow(x)
    nc <- ncol(x)
    
    # Save original column names if they exist
    orig_colnames <- if (!is.null(colnames(x))) colnames(x) else paste0("A", 1:nc)
    
    # Create a new matrix with known dimensions
    result <- matrix(0, nrow=nr, ncol=nc)
    
    # Copy data with bounds - simply apply bounds regardless of column count
    for (i in 1:nr) {
      for (j in 1:nc) {
        val <- x[i, j]
        # Apply bounds
        if (is.na(val) || !is.finite(val)) {
          result[i, j] <- min(bounds)
        } else if (val < min(bounds)) {
          result[i, j] <- min(bounds)
        } else if (val > max(bounds)) {
          result[i, j] <- max(bounds)
        } else {
          result[i, j] <- val
        }
      }
    }
    
    # Only perform row normalization for multi-column matrices (treatment probabilities)
    if (nc > 1) {
      for (i in 1:nr) {
        row_sum <- sum(result[i,])
        if (row_sum > 0 && row_sum != 1) {
          result[i,] <- result[i,] / row_sum
        }
        
        # Re-apply bounds after normalization
        for (j in 1:nc) {
          if (result[i,j] < min(bounds)) result[i,j] <- min(bounds)
          if (result[i,j] > max(bounds)) result[i,j] <- max(bounds)
        }
        
        # Final normalization if needed
        if (sum(result[i,]) > 0 && sum(result[i,]) != 1) {
          result[i,] <- result[i,] / sum(result[i,])
        }
      }
    }
    
    # Assign original column names to the result matrix
    colnames(result) <- orig_colnames
    
    return(result)
  }, error = function(e) {
    # Ultimate fallback - return a matrix with reasonable dimensions
    nr <- if (exists("nr") && is.numeric(nr) && nr > 0) nr else 1
    nc <- if (exists("nc") && is.numeric(nc) && nc > 0) nc else 1
    
    fallback <- matrix(min(bounds), nrow=nr, ncol=nc)
    
    # Try to use original column names
    if (exists("orig_colnames") && length(orig_colnames) == nc) {
      colnames(fallback) <- orig_colnames
    } else {
      colnames(fallback) <- paste0("A", 1:nc)
    }
    
    return(fallback)
  })
}

# proper characters
proper <- function(s) sub("(.)", ("\\U\\1"), tolower(s), pe=TRUE)

# survival plot (from simcausal)
plotSurvEst <- function (surv = list(), xindx = NULL, ylab = "", xlab = "t", 
                         ylim = c(0, 1), legend.xyloc = "topright", ...){
  ptsize <- 1
  counter <- 0
  for (d.j in names(surv)) {
    counter <- counter + 1
    if (is.null(xindx)) 
      xindx <- seq(surv[[d.j]])
    plot(x = xindx, y = surv[[d.j]][xindx], col = counter, 
         type = "b", cex = ptsize, ylab = ylab, xlab = xlab, 
         ylim = ylim, ...)
    par(new = TRUE)
  }
  if(!is.null(legend.xyloc)){
    legend(legend.xyloc, legend = names(surv), col = c(1:length(names(surv))), 
           cex = ptsize, pch = 1)
  }
}

# function to calculate confidence intervals
CI <- function(est,infcurv,alpha=0.05){
  infcurv <- na.omit(infcurv)
  n <- length(infcurv)
  ciwidth <- qnorm(1-(alpha/2)) * sqrt(var(infcurv,na.rm = TRUE)/sqrt(n))
  CI <- c(est-ciwidth, est+ciwidth)
  return(CI)
}