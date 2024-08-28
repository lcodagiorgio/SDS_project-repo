
# Function to exploit chaining methods
library(dplyr)


#------------------------   Multi-purpose functions   --------------------------



# Function for recursive data-dependent binning of predictions
recursive_partition <- function(preds, indices, threshold) {
  # If the number of indices <= threshold return the indices as a single bin
  if (length(indices) <= threshold) {
    return(list(indices))
  }
  
  # Calculate the mean of the predictions for the given indices
  split_val <- mean(preds[indices])
  
  # Split the indices into two groups based on the mean value
  ind_left <- indices[preds[indices] <= split_val]
  ind_right <- indices[preds[indices] > split_val]
  
  # If either split is empty, return the indices as a single bin
  if (length(ind_left) == 0 || length(ind_right) == 0) {
    return(list(indices))
  }
  
  # Partition the left and right groups
  bins_left <- recursive_partition(preds, ind_left, threshold)
  bins_right <- recursive_partition(preds, ind_right, threshold)
  
  # Combine the bins from the left and right partitions
  return(c(bins_left, bins_right))
}



# Function to assign bins based on recursive partitioning
assign_bins <- function(preds, threshold) {
  # Generate a sequence of indices corresponding to the predictions
  indices <- seq_along(preds)
  
  # Use the recursive_partition function to get the list of bins
  partition_result <- recursive_partition(preds, indices, threshold)
  
  # Initialize a vector to store the bin labels for each prediction
  bin_labels <- rep(NA, length(preds))
  
  # Assign bin labels based on the partition_result
  for (i in seq_along(partition_result)) {
    bin_labels[partition_result[[i]]] <- i
  }
  
  # Return the bin labels
  return(as.factor(bin_labels))
}



# Function to return equal-width or data-dependent bins
partition_preds <- function(preds, num_bins, strategy, threshold) {
  # Get the maximum predicted probability for each instance
  confidences <- apply(preds, 1, max)
  
  if (strategy == "width") {
    breaks <- seq(0, 1, length.out = num_bins + 1)
    labels <- seq(1, num_bins)
    bins <- cut(confidences, breaks = breaks, labels = labels, include.lowest = TRUE, right = FALSE)
    return(bins)
  }
  
  if (strategy == "data") {
    bins <- assign_bins(confidences, threshold)
    return(bins)
  }
}



# Function to compute total variation distance between two distributions
tvd <- function(dist_1, dist_2) {
  # Raising error if the distributions have different length
  if(length(dist_1) != length(dist_2)) {
    stop("The two distributions must be of the same length!")
  }
  
  tvdist <- 0.5*sum(abs(dist_1 - dist_2))
  
  return(tvdist)
}



# Function to compute the squared Euclidean distance
euclid <- function(dist_1, dist_2) {
  
  return(sqrt(sum((dist_1 - dist_2)^2)))
}



#-----------------------------   Calibration   ---------------------------------



# Function to compute r_hat
r_hat <- function(preds, labels, bins) {
  # Maximum class for each instance
  max_class <- apply(preds, 1, which.max) - 1
  correct_preds <- (labels == max_class)
  
  bin_levels <- levels(bins)
  # Initialize a 0 vector to store rhat
  rhat <- numeric(length(bin_levels))
  
  for (i in seq_along(bin_levels)) {
    bin_idx <- which(bins == bin_levels[i])
    
    if (length(bin_idx) > 0) {
      rhat[i] <- mean(correct_preds[bin_idx])
    } else rhat[i] <- 0
  }
  
  return(rhat)
}



# Function to compute p_hat
p_hat <- function(preds, bins) {
  bin_sizes <- table(bins)
  total_obs <- length(bins)
  
  if (total_obs == 0) {
    phat <- rep(0, length(bin_sizes))
  } else phat <- bin_sizes/total_obs
  
  return(as.numeric(phat))
}



# Function to compute g_hat
g_hat <- function(preds, bins) {
  max_preds <- apply(preds, 1, max)
  
  bins_lev <- levels(bins)
  # Initialize a 0 vector to store ghat
  ghat <- numeric(length(bins_lev))
  
  for (i in seq_along(bins_lev)) {
    bin_idx <- which(bins == bins_lev[i])
    
    if (length(bin_idx) > 0) {
      ghat[i] <- mean(max_preds[bin_idx])
    } else ghat[i] <- 0
  }
  
  return(ghat)
}



# Function to compute eta_hat (bin-sized weighted sum of the distances between r_hat and g_hat)
miscal_estimates <- function(preds, labels, num_bins, bin_strategy, threshold, dist_fun) {
  bins <- partition_preds(preds, num_bins, bin_strategy, threshold)
  
  rhat <- r_hat(preds, labels, bins)
  ghat <- g_hat(preds, bins)
  phat <- p_hat(preds, bins)
  
  deviation <- rhat - ghat
  
  tdv_vals <- sapply(seq_along(rhat), function(i) dist_fun(c(rhat[i], 1-rhat[i]),
                                                           c(ghat[i], 1-ghat[i])))
  
  etahat <- sum(phat * tdv_vals)
  
  return(list(eta = etahat, r = rhat, g = ghat, p = phat, dev = deviation, bins = bins))
}



# --------------------   Miscalibration statistics   ---------------------------



# Compute mean, 5th and 95 percentile of the deviations for each bin for bootstrap resamples of the generated data
# Compute the data's eta, r, g, p and frequencies for every bin
compute_miscal_stats <- function(model, num_bins, bin_strategy, threshold = 1000,
                                 dist_fun, num_resamples = 100) {
  # Generate original data
  data <- generate_data_mnist()
  
  # Extract the x_test and y_test from the data
  original_x <- data$x_test
  original_y <- data$y_test
  # Convert the One-Hot encoded labels in true labels
  true_labels <- apply(original_y, 1, which.max)-1
  
  # Compute the predictions for the model on x_test
  original_preds <- predict(model, original_x)

  # Compute estimates on the original data
  original_est <- miscal_estimates(original_preds, true_labels, num_bins, bin_strategy, threshold, dist_fun)
  freq_bins <- table(original_est$bins)
  num_bins <- length(levels(original_est$bins))
  
  # Perform bootstrapping to generate resampled datasets
  res_datasets <- replicate(num_resamples, {
    # Resample the data
    nrows <- dim(original_x)[1]
    res_indices <- sample(nrows, replace = T)
    res_x <- original_x[res_indices, , , , drop = F]
    res_y <- original_y[res_indices, , drop = F]
    res_lab <- apply(res_y, 1, which.max) - 1
    # Predictions for the resampled data
    res_preds <- predict(model, res_x)
    
    return(list(x_test = res_x, y_test = res_y, pred = res_preds))
    
  }, simplify = F)
  
  
  # Compute miscalibration estimates for each resampled data
  res_estimates <- lapply(res_datasets, function(s) {
    s_y = s$y_test
    labels <- apply(s_y, 1, which.max)-1
    miscal_estimates(s$pred, labels, num_bins, bin_strategy, threshold, dist_fun)
  })
  
  # Keep only the resampled estimates that have the same number of bins as the original data
  res_estimates <- Filter(function(est) length(levels(est$bins)) == num_bins, res_estimates)
  
  # Initialize a matrix of 0s to store the results,
  # since we need to keep the number of bins even if some are filled with NA
  res_deviations <- matrix(0, nrow = num_bins, ncol = length(res_estimates))
  
  # Fill in the deviations for each resampled data
  for (i in seq_along(res_estimates)) {
    dev_values <- res_estimates[[i]]$dev
    # Assign the deviations to the ith column of the matrix, replace with 0 if NA
    res_deviations[, i] <- ifelse(is.na(dev_values), 0, dev_values)
  }
  
  # Convert the deviations' matrix to a data frame
  res_deviations <- as.data.frame(res_deviations)
  
  # Compute statistics across resamples for each bin
  mean_dev <- rowMeans(res_deviations, na.rm = T)
  pct5_dev <- apply(res_deviations, 1, function(x) quantile(x, probs = 0.05, na.rm = T))
  pct95_dev <- apply(res_deviations, 1, function(x) quantile(x, probs = 0.95, na.rm = T))
  
  # Combine results into a data frame
  dev_stats <- data.frame(mean_dev = mean_dev,
                          pct5_dev = pct5_dev,
                          pct95_dev = pct95_dev)
  
  # Add original data estimates and frequency bins to the result
  dev_stats$data_dev <- original_est$dev
  dev_stats$data_eta <- original_est$eta
  dev_stats$data_r <- original_est$r
  dev_stats$data_g <- original_est$g
  dev_stats$data_p <- original_est$p
  dev_stats$data_freq_bins <- as.numeric(freq_bins)
  dev_stats$bins <- list(original_est$bins)
  
  # Set all rows to have zeros when data_freq_bins == 0
  dev_stats[dev_stats$data_freq_bins == 0, c("mean_dev", "pct5_dev", "pct95_dev",
                                             "data_dev", "data_eta", "data_r",
                                             "data_g", "data_p")] <- 0
  
  return(dev_stats)
}





