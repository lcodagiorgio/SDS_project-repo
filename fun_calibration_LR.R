
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
  
  # Assign bin labels based on the partition result
  for (i in seq_along(partition_result)) {
    bin_labels[partition_result[[i]]] <- i
  }
  
  # Return the bin labels
  return(as.factor(bin_labels))
}



# Function to return equal-width or data-dependent bins
partition_preds <- function(preds, num_bins, strategy, threshold) {
  if (strategy == "width") {
    breaks <- seq(0, 1, length.out = num_bins + 1)
    labels <- seq(1, num_bins)
    bins <- cut(preds, breaks = breaks, labels = labels, include.lowest = T, right = F)
    
    return(bins)
  }
  
  if (strategy == "data") {
    bins <- assign_bins(preds, threshold)
    
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



#-----------------------------   Calibration   ---------------------------------



# Function to compute r_hat
# (frequencies of first class (-1) in a bin divided by the size of the bin)
r_hat <- function(preds, labels, bins) {
  df <- data.frame(preds = preds, true_labels = labels, bins = bins)
  
  counts <- df %>%
    group_by(bins) %>%
    summarise(class_freq = sum(true_labels == -1),
              tot_instances = n()
              ) %>%
    mutate(rhat = if (n() == 0) rep(0, length(table(bins))) else class_freq/tot_instances
           ) %>%
    pull(rhat)
  
  return(counts)
}



# Function to compute g_hat (mean predictions for each class in a bin)
g_hat <- function(preds, bins) {
  df <- data.frame(preds = preds, bins = bins)
  
  mean_preds <- df %>%
    group_by(bins) %>%
    summarise(ghat = if (n() == 0) rep(0, length(table(bins))) else mean(preds)
              ) %>% 
    pull(ghat)
  
  return(mean_preds)
}



# Function to compute p_hat (weight of the bin in terms its obs wrt the total)
p_hat <- function(preds, bins) {
  bin_sizes <- table(bins)
  total_obs <- length(bins)
  
  if (total_obs == 0) {
    phat <- rep(0, length(bin_sizes))
  } else phat <- bin_sizes / total_obs
  
  return(as.numeric(phat))
}



# Function to compute eta_hat
# (bin-sized weighted sum of the distances between r_hat and g_hat)
miscal_estimates <- function(preds, labels, num_bins, bin_strategy, threshold, dist_fun = tvd) {
  bins <- partition_preds(preds, num_bins, bin_strategy, threshold)
  
  rhat <- r_hat(preds, labels, bins)
  ghat <- g_hat(preds, bins)
  phat <- p_hat(preds, bins)
  
  deviation <- rhat - ghat
  
  # rhat[i] is the observed probability of class -1 within ith bin
  # ghat[i] represents the average predicted probability of class -1 within ith bin
  tdv_vals <- sapply(seq_along(rhat), function(i) dist_fun(c(rhat[i], 1-rhat[i]),
                                                           c(ghat[i], 1-ghat[i])))
  etahat <- sum(phat * tdv_vals)
  
  return(list(eta = etahat, r = rhat, g = ghat, p = phat, dev = deviation, bins = bins))
}



# --------------------   Miscalibration statistics   ---------------------------



# Compute mean, 5th and 95 percentile of the deviations for each bin for bootstrap resamples of the generated data
# Compute the data's eta, r, g, p and frequencies for every bin
compute_miscal_stats <- function(model, num_bins, bin_strategy, threshold = 1000,
                                 dist_fun = tvd, num_samples = 10000, num_resamples = 100) {
  # Generate original data
  original <- generate_data_gmm(num_samples)
  # Compute the predictions for the model on X
  original$pred <- sapply(original$X, model)
  
  # Compute estimates on the original data
  original_est <- miscal_estimates(original$pred, original$Y, num_bins, bin_strategy, threshold)
  freq_bins <- table(original_est$bins)
  num_bins <- length(levels(original_est$bins))
  
  # Perform bootstrapping to generate resampled datasets
  res_datasets <- replicate(num_resamples, {
    # Resample the data
    resampled_indices <- sample(nrow(original), replace = T)
    resampled_data <- original[resampled_indices, ]
    
    # Add predicted values to resampled data
    resampled_data$pred <- sapply(resampled_data$X, model)
    
    return(resampled_data)
    
  }, simplify = F)
  
  
  # Compute miscalibration estimates for each resampled data
  res_estimates <- lapply(res_datasets, function(s) {
    labels = s$Y
    miscal_estimates(s$pred, labels, num_bins, bin_strategy, threshold)
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



