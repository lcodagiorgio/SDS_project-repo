
# Importing our packages for the Logistic Regression experiments
source("fun_data_generation.R")
source("fun_calibration_LR.R")


#-----------------------------   Models   --------------------------------------



# Pre-trained Logistic Regression models for the experiments
log_reg <- function(x, beta_0, beta_1){
  prob <- 1 / (1 + exp(-(beta_0 + beta_1 * x)))
  
  return(prob)
}

# Perfect Model
perf_model <- function(x, beta_0 = 0, beta_1 = -2) {
  log_reg(x, beta_0, beta_1)
}

# Constant Model
const_model <- function(x, beta_0 = 0, beta_1 = 0) {
  log_reg(x, beta_0, beta_1)
}

# Uncalibrated Model
uncal_model <- function(x, beta_0 = 1, beta_1 = 1) {
  log_reg(x, beta_0, beta_1)
}

# Function to build all the models into a list
build_models <- function() list(perfect = perf_model,
                                constant = const_model,
                                uncalibrated = uncal_model)



#---------------------   Estimated Eta - True Eta (wrt n)   --------------------



# Function to compute the miscalibration varying sample size n,
# given a model, bin size and number of repetitions for each n
compute_diff_stats <- function(model, true_eta, nsamples_vector, num_bins,
                               bin_strategy = "width", threshold = 1000, repetitions = 1000) {
  # Initialize a data frame to store the results for each sample size
  results <- data.frame(nsamples = nsamples_vector,
                        mean_diff = numeric(length(nsamples_vector)),
                        pct5_diff = numeric(length(nsamples_vector)),
                        pct95_diff = numeric(length(nsamples_vector)))
  
  # Iterate over the ith number of samples in the vector
  for (i in seq_along(nsamples_vector)) {
    num_samples = nsamples_vector[i]
    
    # Store the Eta estimates for a number of repetitions (for same n)
    eta_estimates <- replicate(repetitions, {
      # Generate data
      data <- generate_data_gmm(num_samples)
      # Predictions
      data$pred <- sapply(data$X, model)
      # Compute estimates for eta, r, g and p
      est <- miscal_estimates(data$pred, data$Y, num_bins, bin_strategy, threshold, tvd)
      
      # Return only eta_hat
      return(est$eta)
    })
    
    # Compute difference between Estimated Eta and True Eta
    diff <- eta_estimates - true_eta
    
    # Store the mean, 5th and 95th percentiles for the differences for the respective n
    results[i, "mean_diff"] <- mean(diff)
    results[i, "pct5_diff"] <- quantile(diff, probs = 0.05)
    results[i, "pct95_diff"] <- quantile(diff, probs = 0.95)
  }
  
  return(results)
}



# Function to compute the eta_hat and the difference given a fixed sample size
compute_diff_stats_fixed <- function(model, true_eta, num_samples = 1000, num_bins = 10,
                                     bin_strategy = "width", threshold = 1000, repetitions = 1000) {
  # Store the Eta estimates for a number of repetitions (for a fixed n)
  eta_estimates <- replicate(repetitions, {
    # Generate data
    data <- generate_data_gmm(num_samples)
    # Predictions
    data$pred <- sapply(data$X, model)
    # Compute estimates for eta, r, g and p
    est <- miscal_estimates(data$pred, data$Y, num_bins, bin_strategy, threshold, my_tvd)
    
    return(est$eta)
  })
  
  # Compute difference between Estimated Eta and True Eta
  diff <- eta_estimates - true_eta
  
  results <- list(eta_estimates = eta_estimates, differences = diff)
  
  return(results)
}



#------------------------    Plotting Functions   ------------------------------



# Function to plot the difference between Estimated Eta and True Eta
plot_difference <- function(results, ylim_main = c(0, 0.2), xlim_zoom = c(15000, 30000), ylim_zoom = c(0, 0.02),
                            line_col = "darkblue", band_col = "royalblue", title = "") {
  # Whole plot
  plot_main <- ggplot(results, aes(x = nsamples, y = mean_diff)) +
    geom_line(color = "darkblue") +
    geom_ribbon(aes(ymin = pct5_diff, ymax = pct95_diff), fill = "royalblue", alpha = 0.35) +
    labs(x = expression(italic("n")),
         y = expression(paste(italic(hat(eta)[TV])," - ", italic(eta[TV])))) +
    theme_minimal() +
    ylim(ylim_main) +
    xlim(c(0, 30000))
    set_plot_params()
  
  # Zoom plot
  plot_zoom <- ggplot(results, aes(x = nsamples, y = mean_diff)) +
    geom_line(color = "darkblue") +
    geom_ribbon(aes(ymin = pct5_diff, ymax = pct95_diff), fill = "royalblue", alpha = 0.35) +
    labs(x = expression(italic("n")),
         y = expression(paste(italic(hat(eta)[TV])," - ", italic(eta[TV])))) +
    theme_minimal() +
    coord_cartesian(xlim = xlim_zoom, ylim = ylim_zoom) + 
    set_plot_params()
  
  # Combine plots in a single plot
  plot_main + plot_zoom + plot_layout(ncol = 2) +
    plot_annotation(title = title, theme = theme(plot.title = element_text(size = 20, hjust = 0.5)))
}



# Function to plot the Reliability Diagrams for the Logistic Regression models
plot_RD <- function(miscal_stats, model_name = " ?", bin_strategy = " ?") {
  # Maximum value for frequency bins
  max_samples <- max(miscal_stats$data_freq_bins)
  
  if (model_name == "Perfect Model") {
    # Scaling factor for frequency to match the deviation plot
    scaling_factor <- 0.1/max_samples
    
    # Shift deviation by 0.05 to align with the secondary y-axis
    dev_shift <- 0.05
    
    # Adjust the breaks for deviations
    dev_breaks <- seq(-0.05, 0.05, 0.01) + dev_shift
    
    # Aestethics of the plot
    lim <- c(0, 0.1)
    dec_pos <- "%.2f"
    alpha_green <- 0
    alpha_black <- 1
    alpha_hline <- 0.7
    freq_steps <- 500
    
  } else if (model_name == "Constant Model") {
    scaling_factor <- 0.02/max_samples
    dev_shift <- 0.01
    dev_breaks <- seq(-0.01, 0.01, 0.005) + dev_shift
    
    lim <- c(0, 0.02)
    dec_pos <- "%.3f"
    alpha_green <- 0
    alpha_black <- 0
    alpha_hline <- 0.7
    freq_steps <- 5000
    
  } else if (model_name == "Uncalibrated Model") {
    scaling_factor <- 2/max_samples
    dev_shift <- 1
    dev_breaks <- seq(-1, 1, 1) + dev_shift
    
    lim <- c(0, 2)
    dec_pos <- "%.0f"
    alpha_green <- 0.2
    alpha_black <- 1
    alpha_hline <- 0
    freq_steps <- 200
    
  } else stop("The model should be one of the following:\nPerfect Model\nConstant Model\nUncalibrated Model")
  
  # Compute the bin midpoints based on the number of bins
  num_bins <- length(miscal_stats$data_freq_bins)
  bin_width <- 1/num_bins
  bin_midpoints <- seq(bin_width/2, 1-bin_width/2, bin_width)
  
  # To center the error bars we subtract the mean
  mean_pcts <- (miscal_stats$pct5_dev + miscal_stats$pct95_dev)/2
  pct5_dev <- miscal_stats$pct5_dev - mean_pcts
  pct95_dev <- miscal_stats$pct95_dev - mean_pcts
  
  # Data for the histogram
  hist_data <- data.frame(bin_mid = bin_midpoints,
                          freq = miscal_stats$data_freq_bins)
  
  # Data for the error bars
  error_data <- data.frame(bin_mid = bin_midpoints,
                           pct5_dev = pct5_dev,
                           pct95_dev = pct95_dev)
  
  
  # Combine every information about miscalibration to plot the RD
  ggplot(miscal_stats, aes(x = data_g)) +
    geom_bar(data = hist_data, aes(x = bin_mid, y = freq * scaling_factor), 
             stat = "identity", fill = "royalblue", color = "darkblue", alpha = 0.35, width = bin_width) +
    
    geom_line(aes(y = data_dev + dev_shift), linetype = "dashed",
              color = "black", linewidth = 0.65, alpha = alpha_black) +
    
    geom_line(aes(y = data_dev + dev_shift), color = "springgreen", 
              linewidth = 3, alpha = alpha_green) +
    
    geom_point(aes(y = data_dev + dev_shift), shape = 4, color = "black", size = 3.5) +
    
    geom_errorbar(data = error_data, aes(x = bin_mid, ymin = pct5_dev + dev_shift, ymax = pct95_dev + dev_shift),
                  color = "firebrick", width = 0.05, linewidth = 0.65) +
    
    geom_hline(yintercept = dev_shift, color = "springgreen", linewidth = 0.65, alpha = alpha_hline) +
    
    # Scale for y with secondary axis (deviation and # samples)
    scale_y_continuous(name = "Deviation", limits = lim, breaks = dev_breaks, labels = sprintf(dec_pos, dev_breaks - dev_shift),
                       sec.axis = sec_axis(~ . / scaling_factor, name = "# samples", breaks = seq(0, max_samples, freq_steps))) +
    
    # Scale for x
    scale_x_continuous(breaks = seq(0, 1, 0.2)) +
    
    # Set labels
    labs(x = expression(paste("Prediction ", italic("g")[1])), title = paste("Reliability Diagram for", model_name),
         subtitle = bin_strategy) +
    
    # Set theme
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 0, hjust = 1),
          axis.title.y.right = element_text(color = "black", angle = 90),
          axis.title.y.left = element_text(color = "black"),
          panel.grid.major.y = element_line(color = "gray", linewidth = 0.5),
          panel.grid.minor.y = element_line(color = "lightgray", linewidth = 0.25),
          plot.title = element_text(size = 17, hjust = 0.5, vjust = 2),
          plot.subtitle = element_text(size = 12, hjust = 0.5, vjust = 1.5),
          plot.background = element_rect(fill = "white", color = NA),
          plot.margin = margin(t = 1.5, r = 1, b = 1, l = 1, unit = "cm")) +
    
    # Prevent clipping of points and error bars
    coord_cartesian(clip = "off") +
    set_plot_params()
}


