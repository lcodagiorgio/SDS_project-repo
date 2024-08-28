
# Importing our packages
source("fun_data_generation.R")
source("fun_calibration_NN.R")

# Importing useful ML packages
library(ggplot2)
library(gridExtra)

library(reticulate)
library(keras3)
library(keras)
library(tensorflow)
library(Metrics)


#------------------------------   Models   -------------------------------------



# Function to create the LeNet model
create_lenet_model <- function(input_shape = c(28, 28, 1), num_class = 10) {
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 6, kernel_size = c(5, 5), activation = 'relu', input_shape = input_shape) %>%
    layer_average_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 16, kernel_size = c(5, 5), activation = 'relu') %>%
    layer_average_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 120, activation = 'relu') %>%
    layer_dense(units = 84, activation = 'relu') %>%
    layer_dense(units = num_class, activation = 'softmax')
  
  return(model)
}



# Function to create the DenseNet model
create_densenet_model <- function(input_shape = c(28, 28, 1), num_class = 10) {
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = 'same', input_shape = input_shape) %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    
    # First dense block
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = 'same') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = 'same') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    
    # First transition layer
    layer_conv_2d(filters = 64, kernel_size = c(1, 1), padding = 'same') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    layer_average_pooling_2d(pool_size = c(2, 2)) %>%
    
    # Second dense block
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = 'same') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = 'same') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    
    # Second transition block
    layer_conv_2d(filters = 128, kernel_size = c(1, 1), padding = 'same') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    layer_average_pooling_2d(pool_size = c(2, 2)) %>%
    
    # Flattening layer
    layer_flatten() %>%
    layer_dense(units = 256, activation = 'relu') %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = num_class, activation = 'softmax')
  
  return(model)
}



# Function to create a residual block
residual_block <- function(x, filters, kernel_size = c(3, 3), stride = 1) {
  shortcut <- x
  
  x <- layer_conv_2d(x, filters = filters, kernel_size = kernel_size, strides = stride, padding = 'same') %>%
    layer_batch_normalization() %>%
    layer_activation('relu')
  
  x <- layer_conv_2d(x, filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same') %>%
    layer_batch_normalization()
  
  if (stride != 1) {
    shortcut <- layer_conv_2d(shortcut, filters = filters, kernel_size = c(1, 1), strides = stride, padding = 'same') %>%
      layer_batch_normalization()
  }
  
  x <- layer_add(list(x, shortcut)) %>%
    layer_activation('relu')
  
  return(x)
}

# Function to create the ResNet model
create_resnet_model <- function(input_shape = c(28, 28, 1), num_class = 10) {
  input <- layer_input(shape = input_shape)
  
  x <- layer_conv_2d(input, filters = 64, kernel_size = c(3, 3), strides = 1, padding = 'same') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = 2, padding = 'same')
  
  x <- residual_block(x, filters = 64)
  x <- residual_block(x, filters = 64)
  
  x <- residual_block(x, filters = 128, stride = 2)
  x <- residual_block(x, filters = 128)
  
  x <- residual_block(x, filters = 256, stride = 2)
  x <- residual_block(x, filters = 256)
  
  x <- layer_global_average_pooling_2d(x) %>%
    layer_dense(units = num_class, activation = "softmax")
  
  model <- keras_model(input, x)
  
  return(model)
}



fit_model <- function(model, x_train, y_train, loss = "categorical_crossentropy", optimizer = optimizer_adam(),
                      metrics = c("accuracy"), epochs = 50, batch = 64, val_split = 0.2) {
  # Compile the model
  model %>% compile(
    loss = loss,
    optimizer = optimizer,
    metrics = metrics)
  
  # Fit the model
  history <- model %>% fit(
    x_train, y_train,
    epochs = epochs,
    batch_size = batch,
    validation_split = val_split,
    verbose = 0)
  
  return(history)
}



#-----------------------------   ECE and Metrics   -----------------------------



# Function to compute ECE
compute_ece <- function(predictions, true_labels, num_bins = 10) {
  # Extract the maximum predicted probabilities (confidences)
  confidences <- apply(predictions, 1, max)
  # Extract the predicted labels
  predicted_labels <- apply(predictions, 1, which.max) - 1
  
  # Bin confidences
  bin_indices <- cut(confidences, breaks = seq(0, 1, length.out = num_bins + 1), include.lowest = TRUE, right = FALSE)
  bin_data <- data.frame(confidences, true_labels, predicted_labels, bin_indices)
  
  # Initialize ECE
  ece <- 0
  # Iterate over each bin to calculate bin confidence, accuracy, and weight
  for (bin in levels(bin_data$bin_indices)) {
    bin_subset <- subset(bin_data, bin_indices == bin)
    if (nrow(bin_subset) > 0) {
      bin_confidence <- mean(bin_subset$confidences)
      bin_accuracy <- mean(bin_subset$true_labels == bin_subset$predicted_labels)
      bin_weight <- nrow(bin_subset) / nrow(bin_data)
      ece <- ece + bin_weight * abs(bin_confidence - bin_accuracy)
    }
  }
  
  return(ece)
}



compute_model_metrics <- function(model_creation_f = create_lenet_model, x_train, y_train,
                                  num_init = 5, epochs = 50, val_split = 0.2) {
  # Store results for each initialization
  results <- list()
  
  for (i in 1:num_init) {
    model <- model_creation_f()
    
    # Compile the model
    model %>% compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_adam(),
      metrics = c('accuracy')
    )
    
    # Initialize lists to store metrics
    nll_values <- numeric(epochs)
    accuracy_values <- numeric(epochs)
    ece_values <- numeric(epochs)
    
    for (epoch in 1:epochs) {
      # Train the model for one epoch
      history <- model %>% fit(
        x_train, y_train,
        epochs = 1,
        batch_size = 128,
        validation_split = val_split,
        verbose = 0
      )
      
      # Predict on the validation data
      train_pred <- model %>% predict(x_train)
      train_labels <- apply(y_train, 1, which.max) - 1
      
      # Calculate nLL
      nll_values[epoch] <- history$metrics$val_loss
      
      # Calculate Accuracy
      accuracy_values[epoch] <- history$metrics$val_accuracy
      
      # Calculate ECE
      ece_values[epoch] <- compute_ece(train_pred, train_labels)
    }
    
    # Store the metrics for each initialization
    results[[i]] <- list(nll = nll_values, accuracy = accuracy_values, ece = ece_values)
  }
  
  return(results)
}



#------------------------    Plotting Functions   ------------------------------



# Function to plot metrics against epochs
plot_metrics <- function(data, model_name = "?") {
  nll <- ggplot(data, aes(x = epoch, y = nLL, color = init)) +
    geom_line(linewidth = 0.65) +
    labs(title = model_name, y = "nLL", x = "Epoch") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
    set_plot_params()
  
  acc <- ggplot(data, aes(x = epoch, y = Accuracy, color = init)) +
    geom_line(linewidth = 0.65) +
    labs(title = model_name, y = "Accuracy", x = "Epoch") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
    set_plot_params()
  
  ece <- ggplot(data, aes(x = epoch, y = ECE, color = init)) +
    geom_line(linewidth = 0.65) +
    labs(title = model_name, y = "ECE", x = "Epoch") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
    set_plot_params()
  
  print(nll)
  print(acc)
  print(ece)
}



# Function to plot the Reliability Diagrams for the Neural Network models
plot_RD <- function(miscal_stats, model_name = " ?", bin_strategy = " ?") {
  # Maximum value for frequency bins
  max_samples <- max(miscal_stats$data_freq_bins)
  
  # Scaling factor for frequency to match the deviation plot
  scaling_factor <- 2/max_samples
  
  # Shift deviation by 1 to align with the secondary y-axis
  dev_shift <- 1
  
  # Adjust the breaks for deviations
  dev_breaks <- seq(-1, 1, 0.5) + dev_shift
  
  # Aestethics of the plot
  lim <- c(0, 2)
  dec_pos <- "%.2f"
  alpha_green <- 0
  alpha_black <- 1
  alpha_hline <- 0.7
  freq_steps <- 2000
    
  
  
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
    labs(x = "Maximum prediction", title = paste("Reliability Diagram for", model_name),
         subtitle = bin_strategy) +
    
    # Set theme
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 0, hjust = 1),
          axis.title.y.right = element_text(color = "black", angle = 90),
          axis.title.y.left = element_text(color = "black"),
          panel.grid.major.y = element_line(color = "gray", linewidth = 0.5),  # Add major y-axis grid lines
          panel.grid.minor.y = element_line(color = "lightgray", linewidth = 0.25),  # Add minor y-axis grid lines
          plot.title = element_text(size = 17, hjust = 0.5, vjust = 2),
          plot.subtitle = element_text(size = 12, hjust = 0.5, vjust = 1.5),
          plot.background = element_rect(fill = "white", color = NA),
          plot.margin = margin(t = 1.5, r = 1, b = 1, l = 1, unit = "cm")) +
    
    # Prevent clipping of points and error bars
    coord_cartesian(clip = "off") +
    set_plot_params()
}


