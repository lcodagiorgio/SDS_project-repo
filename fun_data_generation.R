
# Useful plotting libraries
library(ggplot2)
library(patchwork)
library(grid)
library(scales)



# Function to set aestethic plot parameters
set_plot_params <- function(base_size = 12, base_family = "sans", margin_cm = 1,
                            title_size = 16, axis_title_size = 14, axis_text_size = 12) {
  theme(
    plot.margin = unit(rep(margin_cm, 4), "cm"),
    axis.title.x = element_text(size = axis_title_size, margin = margin(t = 10)),
    axis.title.y = element_text(size = axis_title_size, margin = margin(r = 10)),
    axis.text.x = element_text(size = axis_text_size),
    axis.text.y = element_text(size = axis_text_size),
    text = element_text(size = base_size, family = base_family)
  )
}



#-------------------------   Data Generation   ---------------------------------



# Function to generate the data for the experiments
generate_data_gmm <- function(num_samples = 1000, classes = c(-1, 1), sdx = 1, probs = c(0.5, 0.5)) {
  # Generate the class labels
  Y <- sample(classes, size = num_samples, replace = T, prob = probs)
  
  # Generate Xs based on the classes
  X <- rnorm(num_samples, mean = Y, sd = sdx)
  
  # Dataframe
  data <- data.frame(X = X, Y = Y)
  
  return(data)
}



# Function to generate MNIST data
generate_data_mnist <- function(tr_val_split = 0.8) {
  # Load MNIST data
  mnist <- dataset_mnist()
  
  # Split the data in train, validation and test sets
  x <- mnist$train$x
  y <- mnist$train$y
  
  x_test <- mnist$test$x
  y_test <- mnist$test$y

  # One-Hot encoding
  y <- to_categorical(y, num_classes = length(unique(y)))
  y_test <- to_categorical(y_test, num_classes = length(unique(y_test)))
  
  # Train-val split
  val_idx <- as.integer(nrow(x)*tr_val_split)
  x_train <- x[1:val_idx, , ]
  x_val <- x[(val_idx+1):nrow(x), , ]
  
  y_train <- y[1:val_idx, ]
  y_val <- y[(val_idx+1):nrow(y), ]
  
  # Preprocess the data
  x <- x/255
  x_train <- x_train/255
  x_val <- x_val/255
  x_test <- x_test/255
  
  # Reshape the data
  x <- array_reshape(x, c(nrow(x), 28, 28, 1))
  x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
  x_val <- array_reshape(x_val, c(nrow(x_val), 28, 28, 1))
  x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
  
  print("Dimensions of x, x_train, x_val, x_test")
  print(dim(x)); print(dim(x_train)) ; print(dim(x_val)) ; print(dim(x_test))
  print("Dimensions of y, y_train, y_val, y_test")
  print(dim(y)); print(dim(y_train)) ; print(dim(y_val)) ; print(dim(y_test))
  
  
  return(list(x = x, y = y,
              x_train = x_train, y_train = y_train,
              x_val = x_val, y_val = y_val,
              x_test = x_test, y_test = y_test))
}


