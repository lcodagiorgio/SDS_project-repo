

# Setting current working directory to load our packages
setwd("D:/uni/projects/sds")

# Importing functions to perform the experiments
source("fun_experiment_NN.R")

# Using conda environment for ML
use_condaenv("deeplearning", required = T)

# Seed for reproducibility
seed <- 92



# Preprocessed MNIST data
val_split <- 0.2
data <- generate_data_mnist(tr_val_split = 1-val_split)

# Whole original training data
x <- data$x
y <- data$y

# Training and validation sets (if we want to use them distinctly)
x_train <- data$x_train
y_train <- data$y_train

x_val <- data$x_val
y_val <- data$y_val

# Whole original test data
x_test <- data$x_test
y_test <- data$y_test



#-----------------------   Plotting metrics by epochs   ------------------------


# Define number of initializations and epochs
num_init <- 5
epochs <- 50

# Compute the Accuracy, nLL and ECE for the model (by epoch)
metrics <- compute_model_metrics(create_lenet_model, x, y, num_init = num_init, epochs = epochs)

# Save the metrics for faster access
#saveRDS(metrics, "leNet_init_metrics.rds")

# Load the pre-computed metrics for the model
metrics <- readRDS("D:/uni/projects/sds/NN_exp/leNet_init_metrics.rds")


# Prepare data for plotting
plot_data <- data.frame(epoch = rep(1:epochs, num_init),
                        nLL = unlist(lapply(metrics, function(m) m$nll)),
                        Accuracy = unlist(lapply(metrics, function(m) m$accuracy)),
                        ECE = unlist(lapply(metrics, function(m) m$ece)),
                        init = factor(rep(1:num_init, epochs)))


# Plot the metrics by epoch, for each initialization
plot_metrics(plot_data, "LeNet")



#---------------------   Calibration and RDs for Models   ----------------------

#-----------------------------   LeNet Model   ---------------------------------



set.seed(seed)

# Create the LeNet Model
#leNet <- create_lenet_model(input_shape = dim(x)[-1])
#leNet_history <- fit_model(leNet, x, y)

# Saving the fitted model to have faster access
#save_model_hdf5(leNet, "leNet.h5")
leNet <- load_model_hdf5("D:/uni/projects/sds/NN_exp/leNet.h5")
summary(leNet)

# LeNet Model
res_leNet_tvd <- compute_miscal_stats(leNet, num_bins = 10, bin_strategy = "width", threshold = 1000,
                                      num_resamples = 100, dist_fun = tvd)
res_leNet_eucl <- compute_miscal_stats(leNet, num_bins = 10, bin_strategy = "width", threshold = 1000,
                                       num_resamples = 100, dist_fun = euclid)
# Show results
res_leNet_tvd[, -length(res_leNet_tvd)]
res_leNet_eucl[, -length(res_leNet_eucl)]

# Plot the reliability diagrams
plot_RD(res_leNet_tvd, "LeNet Model (MNIST)", "equal-width, TVD")
plot_RD(res_leNet_eucl, "LeNet Model (MNIST)", "equal-width, EUCLIDEAN")



#----------------------------   DenseNet Model   -------------------------------



set.seed(seed)

# Create the DenseNet Model
#denseNet <- create_densenet_model(input_shape = dim(x)[-1])
#denseNet_history <- fit_model(denseNet, x, y)

# Saving the fitted model to have faster access
#save_model_hdf5(denseNet, "denseNet.h5")
denseNet <- load_model_hdf5("D:/uni/projects/sds/NN_exp/denseNet.h5")
summary(denseNet)

# ResNet Model
res_denseNet_tvd <- compute_miscal_stats(denseNet, num_bins = 10, bin_strategy = "width", threshold = 1000,
                                         num_resamples = 100, dist_fun = tvd)
res_denseNet_eucl <- compute_miscal_stats(denseNet, num_bins = 10, bin_strategy = "width", threshold = 1000,
                                          num_resamples = 100, dist_fun = euclid)
# Show results
res_denseNet_tvd[, -length(res_denseNet_tvd)]
res_denseNet_eucl[, -length(res_denseNet_eucl)]

# Plot the reliability diagrams
plot_RD(res_denseNet_tvd, "DenseNet Model (MNIST)", "equal-width, TVD")
plot_RD(res_denseNet_eucl, "DenseNet Model (MNIST)", "equal-width, EUCLIDEAN")



#------------------------------   ResNet Model   -------------------------------



set.seed(seed)

# Create the DenseNet Model
#resNet <- create_resnet_model(input_shape = dim(x)[-1])
#resNet_history <- fit_model(resNet, x, y)

# Saving the fitted model to have faster access
#save_model_hdf5(resNet, "resNet.h5")
resNet <- load_model_hdf5("D:/uni/projects/sds/NN_exp/resNet.h5")
summary(resNet)

# ResNet Model
res_resNet_tvd <- compute_miscal_stats(resNet, num_bins = 10, bin_strategy = "width", threshold = 1000,
                                         num_resamples = 100, dist_fun = tvd)
res_resNet_eucl <- compute_miscal_stats(resNet, num_bins = 10, bin_strategy = "width", threshold = 1000,
                                          num_resamples = 100, dist_fun = euclid)
# Show results
res_resNet_tvd[, -length(res_resNet_tvd)]
res_resNet_eucl[, -length(res_resNet_eucl)]

# Plot the reliability diagrams
plot_RD(res_resNet_tvd, "ResNet Model (MNIST)", "equal-width, TVD")
plot_RD(res_resNet_eucl, "ResNet Model (MNIST)", "equal-width, EUCLIDEAN")



