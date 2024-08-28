

# Setting current working directory to load our packages
setwd("D:/uni/projects/sds")

# Importing functions to perform the experiments
source("fun_experiment_LR.R")

# Seed for reproducibility
seed <- 92


# List of the logistic regression models
models <- build_models()

# Range of values to test for the sample size
nsamples_vector <- seq(2, 30000, 100)

# Times to compute the estimates for each specific sample size
repetitions <- 100



#----------------------------   10 BINS   --------------------------------------


nbins <- 10


# Perfect Model
set.seed(seed)
true_eta <- 0
#results_perfect_10 <- compute_diff_stats(models$perfect, true_eta, nsamples_vector,
#                                          num_bins = nbins, bin_strategy = "width", repetitions = repetitions)
plot_difference(results_perfect_10, title = "Perfect Model (bins = 10)",
                ylim_main = c(0, 0.15), ylim_zoom = c(0, 0.015), xlim_zoom = c(15000, 30000))

# Structure of the results
head(results_perfect_10)



# Constant Model
set.seed(seed)
true_eta <- 0
#results_constant_10 <- compute_diff_stats(models$constant, true_eta, nsamples_vector,
#                                           num_bins = nbins, bin_strategy = "width", repetitions = repetitions)
plot_difference(results_constant_10, title = "Constant Model (bins = 10)",
                ylim_main = c(0, 0.1), ylim_zoom = c(-0.01, 0.02), xlim_zoom = c(15000, 30000))



# Uncalibrated Model
set.seed(seed)
true_eta <- 0.56
#results_uncalibrated_10 <- compute_diff_stats(models$uncalibrated, true_eta, nsamples_vector,
#                                               num_bins = nbins, bin_strategy = "width", repetitions = repetitions)
plot_difference(results_uncalibrated_10, title = "Uncalibrated Model (bins = 10)",
                ylim_main = c(-0.06, 0.06), ylim_zoom = c(-0.02, 0.02), xlim_zoom = c(15000, 30000))


# all_results_10 <- list(perfect_10 = results_perfect_10,
#                     constant_10 = results_constant_10,
#                     uncalibrated_10 = results_uncalibrated_10)
# 
# # Save the final results (every combination of models and bin size)
# saveRDS(all_results_10, file = "miscalibration_results_10.rds")



#---------------------------   100 BINS   --------------------------------------



nbins <- 100


# Perfect Model
set.seed(seed)
true_eta <- 0
#results_perfect_100 <- compute_diff_stats(models$perfect, true_eta, nsamples_vector,
#                                          num_bins = nbins, bin_strategy = "width", repetitions = repetitions)
plot_difference(results_perfect_100, title = "Perfect Model (bins = 100)",
                ylim_main = c(0, 0.15), ylim_zoom = c(0.01, 0.03), xlim_zoom = c(15000, 30000))


# Constant Model
set.seed(seed)
true_eta <- 0
#results_constant_100 <- compute_diff_stats(models$constant, true_eta, nsamples_vector,
#                                           num_bins = nbins, bin_strategy = "width", repetitions = repetitions)
plot_difference(results_constant_100, title = "Constant Model (bins = 100)",
                ylim_main = c(0, 0.1), ylim_zoom = c(-0.01, 0.02), xlim_zoom = c(15000, 30000))



# Uncalibrated Model
set.seed(seed)
true_eta <- 0.56
#results_uncalibrated_100 <- compute_diff_stats(models$uncalibrated, true_eta, nsamples_vector,
#                                               num_bins = nbins, bin_strategy = "width", repetitions = repetitions)
plot_difference(results_uncalibrated_100, title = "Uncalibrated Model (bins = 100)",
                ylim_main = c(-0, 0.2), ylim_zoom = c(0, 0.04), xlim_zoom = c(15000, 30000))


# all_results_100 <- list(perfect_100 = results_perfect_100,
#                         constant_100 = results_constant_100,
#                         uncalibrated_100 = results_uncalibrated_100)
# 
# 
# saveRDS(all_results_100, file = "miscalibration_results_100.rds")
# 
# 
# all_results <- list(perfect_10 = results_perfect_10,
#                     constant_10 = results_constant_10,
#                     uncalibrated_10 = results_uncalibrated_10,
#                     perfect_100 = results_perfect_100,
#                     constant_100 = results_constant_100,
#                     uncalibrated_100 = results_uncalibrated_100)
# 
# 
# # Save the final results (every combination of models and bin size)
# saveRDS(all_results, file = "miscalibration_results.rds")



#-----------------------   Check fixed n results   -----------------------------


res_p <- compute_diff_stats_fixed(models$perfect, 0, num_samples = 50000,
                                  num_bins = 100, bin_strategy = "width", repetitions = 100)
mean(res_p$differences)



res_c <- compute_diff_stats_fixed(models$constant, 0, num_samples = 50000,
                                  num_bins = 100, bin_strategy = "width", repetitions = 100)
mean(res_c$differences)



res_u <- compute_diff_stats_fixed(models$uncalibrated, 0.56, num_samples = 50000,
                                  num_bins = 100, bin_strategy = "width", repetitions = 100)
mean(res_u$differences)



#---------------------------   Load Results   ----------------------------------



results <- readRDS("D:/uni/projects/sds/logreg_exp/miscalibration_results.rds")


# Perfect Model
plot_difference(results$perfect_10, title = "Perfect Model (bins = 10)",
                ylim_main = c(0, 0.15), ylim_zoom = c(0, 0.015), xlim_zoom = c(15000, 30000))
# Perfect Model
plot_difference(results$perfect_100, title = "Perfect Model (bins = 100)",
                ylim_main = c(0, 0.15), ylim_zoom = c(0.01, 0.03), xlim_zoom = c(15000, 30000))



# Constant Model
plot_difference(results$constant_10, title = "Constant Model (bins = 10)",
                ylim_main = c(0, 0.1), ylim_zoom = c(-0.01, 0.02), xlim_zoom = c(15000, 30000))
# Constant Model
plot_difference(results$constant_100, title = "Constant Model (bins = 100)",
                ylim_main = c(0, 0.1), ylim_zoom = c(-0.01, 0.02), xlim_zoom = c(15000, 30000))



# Uncalibrated Model
plot_difference(results$uncalibrated_10, title = "Uncalibrated Model (bins = 10)",
                ylim_main = c(-0.06, 0.06), ylim_zoom = c(-0.02, 0.02), xlim_zoom = c(15000, 30000))
# Uncalibrated Model
plot_difference(results$uncalibrated_100, title = "Uncalibrated Model (bins = 100)",
                ylim_main = c(0, 0.2), ylim_zoom = c(0, 0.04), xlim_zoom = c(15000, 30000))



#------------------------   Reliability Diagrams   -----------------------------


# Rebuilding the LR models
models <- build_models()

# Sample size on which to perform the experiment
sample_size <- 10000




# Perfect Model
set.seed(seed)
miscal_perfect_eqw <- compute_miscal_stats(models$perfect, num_samples = sample_size,
                                       num_bins = 10, bin_strategy = "width", threshold = 1000, num_resamples = 1000)
miscal_perfect_dd <- compute_miscal_stats(models$perfect, num_samples = sample_size,
                                       num_bins = 10, bin_strategy = "data", threshold = 1000, num_resamples = 1000)
# Show results
miscal_perfect_eqw[, -ncol(miscal_perfect_eqw)]
miscal_perfect_dd[, -ncol(miscal_perfect_dd)]

# Plot the reliability diagrams
plot_RD(miscal_perfect_eqw, "Perfect Model", "equal-width")
plot_RD(miscal_perfect_dd, "Perfect Model", "data-dependent")





# Constant Model
set.seed(seed)
miscal_constant_eqw <- compute_miscal_stats(models$constant, num_samples = sample_size,
                                        num_bins = 10, bin_strategy = "width", num_resamples = 1000)
miscal_constant_dd <- compute_miscal_stats(models$constant, num_samples = sample_size,
                                        num_bins = 10, bin_strategy = "data", num_resamples = 1000)
# Show results
miscal_constant_eqw[, -ncol(miscal_constant_eqw)]
miscal_constant_dd[, -ncol(miscal_constant_dd)]

# Plot the reliability diagrams
plot_RD(miscal_constant_eqw, "Constant Model", "equal-width")
plot_RD(miscal_constant_dd, "Constant Model", "data-dependent")





# Uncalibrated Model
set.seed(seed)
miscal_uncalibrated_eqw <- compute_miscal_stats(models$uncalibrated, num_samples = sample_size,
                                        num_bins = 10, bin_strategy = "width", num_resamples = 1000)
miscal_uncalibrated_dd <- compute_miscal_stats(models$uncalibrated, num_samples = sample_size,
                                        num_bins = 10, bin_strategy = "data", num_resamples = 1000)
# Show results
miscal_uncalibrated_eqw[, -ncol(miscal_uncalibrated_eqw)]
miscal_uncalibrated_dd[, -ncol(miscal_uncalibrated_dd)]

# Plot the reliability diagrams
plot_RD(miscal_uncalibrated_eqw, "Uncalibrated Model", "equal-width")
plot_RD(miscal_uncalibrated_dd, "Uncalibrated Model", "data-dependent")



