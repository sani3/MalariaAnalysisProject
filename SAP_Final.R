# ============================================================================
# Project: Malaria Incidence Analysis in Five West African Countries (2000-2021)
# Purpose: Implement the statistical analysis plan for objectives 1-4
# Objectives:
#   1. Quantify and visualize trends
#   2. Compare incidence rates across countries
#   3. Identify key determinants (LMM, GAM, RF, GB)
#   4. Forecast incidence rates (ARIMA, ETS, DNN with h2o)
# Date: March 26, 2025
# ============================================================================

# ============================================================================
# Load Required Libraries
# ============================================================================
library(dplyr)         # Data manipulation
library(tidyr)         # Data tidying
library(ggplot2)       # Plotting
library(lme4)          # Linear Mixed-Effects Models
library(mgcv)          # Generalized Additive Models
library(randomForest)  # Random Forests
library(xgboost)       # Gradient Boosting
library(forecast)      # ARIMA and ETS
library(h2o)           # Deep Neural Networks
library(moments)       # Skewness and kurtosis
library(car)           # VIF calculation
library(gridExtra)     # Arrange multiple plots

# Set seed for reproducibility
set.seed(1234)

# Initialize H2O cluster
h2o.init(nthreads = -1)  # Use all available threads

# Create output directory if it doesn't exist
if (!dir.exists("output")) dir.create("output")

# ============================================================================
# 2. Data Description and Preprocessing
# ============================================================================

# Load dataset (assumes data is in CSV format as provided)
data <- read.csv("data/imputed/data_imput.csv")

# Verify structure: 5 countries, 22 years
expected_rows <- 5 * 22  # 110 observations
if (nrow(data) != expected_rows) stop("Data does not match expected 110 observations")

# Define predictors
predictors <- c("population_density", "urban_population_growth", "population_living_in_slums",
                "rural_population_growth", "people_using_safely_managed_sanitation_services_rural",
                "people_using_at_least_basic_sanitation_services", "che_gdp", "ope_che")

# Check missingness
missing_summary <- data %>%
    summarise(across(everything(), ~mean(is.na(.)) * 100)) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "percent_missing")
write.csv(missing_summary, "output/missing_summary.csv", row.names = FALSE)

# Preprocessing
data <- data %>%
    mutate(
        log_malaria_incidence = log(malaria_incidence_rate + 1),
        across(all_of(predictors), scale)
    ) %>%
    group_by(country) %>%
    mutate(
        lag1_malaria = lag(malaria_incidence_rate, 1),
        lag2_malaria = lag(malaria_incidence_rate, 2),
        lag3_malaria = lag(malaria_incidence_rate, 3)
    ) %>%
    ungroup()

# Split data
train_data <- data %>% filter(year <= 2018)  # 95 obs
test_data <- data %>% filter(year > 2018)   # 15 obs

# ============================================================================
# 3. Objective 1: Quantify and Visualize Trends
# ============================================================================

# Descriptive statistics
desc_stats <- data %>%
    group_by(country) %>%
    summarise(
        mean_incidence = mean(malaria_incidence_rate),
        sd_incidence = sd(malaria_incidence_rate),
        min_incidence = min(malaria_incidence_rate),
        max_incidence = max(malaria_incidence_rate),
        skewness = skewness(malaria_incidence_rate),
        kurtosis = kurtosis(malaria_incidence_rate)
    )
write.csv(desc_stats, "output/descriptive_stats.csv", row.names = FALSE)

# Time-series plot
trend_plot <- ggplot(data, aes(x = year, y = malaria_incidence_rate, color = country)) +
    geom_line() +
    labs(title = "Malaria Incidence Trends (2000-2021)", x = "Year", y = "Incidence Rate (per 1,000)") +
    theme_minimal() +
    scale_color_brewer(palette = "Set1")
ggsave("output/trend_plot.png", trend_plot, width = 10, height = 6)

# Density plot
density_plot <- ggplot(data, aes(x = malaria_incidence_rate, fill = country)) +
    geom_density(alpha = 0.5) +
    labs(title = "Density of Malaria Incidence Rates by Country", x = "Incidence Rate (per 1,000)", y = "Density") +
    theme_minimal() +
    scale_fill_brewer(palette = "Set1")
ggsave("output/density_plot_obj1.png", density_plot, width = 10, height = 6)

# ============================================================================
# 4. Objective 2: Compare Malaria Incidence Across Countries
# ============================================================================

# Correlation analysis
cor_matrix <- data %>%
    select(malaria_incidence_rate, all_of(predictors)) %>%
    cor(method = "pearson")
write.csv(cor_matrix, "output/correlation_matrix.csv")

# ANOVA
anova_result <- aov(malaria_incidence_rate ~ country, data = data)
anova_summary <- summary(anova_result)
sink("output/anova_summary.txt")
print(anova_summary)
sink()

# Diagnostics for ANOVA
shapiro_test <- shapiro.test(residuals(anova_result))
levene_test <- leveneTest(malaria_incidence_rate ~ country, data = data)
sink("output/anova_diagnostics.txt")
cat("Shapiro-Wilk Normality Test:\n")
print(shapiro_test)
cat("\nLevene's Test for Homogeneity of Variance:\n")
print(levene_test)
sink()

# If ANOVA assumptions fail, use Kruskal-Wallis
if (shapiro_test$p.value < 0.05 || levene_test$`Pr(>F)`[1] < 0.05) {
    kw_result <- kruskal.test(malaria_incidence_rate ~ country, data = data)
    sink("output/kruskal_wallis.txt")
    print(kw_result)
    sink()
}

# Boxplot
box_plot <- ggplot(data, aes(x = country, y = malaria_incidence_rate, fill = country)) +
    geom_boxplot() +
    labs(title = "Malaria Incidence by Country", x = "Country", y = "Incidence Rate (per 1,000)") +
    theme_minimal() +
    scale_fill_brewer(palette = "Set1") +
    theme(legend.position = "none")
ggsave("output/box_plot.png", box_plot, width = 8, height = 6)

# Density plot
density_plot_obj2 <- ggplot(data, aes(x = malaria_incidence_rate, fill = country)) +
    geom_density(alpha = 0.5) +
    labs(title = "Comparison of Malaria Incidence Distributions", x = "Incidence Rate (per 1,000)", y = "Density") +
    theme_minimal() +
    scale_fill_brewer(palette = "Set1")
ggsave("output/density_plot_obj2.png", density_plot_obj2, width = 10, height = 6)

# ============================================================================
# 5. Objective 3: Identify Key Determinants
# ============================================================================

# Function to compute metrics
compute_metrics <- function(actual, predicted) {
    rmse <- sqrt(mean((actual - predicted)^2))
    mae <- mean(abs(actual - predicted))
    r2 <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
    return(list(rmse = rmse, mae = mae, r2 = r2))
}

# Train-test split for Objective 3
set.seed(1234)
train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_obj3 <- data[train_indices, ]
test_obj3 <- data[-train_indices, ]

# --- Linear Mixed-Effects Model (LMM) ---
lmm_model <- lmer(log_malaria_incidence ~ population_density + urban_population_growth +
                      population_living_in_slums + rural_population_growth +
                      people_using_safely_managed_sanitation_services_rural +
                      people_using_at_least_basic_sanitation_services + che_gdp + ope_che +
                      (1 | country), data = train_obj3)
lmm_summary <- summary(lmm_model)
sink("output/lmm_summary.txt")
print(lmm_summary)
sink()

# Predictions and diagnostics
lmm_pred_train <- predict(lmm_model, train_obj3)
lmm_pred_test <- predict(lmm_model, test_obj3)
lmm_metrics <- compute_metrics(test_obj3$log_malaria_incidence, lmm_pred_test)
write.csv(as.data.frame(lmm_metrics), "output/lmm_metrics.csv", row.names = FALSE)

lmm_resid_plot <- ggplot(data.frame(fitted = fitted(lmm_model), resid = residuals(lmm_model)),
                         aes(x = fitted, y = resid)) +
    geom_point() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = "LMM Residuals vs Fitted", x = "Fitted Values", y = "Residuals") +
    theme_minimal()
ggsave("output/lmm_resid_plot.png", lmm_resid_plot, width = 6, height = 4)

# --- Generalized Additive Model (GAM) ---
gam_model <- gam(log_malaria_incidence ~ s(population_density) + s(urban_population_growth) +
                     s(population_living_in_slums) + s(rural_population_growth) +
                     s(people_using_safely_managed_sanitation_services_rural) +
                     s(people_using_at_least_basic_sanitation_services) + s(che_gdp) + s(ope_che),
                 data = train_obj3, method = "REML")
gam_summary <- summary(gam_model)
sink("output/gam_summary.txt")
print(gam_summary)
sink()

# Predictions and diagnostics
gam_pred_train <- predict(gam_model, train_obj3)
gam_pred_test <- predict(gam_model, test_obj3)
gam_metrics <- compute_metrics(test_obj3$log_malaria_incidence, gam_pred_test)
write.csv(as.data.frame(gam_metrics), "output/gam_metrics.csv", row.names = FALSE)

gam_check <- gam.check(gam_model)
sink("output/gam_diagnostics.txt")
print(gam_check)
sink()

# --- Random Forest (RF) ---
rf_model <- randomForest(malaria_incidence_rate ~ population_density + urban_population_growth +
                             population_living_in_slums + rural_population_growth +
                             people_using_safely_managed_sanitation_services_rural +
                             people_using_at_least_basic_sanitation_services + che_gdp + ope_che,
                         data = train_obj3, ntree = 500, mtry = sqrt(length(predictors)))
rf_pred_train <- predict(rf_model, train_obj3)
rf_pred_test <- predict(rf_model, test_obj3)
rf_metrics <- compute_metrics(test_obj3$malaria_incidence_rate, rf_pred_test)
write.csv(as.data.frame(rf_metrics), "output/rf_metrics.csv", row.names = FALSE)

# Feature importance
rf_importance <- importance(rf_model)
write.csv(as.data.frame(rf_importance), "output/rf_importance.csv")
rf_importance_plot <- ggplot(as.data.frame(rf_importance), aes(x = reorder(rownames(rf_importance), IncNodePurity), y = IncNodePurity)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(title = "Random Forest Feature Importance", x = "Predictor", y = "% Increase in MSE") +
    theme_minimal()
ggsave("output/rf_importance_plot.png", rf_importance_plot, width = 8, height = 6)

# --- Gradient Boosting (GB) ---
gb_data <- as.matrix(train_obj3[, predictors])
gb_model <- xgboost(data = gb_data, label = train_obj3$malaria_incidence_rate,
                    nrounds = 1000, max_depth = 3, eta = 0.01, objective = "reg:squarederror")
gb_pred_train <- predict(gb_model, gb_data)
gb_pred_test <- predict(gb_model, as.matrix(test_obj3[, predictors]))
gb_metrics <- compute_metrics(test_obj3$malaria_incidence_rate, gb_pred_test)
write.csv(as.data.frame(gb_metrics), "output/gb_metrics.csv", row.names = FALSE)

# Feature importance
gb_importance <- xgb.importance(feature_names = predictors, model = gb_model)
write.csv(as.data.frame(gb_importance), "output/gb_importance.csv")
gb_importance_plot <- ggplot(gb_importance, aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(title = "Gradient Boosting Feature Importance", x = "Predictor", y = "Gain") +
    theme_minimal()
ggsave("output/gb_importance_plot.png", gb_importance_plot, width = 8, height = 6)

# Model comparison
metrics_df <- bind_rows(
    LMM = lmm_metrics,
    GAM = gam_metrics,
    RF = rf_metrics,
    GB = gb_metrics,
    .id = "Model"
)
write.csv(metrics_df, "output/model_comparison_metrics.csv", row.names = FALSE)

# ============================================================================
# 6. Objective 4: Forecast Future Incidence
# ============================================================================

# --- ARIMA ---
arima_forecasts <- list()
arima_metrics <- list()
for (ctry in unique(train_data$country)) {
    ts_data <- ts(train_data$malaria_incidence_rate[train_data$country == ctry], start = 2000, frequency = 1)
    arima_model <- auto.arima(ts_data, stepwise = TRUE, approximation = FALSE)
    arima_fc <- forecast(arima_model, h = 6)  # 2019-2024
    test_actual <- test_data$malaria_incidence_rate[test_data$country == ctry]
    test_pred <- arima_fc$mean[1:3]  # 2019-2021
    arima_metrics[[ctry]] <- compute_metrics(test_actual, test_pred)
    arima_forecasts[[ctry]] <- arima_fc
}

arima_metrics_df <- bind_rows(arima_metrics, .id = "country")
write.csv(arima_metrics_df, "output/arima_metrics.csv", row.names = FALSE)

# Diagnostics
arima_diagnostics <- lapply(unique(train_data$country), function(ctry) {
    ts_data <- ts(train_data$malaria_incidence_rate[train_data$country == ctry], start = 2000, frequency = 1)
    arima_model <- auto.arima(ts_data)
    list(
        acf = acf(residuals(arima_model), plot = FALSE),
        ljung_box = Box.test(residuals(arima_model), type = "Ljung-Box")
    )
})
sink("output/arima_diagnostics.txt")
for (i in seq_along(arima_diagnostics)) {
    cat(paste("Country:", unique(train_data$country)[i], "\n"))
    print(arima_diagnostics[[i]]$ljung_box)
    cat("\n")
}
sink()

# --- ETS ---
ets_forecasts <- list()
ets_metrics <- list()
for (ctry in unique(train_data$country)) {
    ts_data <- ts(train_data$malaria_incidence_rate[train_data$country == ctry], start = 2000, frequency = 1)
    ets_model <- ets(ts_data)
    ets_fc <- forecast(ets_model, h = 6)
    test_actual <- test_data$malaria_incidence_rate[test_data$country == ctry]
    test_pred <- ets_fc$mean[1:3]
    ets_metrics[[ctry]] <- compute_metrics(test_actual, test_pred)
    ets_forecasts[[ctry]] <- ets_fc
}

ets_metrics_df <- bind_rows(ets_metrics, .id = "country")
write.csv(ets_metrics_df, "output/ets_metrics.csv", row.names = FALSE)

# --- Deep Neural Network (DNN) with H2O ---
dnn_train <- train_data %>% filter(!is.na(lag3_malaria)) %>%
    select(malaria_incidence_rate, lag1_malaria, lag2_malaria, lag3_malaria)
train_h2o <- as.h2o(dnn_train)
dnn_model <- h2o.deeplearning(
    x = c("lag1_malaria", "lag2_malaria", "lag3_malaria"),
    y = "malaria_incidence_rate",
    training_frame = train_h2o,
    hidden = c(10),
    activation = "Rectifier",
    epochs = 50,
    l2 = 0.01,
    adaptive_rate = TRUE,
    seed = 1234
)

# Predictions and forecasting
dnn_metrics <- list()
dnn_forecasts <- list()
for (ctry in unique(train_data$country)) {
    ctry_data <- train_data %>% filter(country == ctry) %>% arrange(year)
    last_obs <- tail(ctry_data, 1) %>% select(lag1_malaria, lag2_malaria, lag3_malaria)
    fc_values <- numeric(6)  # 2019-2024
    for (i in 1:6) {
        pred <- as.vector(h2o.predict(dnn_model, as.h2o(last_obs)))
        fc_values[i] <- pred
        last_obs <- last_obs %>% mutate(
            lag3_malaria = lag2_malaria,
            lag2_malaria = lag1_malaria,
            lag1_malaria = pred
        )
    }
    test_actual <- test_data$malaria_incidence_rate[test_data$country == ctry]
    test_pred <- fc_values[1:3]
    dnn_metrics[[ctry]] <- compute_metrics(test_actual, test_pred)
    dnn_forecasts[[ctry]] <- data.frame(year = 2019:2024, forecast = fc_values)
}

dnn_metrics_df <- bind_rows(dnn_metrics, .id = "country")
write.csv(dnn_metrics_df, "output/dnn_metrics.csv", row.names = FALSE)

# Combine forecasts for plotting
forecast_df <- bind_rows(
    lapply(names(arima_forecasts), function(ctry) {
        data.frame(country = ctry, year = 2019:2024, forecast = arima_forecasts[[ctry]]$mean[1:6], model = "ARIMA")
    }),
    lapply(names(ets_forecasts), function(ctry) {
        data.frame(country = ctry, year = 2019:2024, forecast = ets_forecasts[[ctry]]$mean[1:6], model = "ETS")
    }),
    lapply(names(dnn_forecasts), function(ctry) {
        dnn_forecasts[[ctry]] %>% mutate(country = ctry, model = "DNN")
    })
)

forecast_plot <- ggplot(forecast_df, aes(x = year, y = forecast, color = model)) +
    geom_line() +
    facet_wrap(~country, scales = "free_y") +
    labs(title = "Malaria Incidence Forecasts (2019-2024)", x = "Year", y = "Incidence Rate") +
    theme_minimal() +
    scale_color_brewer(palette = "Set1")
ggsave("output/forecast_plot.png", forecast_plot, width = 12, height = 10)

# Model comparison
mse_values <- data.frame(
    ARIMA = unlist(lapply(arima_metrics, function(x) x$rmse^2)),
    ETS = unlist(lapply(ets_metrics, function(x) x$rmse^2)),
    DNN = unlist(lapply(dnn_metrics, function(x) x$rmse^2))
)
t_test_arima_ets <- t.test(mse_values$ARIMA, mse_values$ETS, paired = TRUE)
t_test_arima_dnn <- t.test(mse_values$ARIMA, mse_values$DNN, paired = TRUE)
t_test_ets_dnn <- t.test(mse_values$ETS, mse_values$DNN, paired = TRUE)
sink("output/forecast_model_comparison.txt")
print(t_test_arima_ets)
print(t_test_arima_dnn)
print(t_test_ets_dnn)
sink()

# ============================================================================
# 7. Validation and Robustness
# ============================================================================

# Imputation sensitivity (mean vs median for rural sanitation)
data_mean_imp <- data %>%
    mutate(people_using_safely_managed_sanitation_services_rural = replace(
        people_using_safely_managed_sanitation_services_rural,
        country == "Cameroon",
        mean(data$people_using_safely_managed_sanitation_services_rural[data$country != "Cameroon"], na.rm = TRUE)
    ))
write.csv(data_mean_imp, "output/data_mean_imputed.csv", row.names = FALSE)

# ============================================================================
# Cleanup and Save Workspace
# ============================================================================
save.image("output/malaria_analysis.RData")
h2o.shutdown(prompt = FALSE)

# List output files for reference
output_files <- list.files("output", full.names = TRUE)
write.csv(data.frame(file = output_files), "output/output_file_list.csv", row.names = FALSE)