set.seed(424)

micro_data <- read_csv(file.path(data_path, "micro_dataset_original.csv")) |> 
  mutate(avg_wind_speed = avg_wind_speed / 10) |> 
  arrange(date)

# Prep ------------------------------------------------------------------------
# Group relevant covariates
fe_cov <- c("routeside")
weather_cov <- c("valueTMAX_MIDWAY", "valuePRCP_MIDWAY", "avg_wind_speed")
treatment <- "treatment"
dvs <- c("violent", "stand_crimes")
vars_all <- c(fe_cov, weather_cov, treatment, dvs)

# Drop missing values
# and create route cluster ID
micro_data <- micro_data |> 
  filter(insample == 1) |> 
  drop_na(any_of(vars_all)) |> 
  mutate(across(c(routeside), as.numeric),
         route_id = as.numeric(factor(route_num1_mod,
                                      levels = c("I290", "I55", "I57", "I90_A", 
                                                 "I90_B", "I90_C", "I94"))))

# Focus on violent crimes
violent_data <- micro_data |> 
  filter(violent == 1)

# Create cluster, treatment, and outcome vectors,
# and features (covariate) matrix
route_id <- violent_data$route_id
X_raw <- violent_data |> select(all_of(c(fe_cov, weather_cov, "route_id")))
D <- violent_data |> pull(treatment)
Y <- violent_data |> pull("stand_crimes")

make_X <- function(X_raw, fe) {
  X <- X_raw |> 
    select(all_of(c(fe, weather_cov)))
  X <- dummy_cols(X, select_columns = fe,
                  remove_first_dummy = TRUE) |> 
    select(-all_of(fe))
  int_dummy <- colnames(X)[str_detect(colnames(X), paste0("routeside", "(_\\d)?"))]
  X <- X |> 
    mutate(across(all_of(int_dummy), .fns = ~ . * valueTMAX_MIDWAY, .names = "{.col}_x_max_temp"),
           across(all_of(int_dummy), .fns = ~ . * valuePRCP_MIDWAY, .names = "{.col}_x_prcp"))
  return(X)
}

X_no_cluster <- make_X(X_raw, fe = c("routeside"))
X_cluster <- make_X(X_raw, fe = c("routeside", "route_id"))

X_list <- list(X_no_cluster, X_cluster)

# Model fitting ----------------------------------------------------------------
# Create a list of covariates, FE, and arguments for loop
model_tags <- c("Non-clustered", "Clustered")

args_list <- list(
  list(Y = Y,
       W = D,
       clusters = NULL,
       equalize.cluster.weights = FALSE),
  list(Y = Y,
       W = D,
       clusters = route_id,
       equalize.cluster.weights = TRUE)
)

# Iteratively fit causal forests
forest_results <- list()
for (i in seq_along(model_tags)) {
  
  args <- args_list[[i]]
  args <- append(args, list(X = X_list[[i]]))
  
  print(paste("Fitting", model_tags[i], "model..."))
  forest_results[[i]] <- do.call(causal_forest, args)
}

# Results ----------------------------------------------------------------------
# common gof map for modelsummary
gof_f <- function(x) format(round(x, 3), big.mark = ",")
gm <- list(
  list("raw" = "nobs", "clean" = "Observations", "fmt" = gof_f)
)

# Generate table for heterogeneity calibration ("omnibus" test)
calibration_res <- list()
for (i in seq_along(model_tags)) {
  res <- test_calibration(forest_results[[i]], vcov.type = "HC1")
  calibration_res[[i]] <- res
}
names(calibration_res) <- model_tags

cm_calib <- c("mean.forest.prediction" = "Mean forest prediction",
              "differential.forest.prediction" = "Differential forest prediction")

add_rows <- tibble(term = c("Route \U00D7 side fixed effects", 
                            "Route \U00D7 side weather interaction",
                            "Route fixed effects"),
                   col1 = c("X", "X", NA),
                   col2 = c("X", "X", "X"))
attr(add_rows, "position") <- c(5:7)


msummary(calibration_res, fmt = 3,
         coef_map = cm_calib, gof_map = gm,
         add_rows = add_rows,
         output = "gt") |> 
  sub_missing(missing_text = "") |> 
  tab_options(table.font.size = "9pt",
              table.width = pct(100),) |>
  gtsave("grf_calibration.tex", path = file.path(output_path, "tables"))


# Generate table for ATO and CATE by covariate
linproj_res <- list()
for (i in seq_along(model_tags)) {
  cf <- forest_results[[i]]
  res <- best_linear_projection(cf,
                                cf$X.orig[, c("valueTMAX_MIDWAY", "valuePRCP_MIDWAY", 
                                              "avg_wind_speed")],
                                target.sample = "overlap",
                                vcov.type = "HC1")
  linproj_res[[i]] <- res
}
names(linproj_res) <- model_tags

ato_res <- list()
for (i in seq_along(model_tags)) {
  ato <- average_treatment_effect(forest_results[[i]], target.sample = "overlap")
  ti <- as_tibble(as.list(ato)) |> 
    add_column(term = "treatment", .before = "estimate") |> 
    rename("std.error" = std.err)
  gl <- tibble()
  mod <- list(
    tidy = ti,
    glance = gl
  )
  class(mod) <- "modelsummary_list"
  ato_res[[i]] <- mod
}
names(ato_res) <- model_tags

panels <- list(
  ato_res,
  linproj_res
)
panel_a_title <- "Average treatment effect for the overlap (ATO)"
panel_b_title <- "Conditional average treatment effect (CATE)"
cm_cate <- c("treatment" = "Treatment (downwind)",
             "valueTMAX_MIDWAY" = "Daily maximum temperature",
             "valuePRCP_MIDWAY" = "Daily precipitation",
             "avg_wind_speed" = "Daily average wind speed")
add_rows <- tibble(term = c("",
                            "Route \U00D7 side fixed effects", 
                            "Route \U00D7 side weather interaction",
                            "Route fixed effects"),
                   col1 = c(NA, "X", "X", NA),
                   col2 = c(NA, "X", "X", "X"))
attr(add_rows, "position") <- c(3, 10:13)

msummary(panels, fmt = 4, shape = "rbind",
         coef_map = cm_cate, gof_map = gm,
         add_rows = add_rows,
         output = "gt") |> 
  sub_missing(missing_text = "") |> 
  tab_row_group(rows = 1:3, label = panel_a_title) |> 
  tab_row_group(rows = 4:13, label = panel_b_title) |> 
  row_group_order(groups = c(panel_a_title, panel_b_title)) |> 
  cols_width(1 ~ px(300),
             2 ~ px(100),
             3~ px(100)) |> 
  tab_options(table.font.size = "9pt",
              table.width = pct(100),) |> 
  gtsave("ate_cate.tex", path = file.path(output_path, "tables"))

# CATE visuals -----------------------------------------------------------------
theme_custom <- theme_minimal() +
  theme(legend.title = element_blank(),
        panel.border = element_rect(color = "grey", fill = NA),
        axis.title = element_text(size = 11,),
        axis.text = element_text(size = 9),
        axis.title.x = element_text(margin = margin(7,0,0,0)),
        axis.title.y = element_text(margin = margin(0,7,0,0)),
        legend.text = element_text(size = 8))

# Explore effect heterogeneity with clustered model
forest_cluster <- forest_results[[2]]
X_orig <- forest_cluster$X.orig
tau_pred <- predict(forest_cluster, estimate.variance = TRUE)
cate_df <- tibble(
  tau_hat = tau_pred$predictions,
  var_hat = tau_pred$variance.estimates,
  max_temp = forest_cluster$X.orig$valueTMAX_MIDWAY,
  prcp = forest_cluster$X.orig$valuePRCP_MIDWAY,
  avg_wind_speed = forest_cluster$X.orig$avg_wind_speed,
  route_id = route_id
)

# Histogram of CATE
cate_hist <- cate_df |> 
  ggplot(aes(x = tau_hat)) +
  geom_histogram(bins = 30, color = "#e9ecef", fill = "#69b3a2", alpha = 0.7,) +
  labs(x = "CATE", y = "Frequency") +
  theme_custom 

ggsave(file.path(output_path, "figures/cate_histogram.svg"), cate_hist,
       width = 7, height = 4, units = "in", dpi = 300)

# T-test on effect heterogeneity by weather covariates
high_temp <- cate_df$max_temp > median(cate_df$max_temp)
ate_temp_high <- average_treatment_effect(forest_cluster, 
                                          subset = high_temp, target.sample = "overlap")
ate_temp_low <- average_treatment_effect(forest_cluster,
                                         subset = !high_temp, target.sample = "overlap")
round(ate_temp_high[1] - ate_temp_low[1], 3) -
  round(qnorm(0.975) * sqrt(ate_temp_high[2]^2 + ate_temp_low[2]^2), 3)
round(ate_temp_high[1] + ate_temp_low[1], 3) +
  round(qnorm(0.975) * sqrt(ate_temp_high[2] ^2 + ate_temp_low[2] ^ 2), 3)

high_prcp <- cate_df$prcp > median(cate_df$prcp)
ate_prcp_high <- average_treatment_effect(forest_cluster, 
                                          subset = high_prcp, target.sample = "overlap")
ate_prcp_low <- average_treatment_effect(forest_cluster,
                                         subset = !high_prcp, target.sample = "overlap")
round(ate_prcp_high[1] - ate_prcp_low[1], 3) -
  round(qnorm(0.975) * sqrt(ate_prcp_high[2]^2 + ate_prcp_low[2]^2), 3)
round(ate_prcp_high[1] + ate_prcp_low[1], 3) +
  round(qnorm(0.975) * sqrt(ate_prcp_high[2] ^2 + ate_prcp_low[2] ^ 2), 3)

high_wind <- cate_df$avg_wind_speed > median(cate_df$avg_wind_speed)
ate_wind_high <- average_treatment_effect(forest_cluster, 
                                          subset = high_wind, target.sample = "overlap")
ate_wind_low <- average_treatment_effect(forest_cluster,
                                         subset = !high_wind, target.sample = "overlap")
round(ate_wind_high[1] - ate_wind_low[1], 3) -
  round(qnorm(0.975) * sqrt(ate_wind_high[2]^2 + ate_wind_low[2]^2), 3)
round(ate_wind_high[1] + ate_wind_low[1], 3) +
  round(qnorm(0.975) * sqrt(ate_wind_high[2] ^2 + ate_wind_low[2] ^ 2), 3)


# Visualize partial effect of temperature and wind
# Create test X from all combinations of 5% quantile values of each variable
cov <- c("valueTMAX_MIDWAY", "valuePRCP_MIDWAY", "avg_wind_speed")
cov_quantile <- map(cov, ~ unique(quantile(X_orig[[.x]],probs = seq(0, 1, 0.05))))
names(cov_quantile) <- cov

dummy_var_names <- colnames(X_orig)[!colnames(X_orig) %in% c(cov)]
dummy_zeros <- map(dummy_var_names, \(x) 0)  # Variable importance of FE is all zero
names(dummy_zeros) <- dummy_var_names
X_test <- expand_grid(!!!cov_quantile,
                      !!!dummy_zeros
                      )
tau_pred_test <- predict(forest_cluster, X_test, estimate.variance = TRUE)
cate_test_df <- tibble(
  tau_hat = tau_pred_test$predictions,
  var_hat = tau_pred_test$variance.estimates,
  max_temp = X_test$valueTMAX_MIDWAY,
  prcp  = X_test$valuePRCP_MIDWAY,
  avg_wind_speed = X_test$avg_wind_speed
)

temp_cate <- cate_test_df |> 
  filter((avg_wind_speed == median(avg_wind_speed)) & 
           (prcp == unique(cate_test_df$prcp)[4])) |> 
  mutate(lower = tau_hat - sqrt(var_hat) * qnorm(0.975),
         upper = tau_hat + sqrt(var_hat) * qnorm(0.975)) |> 
  ggplot(aes(x = max_temp, y = tau_hat)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#69b3a2", alpha = 0.5) +
  geom_line(linewidth = 0.8, color = "black", alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "black", alpha = 0.5) +
  labs(x = "Maximum temperature", y = "CATE") +
  theme_custom +
  theme(plot.margin = margin(5, 10, 5, 3),
        axis.title.y = element_text(margin = margin(0,0,0,0)),)
  

wind_cate <- cate_test_df |> 
  filter((max_temp == median(max_temp)) & 
           (prcp == unique(cate_test_df$prcp)[4])) |> 
  mutate(lower = tau_hat - sqrt(var_hat) * qnorm(0.975),
         upper = tau_hat + sqrt(var_hat) * qnorm(0.975)) |> 
  ggplot(aes(x = avg_wind_speed, y = tau_hat,)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#69b3a2", alpha = 0.5) +
  geom_line(linewidth = 0.8, color = "black", alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "black", alpha = 0.5) +
  labs(x = "Wind speed", y = "CATE") +
  theme_custom + 
  theme(plot.margin = margin(5, 3, 5, 10),
        axis.title.y = element_text(margin = margin(0,0,0,0)),)

temp_wind_cate_plot <- grid.arrange(temp_cate, wind_cate, nrow = 1)
temp_wind_cate_plot
ggsave(file.path(output_path, "figures/cate_temp_wind.svg"), temp_wind_cate_plot,
       width = 7, height = 4, units = "in", dpi = 300)


prcp_cate_plot <- cate_test_df |> 
  filter((avg_wind_speed == median(avg_wind_speed)) & 
           (max_temp == unique(cate_test_df$max_temp))) |> 
  mutate(lower = tau_hat - sqrt(var_hat) * qnorm(0.975),
         upper = tau_hat + sqrt(var_hat) * qnorm(0.975)) |> 
  ggplot(aes(x = prcp, y = tau_hat)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#69b3a2", alpha = 0.5) +
  geom_line(linewidth = 0.8, color = "black", alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "black", alpha = 0.5) +
  labs(x = "Precipitation", y = "CATE") +
  theme_custom

ggsave(file.path(output_path, "figures/cate_prcp.svg"), prcp_cate_plot,
       width = 7, height = 4, units = "in", dpi = 300)


# Violin plot of CATE by interstate
cate_by_interstate <- cate_df |> 
  mutate(route_id = factor(route_id, 
                           labels = c("I290", "I55", "I57", "I90A", 
                                      "I90B", "I90C", "I94"))) |> 
  ggplot(aes(route_id, y = tau_hat, group = route_id, fill = route_id)) +
  geom_violin(width = 1, alpha = 0.8) +
  geom_boxplot(width = 0.1, color = "black", alpha = 0.5) +
  labs(x = "Interstate", y = "CATE") +
  theme_custom +
  scale_fill_okabeito() +
  theme(legend.position = "none")
ggsave(file.path(output_path, "figures/cate_by_interstate.svg"), cate_by_interstate,
       width = 7, height = 4, units = "in", dpi = 300)










