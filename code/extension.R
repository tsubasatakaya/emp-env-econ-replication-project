set.seed(424)

city_data <- read_csv(file.path(data_path, "chicago_citylevel_dataset.csv")) |> 
  arrange(date)

# city_data <- city_data |> 
#   mutate(across(c(max_temp_bins, dew_point_bins, ym, dow, wind_bins_20), as.factor))

missing_pct <- city_data |> 
  summarize(across(everything(), ~mean(is.na(.)) * 100))


# Continuous var
# standardized_pm (treatment), PRCP_MIDWAY, sealevel_pressure_avg, avg_sky_cover,
# avg_wind_speed

# Categorical var
# wind_bins_20 (from wind_dir_avg), max_temp_bins (from TMAX_MIDWAY), 
# dew_point_bins (from dew_point_avg),
# ym, dow

# Dummy var
# month1, jan1, holiday

city_data <- city_data |> 
  mutate(wind_binary = 
           case_when((wind_bins_20 >= 8) & (wind_bins_20 <= 12) ~ 1,
                     .default = 0)) |> 
  mutate(across(c(max_temp_bins, dew_point_bins, ym, dow), as.factor))

weather_cov <- c("avg_wind_speed", "max_temp_bins", "dew_point_bins", "PRCP_MIDWAY",
                 "sealevel_pressure_avg", "avg_sky_cover")
calendar_cov <- c("ym", "dow", "month1", "jan1", "holiday")
hist_temp_cov <- c("mean_TMAX_1991_2000")
treatment <- "standardized_pm"
iv <- c("wind_bins_20")
dvs <- c("ln_violent", "ln_property")
vars_all <- c(weather_cov, calendar_cov, hist_temp_cov, treatment, iv, dvs)

city_data <- city_data |>
  drop_na(vars_all)

X_raw <- city_data |> select(all_of(c(weather_cov, calendar_cov, hist_temp_cov)))
D <- city_data |> pull(treatment)
Z <- city_data |> pull("wind_binary")
Y_v <- city_data |> pull("ln_violent")
Y_p <- city_data |> pull("ln_property")
  

cat_vars <- c("max_temp_bins", "dew_point_bins", "ym", "dow")
X <- dummy_cols(X_raw, select_columns = cat_vars,
                remove_first_dummy = TRUE) |> select(-all_of(cat_vars))

forest_iv_base <- instrumental_forest(X = X,
                                      Y = Y_v,
                                      W = D,
                                      Z = Z,
                                      num.trees = 2000,
                                      tune.parameters = "all")
tau_hat <- predict(forest_iv_base)$predictions
hist(tau_hat)
average_treatment_effect(forest_iv_base)


num_trees <- c(2000, 5000, 8000, 10000, 15000, 20000)
res_l <- list()
for (i in seq_along(num_trees)) {
  Y_forest <- regression_forest(X, Y_v)
  Y_hat <- predict(Y_forest)$predictions
  D_forest <- regression_forest(X, D)
  D_hat <- predict(D_forest)$predictions
  Z_forest <- regression_forest(X, Z)
  Z_hat <- predict(Z_forest)$predictions
  f_raw <- instrumental_forest(X = X,
                               Y = Y_v,
                               W = D,
                               Z = Z,
                               Y.hat = Y_hat, W.hat = D_hat,
                               Z.hat = Z_hat,
                               num.trees = num_trees[i])
  varimp <- variable_importance(f_raw)
  f_final <- instrumental_forest(X = X[, which(varimp > mean(varimp))],
                                 Y = Y_v,
                                 W = D,
                                 Z = Z,
                                 Y.hat = Y_hat, W.hat = D_hat,
                                 Z.hat = Z_hat,
                                 num.trees = num_trees[i],)
  
  res_l[[i]] <- f_final
}

names(res_l) <- num_trees

for (i in seq_along(num_trees)) {
  print(mean(predict(res_l[[i]])$predictions))
}

tau_hat <- predict(res_l[[4]])$predictions

hist(predict(res_l[[3]])$predictions)

df <- cbind(Y_v, D, Z, X)

formula_string <- paste("ln_violent ~", 
                        paste(c(weather_cov, calendar_cov, hist_temp_cov), collapse = "+"),
                        "| standardized_pm ~ wind_dir_avg")

summary(feols(as.formula(formula_string), data = city_data))

ggplot(city_data, aes(x = wind_binary, y = standardized_pm)) +
  geom_point()




# Microgeographic analysis --------------------------------------------------------------------
set.seed(424)

micro_data <- read_csv(file.path(data_path, "micro_dataset_original.csv")) |> 
  arrange(date)













































