set.seed(424)

micro_data <- read_csv(file.path(data_path, "micro_dataset_original.csv")) |> 
  mutate(avg_wind_speed = avg_wind_speed / 10) |> 
  arrange(date)

fe_cov <- c("routeside")
weather_cov <- c("tmax", "valuePRCP_MIDWAY", "avg_wind_speed")
treatment <- "treatment"
dvs <- c("violent", "stand_crimes")
vars_all <- c(fe_cov, weather_cov, treatment, dvs)

micro_data <- micro_data |> 
  filter(insample == 1) |> 
  drop_na(any_of(vars_all)) |> 
  mutate(across(c(routeside), as.numeric),
         route_id = as.numeric(factor(route_num1_mod)))

violent_data <- micro_data |> 
  filter(violent == 1)
property_data <- micro_data |> 
  filter(violent == 0)

route_id_v <- violent_data$route_id
X_raw_v <- violent_data |> select(all_of(c(fe_cov, weather_cov)), side_dummy)
D_v <- violent_data |> pull(treatment)
Y_v <- violent_data |> pull("stand_crimes")

cat_vars <- c("routeside")
X_v <- dummy_cols(X_raw_v, select_columns = cat_vars,
                  remove_first_dummy = TRUE) |> select(-all_of(cat_vars))
X_v <- X_v |> 
  mutate(across(contains("side_dummy"), .fns =  ~ . * tmax, .names = "{.col}_x_tmax"),
         across(contains("side_dummy"), .fns =  ~ . * valuePRCP_MIDWAY, .names = "{.col}_x_prcp"))

Y_forest <- regression_forest(X_v, Y_v, clusters = route_id_v, equalize.cluster.weights = TRUE)
Y_hat <- predict(Y_forest)$predictions
W_forest <- regression_forest(X_v, D_v, clusters = route_id_v, equalize.cluster.weights = TRUE)
W_hat <- predict(W_forest)$predictions

forest_v <- causal_forest(X = X_v,
                          Y = Y_v,
                          W = D_v,
                          Y.hat = Y_hat, W.hat = W_hat,
                          clusters = route_id_v,
                          equalize.cluster.weights = TRUE,
                          )

average_treatment_effect(forest_v, target.sample = "overlap")

test_calibration(forest_v, vcov.type = "HC1")

best_linear_projection(forest_v, X_v[, c("tmax", "valuePRCP_MIDWAY", "avg_wind_speed")],
                       target.sample = "overlap",
                       vcov.type = "HC1")













