set.seed(424)

micro_data <- read_csv(file.path(data_path, "micro_dataset_original.csv")) |> 
  mutate(avg_wind_speed = avg_wind_speed / 10) |> 
  arrange(date)

fe_cov <- c("routeside")
weather_cov <- c("tmax", "valuePRCP_MIDWAY", "avg_wind_speed")
dummies <- c("side_dummy")
treatment <- "treatment"
dvs <- c("violent", "stand_crimes")
vars_all <- c(fe_cov, weather_cov, dummies, treatment, dvs)

micro_data <- micro_data |> 
  filter(insample == 1) |> 
  drop_na(any_of(vars_all)) |> 
  mutate(across(c(routeside), as.numeric),
         route_id = as.numeric(factor(route_num1_mod)))

violent_data <- micro_data |> 
  filter(violent == 1)

route_id <- violent_data$route_id
X_raw <- violent_data |> select(all_of(c(fe_cov, weather_cov)), dummies)
D <- violent_data |> pull(treatment)
Y <- violent_data |> pull("stand_crimes")

# Create a list of covariates, FE, and arguments for loop
model_tags <- c("non_clustered", "clustered")
cov_list <- list(
  c(fe_cov, weather_cov),
  c(weather_cov, dummies)
)

fe <- c("routeside", "side_dummy")

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

forest_results <- list()
for (i in seq_along(model_tags)) {
  X <- X_raw |> 
    select(all_of(cov_list[[i]]))
  if (nrow(unique(X[fe[i]])) != 2) {
    X <- dummy_cols(X, select_columns = fe[i],
                    remove_first_dummy = TRUE) |> select(-all_of(fe[i]))
  }
  dummy_colnames <- colnames(X)[str_detect(colnames(X), paste0(fe[i], "(_\\d)?"))]
  X <- X |> 
    mutate(across(dummy_colnames, .fns = ~ . * tmax, .names = "{.col}_x_tmax"),
           across(dummy_colnames, .fns = ~ . * valuePRCP_MIDWAY, .names = "{.col}_x_prcp"))
  
  args <- args_list[[i]]
  args <- append(args, list(X = X))
  
  print(paste("Fitting", model_tags[i], "model..."))
  forest_results[[i]] <- do.call(causal_forest, args)
}

calibration_results <- list()
for (i in seq_along(model_tags)) {
  res <- test_calibration(forest_results[[i]], vcov.type = "HC1")
  calibration_results[[i]] <- res
}

linproj_res <- list()
for (i in seq_along(model_tags)) {
  cf <- forest_results[[i]]
  res <- best_linear_projection(cf,
                                cf$X.orig[, c("tmax", "valuePRCP_MIDWAY", "avg_wind_speed")],
                                target.sample = "overlap",
                                vcov.type = "HC1")
}

ato <- average_treatment_effect(forest_results[[2]], target.sample = "overlap")















