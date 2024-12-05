city_data <- read_csv(file.path(data_path, "chicago_citylevel_dataset.csv")) |> 
  arrange(date)

city_data <- city_data |> 
  mutate(across(c(max_temp_bins, dew_point_bins, ym, dow, wind_bins_20), as.factor))

weather_cov <- c("avg_wind_speed", "max_temp_bins", "dew_point_bins", "PRCP_MIDWAY",
                 "sealevel_pressure_avg", "avg_sky_cover")
calendar_cov <- c("ym", "dow", "month1", "jan1", "holiday")
poll_cov <- c("avg_co_mean", "avg_no2_mean", "avg_ozone_mean")
hist_temp_cov <- c("mean_TMAX_1991_2000")
iv <- c("wind_bins_20")
treatment <- "standardized_pm"


fit_ols <- function(dv, treatment, cov) {
   formula <- as.formula(
     paste0(dv, "~", 
            paste0(c(treatment, cov), collapse = "+")))
   res <- feols(formula, data = city_data, vcov = vcov_NW(lag = 1, time = ~ date))
}

fit_iv <- function(dv, endo, cov, iv) {
  formula <- as.formula(
    paste0(dv, "~",
           paste0(cov, collapse = "+"),
           "|",
           paste0(endo,
                  "~",
                  iv))
  )
  res <- feols(formula, data = city_data, vcov = vcov_NW(lag = 1, time = ~ date))
}

# OLS -------------------------------------------------------------------------
ols_results <- list()

ols_names <- c("(1)", "(2)", "(4)", "(5)")
ols_dv <- c("ln_violent", "ln_violent", "ln_property", "ln_property")
ols_cov <- list(c(calendar_cov), 
                c(calendar_cov, weather_cov, hist_temp_cov),
                c(calendar_cov),
                c(calendar_cov, weather_cov, hist_temp_cov))
for (i in seq_along(ols_names)) {
  res <- fit_ols(ols_dv[i], treatment, pluck(ols_cov, i))
  ols_results[[i]] <- res
}
names(ols_results) <- ols_names

# IV --------------------------------------------------------------------------
iv_results <- list()
iv_names <- c("(3)", "(6)")
iv_dv <- c("ln_violent", "ln_property")
for (i in seq_along(iv_names)) {
  res <- fit_iv(iv_dv[i], treatment, pluck(ols_cov, 2), iv)
  iv_results[[i]] <- res
}
names(iv_results) <- iv_names

# Combine
results <- c(ols_results, iv_results)
results <- results[sort(names(results))]

# Generate table --------------------------------------------------------------
cm <- c("standardized_pm" = "Standardized PM10 reading",
        "fit_standardized_pm" = "Standardized PM10 reading")
gof_f <- function(x) format(round(x, 2), big.mark = ",")
gm <- list(
  list("raw" = "nobs", "clean" = "Observations", "fmt" = gof_f),
  list("raw" = "r.squared", "clean" = "R\U00B2", "fmt" = gof_f)
)

f_violent <- round(fitstat(results[[3]], "ivf")[["ivf1::standardized_pm"]][["stat"]], 1)
f_property <- round(fitstat(results[[6]], "ivf")[["ivf1::standardized_pm"]][["stat"]], 1)

add_rows <- tibble(term = c("First stage F-statistic", "Calendar fixed effects",
                            "Weather controls", "Historical mean temp"),
                   col1 = c(NA, "X", NA, NA),
                   col2 = c(NA, "X", "X", "X"),
                   col3 = c(as.character(f_violent), "X", "X", "X"),
                   col4 = c(NA, "X", NA, NA),
                   col5 = c(NA, "X", "X", "X"),
                   col6 = c(as.character(f_property), "X", "X", "X"))
attr(add_rows, "position") <- c(3, 4, 5, 6)

msummary(results, fmt = 4,
         coef_map = cm, gof_map = gm,
         add_rows = add_rows,
         output = "gt") |> 
  sub_missing(missing_text = "") |> 
  tab_spanner(label = "OLS", columns = c(2, 3, 5, 6), gather = FALSE) |> 
  tab_spanner(label = "IV", columns = c(4, 7), gather = FALSE) |> 
  tab_spanner(label = "Violent crimes", columns = 2:4) |> 
  tab_spanner(label = "Property crimes", columns = 5:7) |> 
  tab_options(table.font.size = "9pt",
              table.width = pct(100),) |>
  gtsave("table_2_rep.tex", path = file.path(output_path, "tables"))



# Make effect plot -------------------------------------------------------------
dv_list <- c(rep("violent", 3), rep("property", 3))
specs <- c("OLS - calendar FE only", "OLS - calendar FE + weather controls",
           "IV - calendar FE + weather controls")
spec_list <- c(rep(specs, 2))

coef_df <- tibble()
for (i in seq_along(results)) {
  df <- tidy(results[[i]], conf.int = TRUE) |> 
    filter(str_detect(term, treatment)) |> 
    mutate(dv = dv_list[i],
           spec = spec_list[i])
  coef_df <- coef_df |> 
    bind_rows(df)
}

coef_df <- coef_df |> 
  mutate(spec = factor(spec, levels = specs),
         dv = factor(dv, levels = c("violent", "property")))

cityreg_plot <- ggplot(coef_df, aes(x = dv, ymin = conf.low, ymax = conf.high)) +
  geom_hline(yintercept = 0, linetype = "dashed") + 
  geom_linerange(aes(col = spec), linewidth = 1, 
                 position = position_dodge(width = 0.15)) +
  geom_point(aes(x = dv, y = estimate, col = spec), size = 3,
             position = position_dodge(width = 0.15)) + 
  scale_x_discrete(labels = c("Violent crimes", "Property crimes")) + 
  scale_color_okabeito() +
  theme_minimal() +
  labs(x = "", y = "Treatment effect\n") + 
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        panel.border = element_rect(color = "grey", fill = NA),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.title = element_text(size = 12,),
        axis.text = element_text(size = 11),
        axis.text.x = element_text(face = "bold"),
        legend.text = element_text(size = 10))
ggsave(file.path(output_path, "figures/cityreg_coef_plot.svg"), cityreg_plot,
       width = 12, height = 7.8, units = "in", dpi = 300
       )








