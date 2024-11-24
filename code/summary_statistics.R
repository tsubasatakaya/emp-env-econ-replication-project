city_data <- read_csv(file.path(data_path, "chicago_citylevel_dataset.csv")) |> 
  arrange(date)

city_data <- city_data |> 
  mutate(wind_speed_kph = avg_wind_speed * 1000 / 3600,
         dew_point_avg = dew_point_avg / 10,
         sealevel_pressure_avg = sealevel_pressure_avg / 10,
         avg_sky_cover = avg_sky_cover * 100)

summary_varlist <- c("total_violent", "total_property", "PRCP_MIDWAY", "TMAX_MIDWAY",
                     "avg_co_mean", "avg_no2_mean", "avg_ozone_mean", "avg_pm10_mean",
                     "wind_speed_kph", "dew_point_avg", "sealevel_pressure_avg",
                     "avg_sky_cover")

eff_N <- nrow(city_data |> drop_na(any_of(summary_varlist)))

summary_table <- city_data |> 
  select(date, all_of(summary_varlist)) |> 
  pivot_longer(!date, names_to = "variable", values_to = "value") |> 
  group_by(factor(variable,
                  levels = summary_varlist)) |> 
  summarize(mean = mean(value, na.rm = TRUE),
            sd = sd(value, na.rm = TRUE))

colnames(summary_table) <- c("variable", "mean", "sd")

summary_table |> 
  gt() |> 
  cols_label(
    variable = "",
    mean = "Mean",
    sd = "Standard deviation",
    .fn = md
  ) |> 
  rows_add(variable = "Citywide sample:",
           mean = eff_N, sd = NA,
           .before = 1) |> 
  fmt_number(columns = c("mean"),
             rows = 1,
             decimal = 0) |> 
  fmt_number(columns = c("mean"),
             rows = c(2, 3, 5, 9, 10, 12, 13),
             decimals = 1) |> 
  fmt_number(columns = c("mean"),
             rows = c(4, 6, 11),
             decimals = 2) |> 
  fmt_number(columns = c("sd"),
             rows = c(2, 3, 5, 9, 11, 13),
             decimals = 1) |> 
  fmt_number(columns = c("sd"),
             rows = c(4, 6, 10, 12),
             decimals = 2) |> 
  fmt_number(columns = c("mean", "sd"),
             rows = c(7, 8),
             decimals = 3) |> 
  fmt_number(columns = c("sd"),
             rows = 7,
             decimals = 4) |> 
  text_replace("total_violent", "Daily city-wide violent crime") |> 
  text_replace("total_property", " Daily city-wide property crime") |> 
  text_replace("PRCP_MIDWAY", " Precipitation (mm)") |> 
  text_replace("TMAX_MIDWAY",  " Maximum temperature (°C)") |> 
  text_replace("avg_co_mean", " Daily avg. carbon monoxide (ppm)") |> 
  text_replace("avg_no2_mean", " Daily avg. NO2 (ppm)") |> 
  text_replace("avg_ozone_mean", " Daily avg. ozone (ppm)") |> 
  text_replace("avg_pm10_mean", " Daily avg. PM\U2081\U2080 (\U03BCg/m\U00B3)") |> 
  text_replace("wind_speed_kph", " Wind speed (km/h)") |> 
  text_replace("dew_point_avg", " Dew point (°C)") |> 
  text_replace("sealevel_pressure_avg", " Air pressure (hpa)") |> 
  text_replace("avg_sky_cover", " Cloud cover sunrise to sunset (percent)") |> 
  cols_align(align = "left",
             columns = "variable")






























