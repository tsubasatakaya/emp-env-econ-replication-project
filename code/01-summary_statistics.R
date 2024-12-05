city_data <- read_csv(file.path(data_path, "chicago_citylevel_dataset.csv")) |> 
  arrange(date)

city_data <- city_data |> 
  mutate(wind_speed_kph = avg_wind_speed * 1000 / 3600,
         dew_point_avg = dew_point_avg / 10,
         sealevel_pressure_avg = sealevel_pressure_avg / 10,
         avg_sky_cover = avg_sky_cover * 100)

city_varlist <- c("total_violent", "total_property", "PRCP_MIDWAY", "TMAX_MIDWAY",
                  "avg_co_mean", "avg_no2_mean", "avg_ozone_mean", "avg_pm10_mean",
                  "wind_speed_kph", "dew_point_avg", "sealevel_pressure_avg",
                  "avg_sky_cover")

city_eff_N <- nrow(city_data |> drop_na(any_of(city_varlist)))

city_summary <- city_data |> 
  select(date, all_of(city_varlist)) |> 
  pivot_longer(!date, names_to = "variable", values_to = "value") |> 
  group_by(factor(variable,
                  levels = city_varlist)) |> 
  summarize(mean = mean(value, na.rm = TRUE),
            sd = sd(value, na.rm = TRUE))

colnames(city_summary) <- c("variable", "mean", "sd")


micro_data <- read_csv(file.path(data_path, "micro_dataset_original.csv"), lazy = TRUE) |> 
  filter(insample == 1) |> 
  arrange(date) |> 
  collect()

micro_eff_N <- micro_data |> 
  group_by(violent) |>
  summarize(n = n()) |>
  select(n) |> 
  distinct() |> 
  pull(n)

micro_summary <- micro_data |> 
  select(violent, num_crimes) |> 
  group_by(factor(violent, 
                  levels = c(1, 0),
                  labels = c("violent_num", "nonviolent_num"))) |> 
  summarize(mean = mean(num_crimes, na.rm = TRUE),
            sd = sd(num_crimes, na.rm = TRUE),)
colnames(micro_summary) <- c("variable", "mean", "sd")



summary_data <- city_summary |>
  mutate(variable = as.character(variable)) |> 
  bind_rows(micro_summary |> mutate(variable = as.character(variable)))

summary_data |> 
  gt() |> 
  cols_label(
    variable = "",
    mean = "Mean",
    sd = "Standard deviation",
    .fn = md
  ) |> 
  rows_add(variable = "Citywide sample:",
           mean = NA, sd = NA,
           .before = 1) |> 
  rows_add(variable = "Number of dates",
           mean = city_eff_N, sd = NA,
           .after = 1) |> 
  rows_add(variable = NA,
           mean = NA, sd = NA,
           .after = 14) |> 
  rows_add(variable = "Interstate sample:",
           mean = NA, sd = NA,
           .after = 15) |> 
  rows_add(variable = "Interstate-side days",
           mean = micro_eff_N, sd = NA,
           .after = 16) |> 
  fmt_number(columns = c("mean"),
             rows = c(2, 17),
             decimal = 0) |> 
  fmt_number(columns = c("mean"),
             rows = c(3, 4, 6, 10, 11, 13, 14, 18, 19),
             decimals = 1) |> 
  fmt_number(columns = c("mean"),
             rows = c(5, 7, 12),
             decimals = 2) |> 
  fmt_number(columns = c("sd"),
             rows = c(3, 4, 6, 10, 12, 14, 18, 19),
             decimals = 1) |> 
  fmt_number(columns = c("sd"),
             rows = c(5, 7, 11, 13),
             decimals = 2) |> 
  fmt_number(columns = c("mean", "sd"),
             rows = c(8, 9),
             decimals = 3) |> 
  fmt_number(columns = c("sd"),
             rows = 8,
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
  text_replace("violent_num", "Daily interstate-side violent crimes") |> 
  text_replace("nonviolent_num", "Daily interstate-side property crimes") |> 
  cols_align(align = "left",
             columns = "variable") |> 
  sub_missing(missing_text = "") |> 
  tab_options(table.border.top.style = "hidden",
              heading.align = "left",
              table.font.size = "10pt",
              table.width = pct(100),) |> 
  gtsave("table_1_rep.tex", path = file.path(output_path, "tables"))






























