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
  bind_rows(micro_summary |> mutate(variable = as.character(variable))) |> 
  mutate(variable = case_match(variable,
                               "total_violent" ~ "Daily citywide violent crime",
                               "total_property" ~ "Daily citywide property crime",
                               "PRCP_MIDWAY" ~ "Precipitation (mm)",
                               "TMAX_MIDWAY" ~  "Maximum temperature (°C)",
                               "avg_co_mean" ~ "Daily average CO (ppm)",
                               "avg_no2_mean" ~ "Daily average NO2 (ppm)",
                               "avg_ozone_mean" ~ "Daily average ozone (ppm)",
                               "avg_pm10_mean" ~ "Daily average PM10 (mug/m3)",
                               "wind_speed_kph" ~ "Wind speed (km/h)",
                               "dew_point_avg" ~ "Dew point (°C)",
                               "sealevel_pressure_avg" ~ "Air pressure (hpa)",
                               "avg_sky_cover" ~ " Cloud cover sunrise to sunset (percent)",
                               "violent_num" ~ "Daily interstate-side violent crimes",
                               "nonviolent_num" ~ "Daily interstate-side property crimes"
                               ))

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
  cols_align(align = "left",
             columns = "variable") |> 
  cols_align(align = "center",
             columns = "mean") |> 
  cols_align(align = "center",
             columns = "sd") |> 
  cols_width(mean ~ px(80),
             sd ~ px(80),) |> 
  sub_missing(missing_text = "") |> 
  tab_options(table.border.top.style = "hidden",
              heading.align = "left",
              table.font.size = "9pt",
              table.width = pct(100),) |>
  gtsave("table_1_rep.tex", path = file.path(output_path, "tables"))






























