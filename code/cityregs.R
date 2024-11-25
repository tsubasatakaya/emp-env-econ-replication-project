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



