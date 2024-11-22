import os
from pathlib import Path

import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
from billiard.five import values


class DataPreprocessor:
    def __init__(self,
                 input_data_path: Path,
                 output_data_path: Path,):
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path

    def _extract_crime_data(self):
        crime_data = (pl.read_csv(self.input_data_path / "chicago_crime.csv")
                      .rename(lambda col: col.lower().replace(" ", "_"))
                      .rename({"date": "string_date"}))

        crime_data = (crime_data
                      .filter(pl.col("year").is_between(2001, 2012, closed="both"))
                      .with_columns(
            pl.col("string_date").str.to_datetime(format="%m/%d/%Y %I:%M:%S %p"))
                      .with_columns(
            pl.col("string_date").dt.date().alias("date"),
            pl.col("string_date").dt.hour().alias("hour"),
            pl.col("string_date").dt.minute().alias("minute"),
            pl.col("string_date").dt.second().alias("second"), )
                      .with_columns(
            pl.when(pl.col("fbi_code").is_in(["01A", "02", "03", "04A", "04B", "05", "06", "07", "09"]))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("part1"),
            # homicide, offense involving children/sexual assault, robbery, assault, battery/ritualism,
            # burglary, theft, motor vehicle theft, arson
            pl.when(pl.col("fbi_code").is_in(["01A", "02", "04A", "04B", "08A", "08B"]))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("violent"), )  # homicide, offense involving children/sexual assault, assault, battery/ritualism,
                      # assault/stalking, ritualism/battery/domestic violence
                      .with_columns(
            pl.when(pl.col("fbi_code") == "08")
            .then(pl.lit("09"))
            .otherwise(pl.col("fbi_code"))
            .alias("fbi_code"))
                      .drop("string_date")
                      )

        # Save part1 crime data
        crime_data.filter(
            pl.col("part1") == 1
        ).write_csv(self.output_data_path/"chicago_part1_crimes.csv")

        # Save all crimes
        crime_data.drop(
            ["block", "description", "location_description", "beat", "district",
             "ward", "community_area", "x_coordinate", "y_coordinate", "location"]
        ).write_csv(self.output_data_path/"chicago_all_crimes.csv")

    def _extract_crime_interstate_distance(self):
        crime_interstate_data = (pl.read_csv(self.input_data_path/"Chicago_Crime_Interstate_Distance_0606.csv")
                                 .rename(lambda col: col.lower().replace(" ", "_")))

        crime_interstate_data = (crime_interstate_data
                                 .drop(
            ["objectid", "feat_seq", "frequency", "shape_length"])
                                 .sort("id", "near_dist")
                                 .with_columns(
            pl.when(pl.col("id") == pl.col("id").shift(1))
            .then(pl.lit(2))
            .otherwise(pl.lit(1))
            .alias("rank"))
                                 .drop("near_rank")
        )

        crime_interstate_wide = crime_interstate_data.pivot(
            on="rank",
            index=["id", "latitude", "longitude",],
        )

        crime_interstate_wide = (crime_interstate_wide
                                 .with_columns(
            [(pl.col(f"near_angle_{i}") % 360)for i in [1, 2]])
                                 .with_columns(
            [pl.col(f"near_angle_{i}").radians().alias(f"near_dir_{i}") for i in [1, 2]])
                                 )

        crime_interstate_wide = (crime_interstate_wide
                                 .with_columns(
            pl.when(
                (pl.col("route_num_1") == "I90") & (pl.col("latitude") > 41.84) & (pl.col("longitude") > -87.75))
            .then(pl.lit("I90_A"))
            .when(
                (pl.col("route_num_1") == "I90") & (pl.col("latitude") < 41.84) & (pl.col("latitude") > 41.775))
            .then(pl.lit("I90_B"))
            .when(
                (pl.col("route_num_1") == "I90") & (pl.col("latitude") < 41.775))
            .then(pl.lit("I90_C"))
            .otherwise(pl.col("route_num_1"))
            .alias("route_num_1_mod"))
                                 )

        crime_interstate_wide = (crime_interstate_wide
                                 .with_columns(
            pl.when(
                # Drop observations on fringe of city limits
                (pl.col("longitude") < -87.8) |
                # Drop observations far out I-290 or I-55
                ((pl.col("route_num_1").is_in(["I290", "I55"])) & (pl.col("longitude") < -87.74)) |
                # Drop observations more than one mile from the closest interstate
                (pl.col("near_dist_1") > 5280) |
                # Drop observations closer than one mile to two interstates
                (pl.col("near_dist_2") < 5280) |
                # Trim observations to clean treatment/control groups
                # I-90A
                ((pl.col("latitude") - pl.col("longitude")) > 129.69) |
                (((pl.col("latitude") - pl.col("longitude")) < 129.575) & (pl.col("route_num_1_mod") == "I90_A")) |
                # I-90B
                ((pl.col("latitude") < 41.79) & (pl.col("route_num_1_mod") == "I90_B")) |
                # I-90C
                (((pl.col("latitude") - pl.col("longitude")) > 129.36) & (pl.col("route_num_1_mod") == "I90_C")) |
                (((pl.col("latitude") - pl.col("longitude")) < 129.26) & (pl.col("route_num_1_mod") == "I90_C")) |
                # I-55
                ((pl.col("longitude") > -87.65) & (pl.col("route_num_1") == "I55")) |
                # I-94
                (((pl.col("latitude") - pl.col("longitude")) < 129.34) & (pl.col("route_num_1_mod") == "I94")) |
                ((pl.col("latitude") > 41.75) & (pl.col("route_num_1_mod") == "I94"))
            )
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("sample_set")
        )
                                 )

        crime_interstate_wide.write_csv(self.output_data_path/"crime_road_distances.csv")

    def _extract_chicago_aqi(self):
        aqi_data = pd.read_stata(self.input_data_path/"chicago_aqi_2000_2015.dta")
        aqi_data = pl.from_pandas(aqi_data)
        aqi_data.write_csv(self.output_data_path/"chicago_aqi_2000_2015.csv")

    def _extract_chicago_co(self):
        temp_1 = pl.read_csv(self.input_data_path/"co_chicago_20000101_20041231.txt",
                             separator=",", null_values=["END OF FILE"])
        temp_2 = pl.read_csv(self.input_data_path/"co_chicago_20050101_20091231.txt",
                             separator=",", null_values=["END OF FILE"])
        temp_3 = pl.read_csv(self.input_data_path/"co_chicago_20100101_20121231.txt",
                             separator=",", null_values=["END OF FILE"])

        co_data = (pl.concat([temp_1, temp_2, temp_3])
                   .rename(lambda col: col.lower().replace(" ", "_"))
                   .with_columns(
            (pl.col("county_code").cast(pl.String) + "_" + pl.col("site_num").cast(pl.String)
             + "_" + pl.col("poc").cast(pl.String)).alias("monitor_id")
        )
                   )

        # # Save hourly data
        # hourly_co_data = (co_data
        #                   .filter(
        #     pl.col("sample_duration") == "1 HOUR")
        #                   .with_columns(
        #     pl.col("sample_measurement")
        #     .is_not_null()
        #     .cast(pl.Int64)
        #     .sum()
        #     .over(["monitor_id", "date_local"])
        #     .alias("num_hrly_obs"))
        #                   .filter(
        #     pl.col("num_hrly_obs") >= 18)
        #                   .with_columns(
        #     pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
        #                   .with_columns(
        #     pl.col("date_local").dt.date().alias("date"),
        #     pl.col("24_hour_local").str.split(":").list.get(0).cast(pl.Int64).alias("hour"))
        #                   .rename({"sample_measurement": "co_hrly"})
        #                   .select("latitude", "longitude", "datum", "state_code", "county_code",
        #                           "monitor_id", "site_num", "poc", "co_hrly", "date", "hour")
        #                   )
        #
        # hourly_co_data.write_csv(self.output_data_path/"chicago_co_2000_2012_hourly.csv")

        # Save daily data
        daily_co_data = (co_data
                         .filter(
            pl.col("sample_duration") == "1 HOUR")
                         .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local", "sample_duration"])  # this sample_duration is redundant
            .alias("num_hrly_obs_co"),
            pl.col("sample_measurement")
            .max()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("max_co"),
            pl.col("sample_measurement")
            .mean()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("avg_co"))
                         .with_columns(
            pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
                         .with_columns(
            pl.col("date_local").dt.date().alias("date"))
                         .sort("monitor_id", "date")
                         .unique(["monitor_id", "date"])
                         .filter(
            pl.col("num_hrly_obs_co") >= 18)  # original code uses num_hrly_obs, but they are equivalent
                         .sort("monitor_id", "date")
                         .drop(cs.contains("gmt"))
                         )

        daily_co_data.write_csv(self.output_data_path/"chicago_co_2000_2012_daily.csv")

    def _extract_chicago_pm10(self):
        temp_1 = pl.read_csv(self.input_data_path/"pm10_chicago_20000101_20041231.txt",
                             separator=",", null_values=["END OF FILE"], infer_schema_length=10000)
        temp_2 = pl.read_csv(self.input_data_path/"pm10_chicago_20050101_20091231.txt",
                             separator=",", null_values=["END OF FILE"], infer_schema_length=10000)
        temp_3 = pl.read_csv(self.input_data_path/"pm10_chicago_20100101_20121231.txt",
                             separator=",", null_values=["END OF FILE"], infer_schema_length=10000)

        pm_data = (pl.concat([temp_1, temp_2, temp_3])
        .rename(lambda col: col.lower().replace(" ", "_"))
        .with_columns(
            (pl.col("county_code").cast(pl.String) + "_" + pl.col("site_num").cast(pl.String)
             + "_" + pl.col("poc").cast(pl.String)).alias("monitor_id")
        )
        )

        # # Save hourly data
        # hourly_pm_data = (pm_data
        #                   .filter(
        #     pl.col("sample_duration") == "1 HOUR")
        #                   .with_columns(
        #     pl.col("sample_measurement")
        #     .is_not_null()
        #     .cast(pl.Int64)
        #     .sum()
        #     .over(["monitor_id", "date_local"])
        #     .alias("num_hrly_obs"))
        #                   .filter(
        #     pl.col("num_hrly_obs") >= 18)
        #                   .with_columns(
        #     pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
        #                   .with_columns(
        #     pl.col("date_local").dt.date().alias("date"),
        #     pl.col("24_hour_local").str.split(":").list.get(0).cast(pl.Int64).alias("hour"))
        #                   .rename({"sample_measurement": "pm10hrly"})
        #                   .select("latitude", "longitude", "datum", "state_code", "county_code",
        #                           "monitor_id", "site_num", "poc", "pm10hrly", "date", "hour")
        #                   )
        #
        # hourly_pm_data.write_csv(self.output_data_path/"chicago_pm10_2000_2012_hourly.csv")

        # Save daily data
        daily_pm_data = (pm_data
                         .filter(
            ~pl.all_horizontal(pl.all().is_null()))
                         .with_columns(
            pl.col("sample_frequency").str.strip_chars())
                         .with_columns(
            pl.when(pl.col("sample_frequency") == "")
            .then(None)
            .otherwise(pl.col("sample_frequency"))
            .alias("sample_frequency"))
                         .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("temp_obs"),
            pl.col("sample_measurement")
            .max()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("temp_max"),
            pl.col("sample_measurement")
            .mean()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("temp_avg"))
                         .with_columns(
            [pl.when(pl.col("sample_duration") == "24-HR BLK AVG")
             .then(None)
             .otherwise(pl.col(col))
             .alias(col) for col in ["temp_obs", "temp_max", "temp_avg"]])
                         .with_columns(
            pl.col("temp_obs").max().over(["monitor_id", "date_local"]).alias("num_hrly_obs_pm10"),
            pl.col("temp_max").max().over(["monitor_id", "date_local"]).alias("max24hr_pm10_derived"),
            pl.col("temp_avg").max().over(["monitor_id", "date_local"]).alias("avg24hr_pm10_derived"),)
                         .sort("monitor_id", "date_local", "24_hour_local", "sample_frequency")
                         .with_columns(
            pl.when(
                pl.col("sample_frequency").is_null() &
                (pl.col("sample_duration") == "24 HOUR") &
                (pl.col("monitor_id") == pl.col("monitor_id").shift(1)))
            .then(pl.col("sample_frequency").shift(1))
            .otherwise(pl.col("sample_frequency"))
            .alias("sample_frequency"))
                         )

        daily_pm_data = (daily_pm_data
                         .filter(
            (pl.col("sample_duration") == "24-HR BLK AVG") |
            ((pl.col("sample_duration") == "24 HOUR") & (pl.col("sample_frequency") == "EVERY DAY")))
                         .with_columns(
            pl.when(pl.col("sample_duration") == "24 HOUR")
            .then(None)
            .otherwise(pl.col("num_hrly_obs_pm10"))
            .alias("num_hrly_obs_pm10"))
                         )

        daily_pm_data = (daily_pm_data
                         .with_columns(
            pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
                         .with_columns(
            pl.col("date_local").dt.date().alias("date"))
                         .drop(cs.contains("gmt"))
                         .rename({"sample_measurement": "daily_pm10_notderived"})
                         .with_columns(
            pl.when(pl.col("num_hrly_obs_pm10").is_null())
            .then(None)
            .otherwise(pl.col("max24hr_pm10_derived"))
            .alias("max24hr_pm10_derived"))
        )

        daily_pm_data.write_csv(self.output_data_path/"chicago_pm10_2000_2012_daily.csv")

    def _extract_chicago_no2(self):
        temp_1 = pl.read_csv(self.input_data_path/"no2_chicago_20000101_20041231.txt",
                             separator=",", null_values=["END OF FILE"],
                             schema_overrides={col: pl.String for col in ["Sample Measurement", "Horizontal Accuracy"]},
                             infer_schema_length=10000)
        temp_2 = pl.read_csv(self.input_data_path/"no2_chicago_20050101_20091231.txt",
                             separator=",", null_values=["END OF FILE"],
                             schema_overrides={col: pl.String for col in ["Sample Measurement", "Horizontal Accuracy"]},
                             infer_schema_length=10000)
        temp_3 = pl.read_csv(self.input_data_path/"no2_chicago_20100101_20121231.txt",
                             separator=",", null_values=["END OF FILE"],
                             schema_overrides={col: pl.String for col in ["Sample Measurement", "Horizontal Accuracy"]},
                             infer_schema_length=10000)

        no_data = (pl.concat([temp_1, temp_2, temp_3])
        .rename(lambda col: col.lower().replace(" ", "_"))
        .with_columns(
            (pl.col("county_code").cast(pl.String) + "_" + pl.col("site_num").cast(pl.String)
             + "_" + pl.col("poc").cast(pl.String)).alias("monitor_id"),)
        .with_columns(
            pl.col("sample_measurement").cast(pl.Float64),
            pl.col("horizontal_accuracy").cast(pl.Float64)
        )
        )

        # # Save hourly data
        # hourly_no_data = (no_data
        #                   .filter(
        #     pl.col("sample_duration") == "1 HOUR")
        #                   .with_columns(
        #     pl.col("sample_measurement")
        #     .is_not_null()
        #     .cast(pl.Int64)
        #     .sum()
        #     .over(["monitor_id", "date_local"])
        #     .alias("num_hrly_obs"))
        #                   .filter(
        #     pl.col("num_hrly_obs") >= 18)
        #                   .with_columns(
        #     pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
        #                   .with_columns(
        #     pl.col("date_local").dt.date().alias("date"),
        #     pl.col("24_hour_local").str.split(":").list.get(0).cast(pl.Int64).alias("hour"))
        #                   .rename({"sample_measurement": "no2_hrly"})
        #                   .with_columns(
        #     (pl.col("no2_hrly") / 1000).alias("no2_hrly"))
        #                   .select("latitude", "longitude", "datum", "state_code", "county_code",
        #                           "monitor_id", "site_num", "poc", "no2_hrly", "date", "hour")
        #                   )
        #
        # hourly_no_data.write_csv(self.output_data_path/"chicago_no2_2000_2012_hourly.csv")

        # Save daily data
        daily_no_data = (no_data
                         .filter(
            (~pl.all_horizontal(pl.all().is_null())) |
            (pl.col("sample_duration") == "8-HR RUN AVG BEGIN HOUR"))
                         .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local"])
            .alias("num_hrly_obs"))
                         .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("num_hrly_obs_no2"),
            pl.col("sample_measurement")
            .max()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("max_no2"),
            pl.col("sample_measurement")
            .mean()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("avg_no2"))
                         .with_columns(
            (pl.col("max_no2") / 1000).alias("max_no2"),
            (pl.col("avg_no2") / 1000).alias("avg_no2"),)
                         .with_columns(
            pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
                         .with_columns(
            pl.col("date_local").dt.date().alias("date"))
                         .sort("monitor_id", "date")
                         .unique(["monitor_id", "date"])
                         .filter(
            pl.col("num_hrly_obs") >= 18)
                         .sort("monitor_id", "date")
                         .drop(cs.contains("gmt"), pl.col("num_hrly_obs"))
                         )

        daily_no_data.write_csv(self.output_data_path / "chicago_no2_2000_2012_daily.csv")

    def _extract_chicago_ozone(self):
        temp_1 = pl.read_csv(self.input_data_path/"ozone_chicago_20000101_20050101.txt",
                             separator=",", null_values=["END OF FILE"],
                             schema_overrides={col: pl.String for col in ["Horizontal Accuracy"]},
                             infer_schema_length=10000)
        temp_2 = pl.read_csv(self.input_data_path/"ozone_chicago_20050102_20091231.txt",
                             separator=",", null_values=["END OF FILE"],
                             schema_overrides={col: pl.String for col in ["Horizontal Accuracy"]},
                             infer_schema_length=10000)
        temp_3 = pl.read_csv(self.input_data_path/"ozone_chicago_20100101_20121231.txt",
                             separator=",", null_values=["END OF FILE"],
                             schema_overrides={col: pl.String for col in ["Horizontal Accuracy"]},
                             infer_schema_length=10000)

        ozone_data = (pl.concat([temp_1, temp_2, temp_3])
        .rename(lambda col: col.lower().replace(" ", "_"))
        .with_columns(
            (pl.col("county_code").cast(pl.String) + "_" + pl.col("site_num").cast(pl.String)
             + "_" + pl.col("poc").cast(pl.String)).alias("monitor_id"),)
        .with_columns(
            pl.col("horizontal_accuracy").cast(pl.Float64)
        )
        )

        # # Save hourly data
        # hourly_ozone_data = (ozone_data
        #                      .filter(
        #     pl.col("sample_duration") == "1 HOUR")
        #                      .with_columns(
        #     pl.col("sample_measurement")
        #     .is_not_null()
        #     .cast(pl.Int64)
        #     .sum()
        #     .over(["monitor_id", "date_local"])
        #     .alias("num_hrly_obs"))
        #                      .filter(
        #     pl.col("num_hrly_obs") >= 18)
        #                      .with_columns(
        #     pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
        #                      .with_columns(
        #     pl.col("date_local").dt.date().alias("date"),
        #     pl.col("24_hour_local").str.split(":").list.get(0).cast(pl.Int64).alias("hour"))
        #                      .rename({"sample_measurement": "ozone_hrly"})
        #                      .select("latitude", "longitude", "datum", "state_code", "county_code",
        #                              "monitor_id", "site_num", "poc", "ozone_hrly", "date", "hour")
        #                      )
        #
        # hourly_ozone_data.write_csv(self.output_data_path/"chicago_ozone_2000_2012_hourly.csv")

        # Save daily data
        daily_ozone_data = (ozone_data
                            .filter(
            (~pl.all_horizontal(pl.all().is_null())) |
            (pl.col("sample_duration") == "8-HR RUN AVG BEGIN HOUR"))
                            .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local"])
            .alias("num_hrly_obs"))
                            .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("num_hrly_obs_ozone"),
            pl.col("sample_measurement")
            .max()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("max_ozone"),
            pl.col("sample_measurement")
            .mean()
            .over(["monitor_id", "date_local", "sample_duration"])
            .alias("avg_ozone"))
                            .with_columns(
            pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
                            .with_columns(
            pl.col("date_local").dt.date().alias("date"))
                            .sort("monitor_id", "date")
                            .unique(["monitor_id", "date"])
                            .filter(
            pl.col("num_hrly_obs") >= 18)  # original code uses num_hrly_obs, but they are equivalent
                            .sort("monitor_id", "date")
                            .drop(cs.contains("gmt"), pl.col("num_hrly_obs"))
                            )

        daily_ozone_data.write_csv(self.output_data_path/"chicago_ozone_2000_2012_daily.csv")

    def process_pollution_data(self):
        self._extract_chicago_aqi()
        self._extract_chicago_co()
        self._extract_chicago_pm10()
        self._extract_chicago_no2()
        self._extract_chicago_ozone()

    def merge_pollution(self):
        # AQI ---------------------------------------------------------------------------------------------------------
        aqi_data = (pl.read_csv(self.output_data_path/"chicago_aqi_2000_2015.csv")
                    .with_columns(
            pl.col("aqi").cast(pl.Float64))
                    )
        aqi_data  = (aqi_data
                     .with_columns(
            pl.col("datelocal").str.to_datetime(format="%Y-%m-%d"))
                     .with_columns(
            pl.col("datelocal").dt.date().alias("date"))
                     .filter(
            pl.col("date").dt.year().is_between(2000, 2012, closed="both"))
                     .with_columns(
            (pl.col("countycode").cast(pl.String) + "_" + pl.col("sitenum").cast(pl.String)
             + "_" + pl.col("poc").cast(pl.String)).alias("monitor_id"))
                     .filter(
            pl.col("aqi").is_not_null())
                     .with_columns(
            pl.when(pl.col("parametername").str.contains("PM10"))
            .then(pl.lit("PM10"))
            .when(pl.col("parametername").str.contains("Carbon monoxide"))
            .then(pl.lit("CO"))
            .when(pl.col("parametername").str.contains("Ozone"))
            .then(pl.lit("Ozone"))
            .when(pl.col("parametername").str.contains("Nitrogen dioxide"))
            .then(pl.lit("NO2"))
            .otherwise(None)
            .alias("pollutant_name"))
                      .filter(
            pl.col("pollutant_name").is_not_null())
                     )

        temp_aqi = (aqi_data
                    .filter(
            pl.col("cityname").is_in(["Chicago"]))
                    .group_by("date", "pollutant_name")
                    .agg(
            pl.col("aqi").mean().alias("aqi_mean_chicago"))
                    .select("date", "aqi_mean_chicago", "pollutant_name")
                    .sort("pollutant_name", "date")
                    )

        aqi_data_by_date_pollutant = (aqi_data
                                      .with_columns(
            pl.when(
                (pl.col("pollutant_name") == "Ozone") & (pl.col("monitor_id").is_in(["31_64_1", "31_7002_1"])))
            .then(pl.lit(1))
            .when(
                (pl.col("pollutant_name") == "CO") & (
                    pl.col("monitor_id").is_in(["31_3103_1", "31_4002_1", "31_6004_1", "31_63_1"])))
            .then(pl.lit(1))
            .when(
                (pl.col("pollutant_name") == "NO2") & (
                    pl.col("monitor_id").is_in(["31_3103_1", "31_4002_1", "31_63_1"])))
            .then(pl.lit(1))
            .when(
                (pl.col("pollutant_name") == "PM10") & (
                    pl.col("monitor_id").is_in(["31_1016_3", "31_22_3"])))
            .then(pl.lit(1))
            .otherwise(0)
            .alias("keeplist"))
                                      .filter(pl.col("keeplist") == 1)
                                      .group_by("date", "pollutant_name")
                                      .agg(
            pl.col("aqi").mean().alias("aqi_mean_sample"))
                                      .select("date", "aqi_mean_sample", "pollutant_name")
                                      .sort("pollutant_name", "date")
                                      )

        aqi_out = (aqi_data_by_date_pollutant
                   .join(
            temp_aqi, on=["date", "pollutant_name"], how="left", validate="1:1")
                   .group_by("date")
                   .agg(
            pl.col("aqi_mean_sample").max().alias("max_aqi_sample"),
            pl.col("pollutant_name").filter(
                pl.col("aqi_mean_sample") == pl.col("aqi_mean_sample").max()
            ).first().alias("max_aqi_sample_poll"),
            pl.col("aqi_mean_chicago").max().alias("max_aqi_chicago"),
            pl.col("pollutant_name").filter(
                pl.col("aqi_mean_chicago") == pl.col("aqi_mean_chicago").max()
            ).first().alias("max_aqi_chicago_poll"))
                   .sort("date")
                   )

        # OZONE ---------------------------------------------------------------------------------------------------------
        ozone_data = (pl.read_csv(self.output_data_path / "chicago_ozone_2000_2012_daily.csv")
                      .with_columns(
            pl.col("date").str.to_datetime(format="%Y-%m-%d").dt.date())
                      .with_columns(
            pl.col("date").dt.year().alias("year")
        )
                      )

        ozone_data = (ozone_data
                      .filter(
            pl.col("monitor_id").is_in(["31_1003_2", "31_1601_1", "31_1_1", "31_32_1", "31_4002_1",
                                        "31_4007_1", "31_4201_1", "31_64_1", "31_7002_1", "31_72_1", "31_76_1"]))
                      .select("avg_ozone", "max_ozone", "monitor_id", "date")
                      .sort("monitor_id", "date")
                      )

        ozone_out = (ozone_data
                     .pivot(
            on="monitor_id", values=cs.contains("ozone"))
                     .with_columns(
            pl.mean_horizontal(pl.col("^avg.*31_(64_1|7002_1)$")).alias("avg_ozone_mean"),
            pl.mean_horizontal(pl.col("^max.*31_(64_1|7002_1)$")).alias("max_ozone_mean"))
                     .with_columns(
            ((pl.col("avg_ozone_31_64_1").is_not_null().cast(pl.Int64) +
              pl.col("avg_ozone_31_7002_1").is_not_null().cast(pl.Int64)) / 2
             ).alias("monitor_pct_ozone"))
                     .sort("date")
                     .select("avg_ozone_mean", "max_ozone_mean", "date", "monitor_pct_ozone")
                     )

        # CO ---------------------------------------------------------------------------------------------------------
        co_data = (pl.read_csv(self.output_data_path / "chicago_co_2000_2012_daily.csv")
                   .with_columns(
            pl.col("date").str.to_datetime(format="%Y-%m-%d").dt.date())
                   .filter(
            pl.col("monitor_id").is_in(["31_3103_1","31_4002_1","31_6004_1","31_63_1"]))
                   )

        co_wide = (co_data
                   .select("avg_co", "max_co", "monitor_id", "date")
                   .pivot(
            on="monitor_id", values=cs.contains("co"))
                   )
        avg_cols = co_wide.select(pl.col("^avg.*$")).columns
        co_temp = (co_wide
                   .with_columns(
            (pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.Int64) for col in avg_cols]) / len(avg_cols))
            .alias("monitor_pct_co"))
                   .with_columns(
            pl.mean_horizontal(pl.col("^avg.*$")).alias("avg_co_mean"),
            pl.mean_horizontal(pl.col("^max.*$")).alias("max_co_mean"))
                   .drop("avg_co_31_6004_1")  # TODO: author did not drop max_co_31_6004_1
                   )

        avg_cols_drop_290 = co_temp.select(pl.col("^avg.*$").exclude("avg_co_mean")).columns
        co_out = (co_temp
                  .with_columns(
            (pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.Int64) for col in avg_cols_drop_290]) / len(avg_cols_drop_290))
            .alias("monitor_pct_co_drop_290"))
                  .with_columns(
            pl.mean_horizontal(pl.col("^avg.*$")).alias("avg_co_mean_drop_290"),
            pl.mean_horizontal(pl.col("^max.*$")).alias("max_co_mean_drop_290"))
                  .select(pl.col("^avg_co_mean.*$"), pl.col("^max_co_mean.*$"), "date", pl.col("^monitor_pct_co.*$"))
                  .sort("date")
                  )

        # NO2 --------------------------------------------------------------------------------------------------------
        no_data = (pl.read_csv(self.output_data_path / "chicago_no2_2000_2012_daily.csv")
                   .with_columns(
            pl.col("date").str.to_datetime(format="%Y-%m-%d").dt.date())
                   .filter(
            pl.col("monitor_id").is_in(["31_3103_1","31_4002_1","31_63_1"]))
                   )

        no_wide = (no_data
                   .select("avg_no2", "max_no2", "monitor_id", "date")
                   .pivot(
            on="monitor_id", values=cs.contains("no"))
                   )

        avg_cols = no_wide.select(pl.col("^avg.*$")).columns
        no_out = (no_wide
                  .with_columns(
            (pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.Int64) for col in avg_cols]) / len(avg_cols))
            .alias("monitor_pct_no2"))
                  .with_columns(
            pl.mean_horizontal(pl.col("^avg.*$")).alias("avg_no2_mean"),
            pl.mean_horizontal(pl.col("^max.*$")).alias("max_no2_mean"))
                  .select("avg_no2_mean", "max_no2_mean", "date", "monitor_pct_no2")
                  .sort("date")
                   )

        # PM10 --------------------------------------------------------------------------------------------------------
        pm_data = (pl.read_csv(self.output_data_path / "chicago_pm10_2000_2012_daily.csv")
                   .with_columns(
            pl.col("date").str.to_datetime(format="%Y-%m-%d").dt.date())
                   .filter(
            pl.col("monitor_id").is_in(["31_1016_3","31_22_3"]))
                   )

        pm_wide = (pm_data
                   .rename(
            {"max24hr_pm10_derived": "max_pm10",
             "avg24hr_pm10_derived": "avg_pm10",})
                   .select("avg_pm10", "max_pm10", "monitor_id", "date")
                   .pivot(
            on="monitor_id", values=cs.contains("pm10"))
                   )

        avg_cols = pm_wide.select(pl.col("^avg.*$")).columns
        pm_out = (pm_wide
                  .with_columns(
            (pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.Int64) for col in avg_cols]) / len(avg_cols))
            .alias("monitor_pct_pm10"))
                  .with_columns(
            pl.mean_horizontal(pl.col("^avg.*$")).alias("avg_pm10_mean"),
            pl.mean_horizontal(pl.col("^max.*$")).alias("max_pm10_mean"),)
                  .select("avg_pm10_mean", "max_pm10_mean", "date", "monitor_pct_pm10")
                  .sort("date")
                  )

        # Merge -------------------------------------------------------------------------------------------------------
        poll_data = (pm_out
                     .join(
            co_out, on=["date"], how="left", validate="1:1")
                     .join(
            ozone_out, on=["date"], how="left", validate="1:1")
                     .join(
            no_out, on=["date"], how="left", validate="1:1")
                     .join(
            aqi_out, on=["date"], how="left", validate="1:1")
                     )

        poll_data.write_csv(self.output_data_path / "chicago_pollution_2000_2012.csv")

    def _extract_midwayohare_daily_weather(self):
        ghcn_data = pl.read_csv(self.input_data_path / "chicago_midwayohare_ghcn_daily_1991_2012.csv")

        ghcn_data = (ghcn_data
                     .with_columns(
            pl.col("strdate").cast(pl.String).str.to_datetime(format="%Y%m%d").dt.date().alias("date"))
                     .with_columns(
            pl.when(pl.col("qflag").is_not_null())  # != "" (empty string) in the original dataset
            .then(None)
            .otherwise(pl.col("value"))
            .alias("value"))
                     .drop("obstime", "mflag", "sflag", "qflag")
                     )

        ghcn_wide = (ghcn_data
                     .sort("station_id", "date")
                     .pivot(
            on="element", values="value")
                     .with_columns(
            [(pl.col(col) / 10).alias(col) for col in ["PRCP", "TMAX", "TMIN"]])
                     .with_columns(
            cs.numeric().exclude("station_id", "strdate").round(1))
                     .with_columns(
            pl.when(pl.col("station_id") == "USW00094846")
            .then(pl.lit("OHARE"))
            .when(pl.col("station_id") == "USW00014819")
            .then(pl.lit("MIDWAY"))
            .otherwise(pl.lit(""))
            .alias("weather_airport"))
                     .drop("station_id")
                     )

        ghcn_temp = (ghcn_wide
                     .drop("strdate")
                     .pivot(
            on="weather_airport", values=cs.numeric())
                     .with_columns(
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.day().alias("day"))
                     )

        ghcn_doy_mean = (ghcn_temp
                         .filter(
            pl.col("date").dt.year() < 2001)
                         .group_by("month", "day")
                         .agg(
            [pl.col(f"{col}_MIDWAY").mean().alias(f"mean_{col}_1991_2000") for col in ["TMAX", "TMIN", "PRCP"]])
                         )

        ghcn_out = (ghcn_temp
                    .join(
            ghcn_doy_mean, on=["month", "day"], how="left", validate="m:1")
                    .drop("day", "month",)
                    .filter(pl.col("date").dt.year() < 2001))

        ghcn_out.write_csv(self.output_data_path / "chicago_midwayohare_daily_weather.csv")

    def _extract_chicago_hourly_weather(self):
        hourly_weather_data = pd.read_stata(self.input_data_path/"chicago_hourly_weather_stations.dta")
        hourly_weather_data = pl.from_pandas(hourly_weather_data)
        hourly_weather_data.write_csv(self.output_data_path/"chicago_hourly_weather_stations.csv")

    def _generate_weather_variables(self):
        weather_data = (pl.scan_csv(self.output_data_path / "chicago_hourly_weather_stations.csv",
                                    schema_overrides={col: pl.String for col in ["wind_speed_qual", "wind_angle_qual"]})
                        .select("usaf", "wban", "month", "day", "year",  "hour", "min", "latitude", "longitude",
                                "wind_angle", "wind_angle_qual", "wind_obs_type", "wind_speed", "wind_speed_qual",
                                "temp", "temp_qual", "dewpoint", "dewpoint_qual", "sealevel_pressure",
                                "sealevel_pressure_qual", "stationname")
                        .with_columns(
            (pl.col("year").cast(pl.String) + "-" + pl.col("month").cast(pl.String)
             + "-" + pl.col("day").cast(pl.String)).str.to_date(format="%Y-%m-%d").alias("date"))
                        .collect())

        # Wind --------------------------------------------------------------------------------------------------------
        weather_data = (weather_data
                        .with_columns(
            pl.when(
                (pl.col("wind_speed") == 9999) | ((pl.col("wind_angle") == 999) & (pl.col("wind_speed") != 0)))
            .then(None)
            .otherwise(pl.col("wind_speed"))
            .alias("wind_speed"))
                        .with_columns(
            pl.when(
                (pl.col("wind_speed").is_null()) | ((pl.col("wind_angle") == 999) & pl.col("wind_speed") != 0))
            .then(None)
            .when(pl.col("wind_speed") == 0)
            .then(pl.lit(0))
            .otherwise(pl.col("wind_angle"))
            .alias("wind_angle"))
                        .filter(
            pl.col("wind_speed_qual").is_in(["1", "5", "9"]) & pl.col("wind_angle_qual").is_in(["1", "5", "9"]))
                        .with_columns(
            # Convert degrees to radians
            pl.col("wind_angle").radians().alias("wind_angle_radians"))
                        .with_columns(
            # X Y wind vector components
            pl.col("wind_angle_radians").cos().alias("xwind"),
            pl.col("wind_angle_radians").sin().alias("ywind"),)
                        .with_columns(
            # Weight by speed
            (pl.col("wind_speed") * pl.col("xwind")).alias("xwind_speed"),
            (pl.col("wind_speed") * pl.col("ywind")).alias("ywind_speed"))
                        .with_columns(
            # Weighted proportional to power
            (pl.col("wind_speed").pow(3) * pl.col("xwind")).alias("xwind_power"),
            (pl.col("wind_speed").pow(3) * pl.col("ywind")).alias("ywind_power"))
                        .with_columns(
            (pl.col("wind_speed").is_not_null() & pl.col("wind_angle").is_not_null()).cast(pl.Int64).alias("windobsind"))
                        )

        wind_daily_data = (weather_data
                           .group_by("usaf", "wban", "date")
                           .agg(
            pl.col("xwind").mean().alias("xwind_avg"),
            pl.col("xwind_speed").mean().alias("xwind_speed_avg"),
            pl.col("xwind_power").mean().alias("xwind_power_avg"),
            pl.col("ywind").mean().alias("ywind_avg"),
            pl.col("ywind_speed").mean().alias("ywind_speed_avg"),
            pl.col("ywind_power").mean().alias("ywind_power_avg"),
            pl.col("wind_speed").mean().alias("avg_wind_speed"),
            pl.col("windobsind").sum().alias("windobs"))
                           .with_columns(
            # Generate norms
            (pl.col("xwind_speed_avg").pow(2) + pl.col("ywind_speed_avg").pow(2)).sqrt().alias("speed_norm"),
            (
                (pl.col("xwind_power_avg").pow(2) + pl.col("ywind_power_avg").pow(2)).sqrt()
            ).pow(1/3).alias("power_norm"))
                           .with_columns(
            # Rescale power norm
            (pl.col("power_norm") / 1000).alias("power_norm"))
                           .with_columns(
            # Generate average angles
            pl.when(pl.col("ywind_avg") < 0)
            .then(pl.arctan2("ywind_avg", "xwind_avg") + 2 * np.pi)
            .when(pl.col("ywind_avg") >= 0)
            .then(pl.arctan2("ywind_avg", "xwind_avg"))
            .otherwise(None)
            .alias("wind_dir_avg"),
            pl.when(pl.col("ywind_speed_avg") < 0)
            .then(pl.arctan2("ywind_speed_avg", "xwind_speed_avg") + 2 * np.pi)
            .when(pl.col("ywind_speed_avg") >= 0)
            .then(pl.arctan2("ywind_speed_avg", "xwind_speed_avg"))
            .otherwise(None)
            .alias("wind_speed_dir_avg"),
            pl.when(pl.col("ywind_power_avg") < 0)
            .then(pl.arctan2("ywind_power_avg", "xwind_power_avg") + 2 * np.pi)
            .when(pl.col("ywind_power_avg") >= 0)
            .then(pl.arctan2("ywind_power_avg", "xwind_power_avg"))
            .otherwise(None)
            .alias("wind_power_dir_avg"),

            # Generate calm dummy
            (pl.col("speed_norm") == 0).alias("calmday"),

            # Generate wind power
            pl.col("avg_wind_speed").pow(3).alias("wind_power"))
                           )
        # Temperature -------------------------------------------------------------------------------------------------
        weather_data = (weather_data
                        .with_columns(
            pl.when(pl.col("temp_qual").is_in(["6", "7", "3", "2"]) | (pl.col("temp") == 9999))
            .then(None)
            .otherwise(pl.col("temp"))
            .alias("temp_new"))
                        )

        temp_daily_data = (weather_data
                           .group_by("usaf", "wban", "date")
                           .agg(
            pl.col("temp_new").max().alias("tmax"),
            pl.col("temp_new").mean().alias("tavg"),
            pl.col("temp_new").min().alias("tmin"),
            pl.col("temp_new").is_not_null().cast(pl.Int64).sum().alias("totobs"))
                           .with_columns(
            (pl.col("totobs") < 18).alias("tempdataflag"))
                           )

        # Dew point -------------------------------------------------------------------------------------------------
        weather_data = (weather_data
                        .with_columns(
            pl.when(pl.col("dewpoint_qual").is_in(["6", "7", "3", "2"]) | (pl.col("dewpoint") == 9999))
            .then(None)
            .otherwise(pl.col("dewpoint"))
            .alias("dewpoint_new")
        ))

        dew_daily_data = (weather_data
                          .group_by("usaf", "wban", "date")
                          .agg(
            pl.col("dewpoint_new").mean().alias("dew_point_avg"),)
                          )

        # Sea-level pressure ------------------------------------------------------------------------------------------
        weather_data = (weather_data
                        .with_columns(
            pl.when(pl.col("sealevel_pressure_qual").cast(pl.String).is_in(["6", "7", "3", "2"]) |
                    (pl.col("sealevel_pressure") == 9999))
            .then(None)
            .otherwise(pl.col("sealevel_pressure"))
            .alias("sealevel_pressure_new"))
                        )

        sealevel_daily_data = (weather_data
                               .group_by("usaf", "wban", "date")
                               .agg(
            pl.col("sealevel_pressure_new").mean().alias("sealevel_pressure_avg"),)
                               )

        # Merge all daily data
        weather_daily_data = (wind_daily_data
                              .join(
            temp_daily_data, on=["usaf", "wban", "date"], how="full", validate="1:1")
                              .drop(cs.ends_with("right"))
                              .join(
            dew_daily_data, on=["usaf", "wban", "date"], how="full", validate="1:1")
                              .drop(cs.ends_with("right"))
                              .join(
            sealevel_daily_data, on=["usaf", "wban", "date"], how="full", validate="1:1")
                              .select("usaf", "wban", "date", "wind_dir_avg", "wind_speed_dir_avg",
                                      "wind_power_dir_avg", "avg_wind_speed", "windobs", "speed_norm",
                                      "power_norm", "calmday", "tempdataflag", "tmax", "tavg", "tmin",
                                      "dew_point_avg", "sealevel_pressure_avg")
                              .sort("usaf", "wban", "date")
                              )

        # weather_daily_data.filter(pl.all_horizontal(pl.col("usaf", "date").is_duplicated())
        #                           ).sort("usaf", "date")

        weather_daily_data.write_csv(self.output_data_path / "chicago_weather_daily_from_hourly.csv")





if __name__ == '__main__':
    source_path = Path("replication_package")
    input_data_path = source_path / "Raw-Data"
    output_data_path = Path("data")
    preprocessor = DataPreprocessor(input_data_path, output_data_path)
    preprocessor._generate_weather_variables()

























