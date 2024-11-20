import os
from pathlib import Path

import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs


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

        # Save hourly data
        hourly_co_data = (co_data
                          .filter(
            pl.col("sample_duration") == "1 HOUR")
                          .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local"])
            .alias("num_hrly_obs"))
                          .filter(
            pl.col("num_hrly_obs") >= 18)
                          .with_columns(
            pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
                          .with_columns(
            pl.col("date_local").dt.date().alias("date"),
            pl.col("24_hour_local").str.split(":").list.get(0).cast(pl.Int64).alias("hour"))
                          .rename({"sample_measurement": "co_hrly"})
                          .select("latitude", "longitude", "datum", "state_code", "county_code",
                                  "monitor_id", "site_num", "poc", "co_hrly", "date", "hour")
                          )

        hourly_co_data.write_csv(self.output_data_path/"chicago_co_2000_2012_hourly.csv")

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

        # Save hourly data
        hourly_pm_data = (pm_data
                          .filter(
            pl.col("sample_duration") == "1 HOUR")
                          .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local"])
            .alias("num_hrly_obs"))
                          .filter(
            pl.col("num_hrly_obs") >= 18)
                          .with_columns(
            pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
                          .with_columns(
            pl.col("date_local").dt.date().alias("date"),
            pl.col("24_hour_local").str.split(":").list.get(0).cast(pl.Int64).alias("hour"))
                          .rename({"sample_measurement": "pm10hrly"})
                          .select("latitude", "longitude", "datum", "state_code", "county_code",
                                  "monitor_id", "site_num", "poc", "pm10hrly", "date", "hour")
                          )

        hourly_pm_data.write_csv(self.output_data_path/"chicago_pm10_2000_2012_hourly.csv")

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

        # Save hourly data
        hourly_no_data = (no_data
                          .filter(
            pl.col("sample_duration") == "1 HOUR")
                          .with_columns(
            pl.col("sample_measurement")
            .is_not_null()
            .cast(pl.Int64)
            .sum()
            .over(["monitor_id", "date_local"])
            .alias("num_hrly_obs"))
                          .filter(
            pl.col("num_hrly_obs") >= 18)
                          .with_columns(
            pl.col("date_local").str.to_datetime(format="%Y-%m-%d"))
                          .with_columns(
            pl.col("date_local").dt.date().alias("date"),
            pl.col("24_hour_local").str.split(":").list.get(0).cast(pl.Int64).alias("hour"))
                          .rename({"sample_measurement": "no2_hrly"})
                          .with_columns(
            (pl.col("no2_hrly") / 1000).alias("no2_hrly"))
                          .select("latitude", "longitude", "datum", "state_code", "county_code",
                                  "monitor_id", "site_num", "poc", "no2_hrly", "date", "hour")
                          )

        hourly_no_data.write_csv(self.output_data_path/"chicago_no2_2000_2012_hourly.csv")

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







if __name__ == '__main__':
    source_path = Path("replication_package")
    input_data_path = source_path / "Raw-Data"
    output_data_path = Path("data")
    preprocessor = DataPreprocessor(input_data_path, output_data_path)
    preprocessor._extract_chicago_no2()

























