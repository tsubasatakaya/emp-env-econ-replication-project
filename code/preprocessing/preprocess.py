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
        ).write_csv(self.output_data_path / "chicago_part1_crimes.csv")

        # Save all crimes
        crime_data.drop(
            ["block", "description", "location_description", "beat", "district",
             "ward", "community_area", "x_coordinate", "y_coordinate", "location"]
        ).write_csv(self.output_data_path / "chicago_all_crimes.csv")


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








if __name__ == '__main__':
    source_path = Path("replication_package")
    input_data_path = source_path / "Raw-Data"
    output_data_path = Path("data")
    preprocessor = DataPreprocessor(input_data_path, output_data_path)
    preprocessor._extract_crime_interstate_distance()



























