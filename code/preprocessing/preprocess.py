import os
from pathlib import Path

import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs


data_path = Path("data")
source_path = Path("replication_package")
input_data_path = source_path/"Raw-Data"

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


    def extract_crime_interstate_distance(self):
        pass

































