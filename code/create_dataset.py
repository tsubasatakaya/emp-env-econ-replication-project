import os
from pathlib import Path

from preprocessing.preprocess import DataPreprocessor

source_path = Path("replication_package")
input_data_path = source_path/"Raw-Data"
output_data_path = Path("data")

preprocessor = DataPreprocessor(input_data_path,
                                output_data_path,)

preprocessor._extract_crime_data()