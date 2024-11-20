import os
from pathlib import Path

from preprocessing.prepare_data import *

source_path = Path("replication_package")
input_data_path = source_path/"Raw-Data"
output_data_path = Path("data")

extract_crime_data(input_data_path, output_data_path)