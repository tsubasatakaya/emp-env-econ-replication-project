import os
from pathlib import Path

from preprocessing.preprocess import DataPreprocessor

source_path = Path("replication_package")
input_data_path = source_path/"Raw-Data"
output_data_path = Path("data")

preprocessor = DataPreprocessor(input_data_path,
                                output_data_path,)

preprocessor.create_citylevel_dataset(process_raw_data=True)
preprocessor.save_original_micro_dataset()
# preprocessor.create_micro_dataset(wind_dir_threshold=60)