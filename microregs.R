micro_data <- read_csv(file.path(data_path, "micro_dataset_original.csv")) |> 
  arrange(date)
