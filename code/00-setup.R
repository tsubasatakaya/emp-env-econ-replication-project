packages <-  c(
  "tidyverse", 
  "ggplot2", 
  "gridExtra",
  "see",
  "forcats",
  "broom", 
  "fixest", 
  "fastDummies",
  "grf",
  "modelsummary", 
  "gt")

package.check <- lapply(
  packages,
  FUN <-  function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

data_path <- "data"
output_path <- "output"

if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = TRUE)
}

for (folder in c("figures", "tables")) {
  folder_path <- file.path(output_path, folder)
  if (!dir.exists(folder_path)) {
    dir.create(folder_path, recursive = TRUE)
  }
}
