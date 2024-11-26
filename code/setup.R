packages <-  c("tidyverse", "fixest", "modelsummary", "gt")

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