# Empirical Environmental Economics: Paper replication

Hertie School  
Course: GRAD-E1460: Empirical Environmental Economics   
Instructor: Johanna Arlinghaus  

Chosen paper: Herrnstadt, E., Heyes, A., Muehlegger, E., & Saberian, S. (2021). Air Pollution and Criminal Activity: Microgeographic Evidence from Chicago. American Economic Journal: Applied Economics, 13(4), 70–100. https://doi.org/10.1257/app.20190091

## Description
This repository contains all the code to generate the data set, tables, and figures used in my submission of the paper replication project. The repo does **not** include the original replication package created by the authors. You can download it [here](https://www.openicpsr.org/openicpsr/project/119403/version/V1/view)

## How to (re-)reproduce 
Please locate the original replication package folder in the root directory. And then run the following scripts in order:  
1. `create_dateset.py`  
2. `main.R`

`create_dataset.py` automatically creates a `data` folder in the root directory (if it does not exist) and saves a set of .csv files created from the raw data in the replication package (not included in this repo due to the large data size). The main .csv files used for the analysis are `chicago_citylevel_dataset.csv` and `micro_dataset_original.csv`.  

`main.R` runs numbered .R scripts sequentially. `00-setup.R` first installs and loads required packages and creates an `output` folder (if it does not exist). `01-summary_statistics.R`, `02-cityregs.R`, and `03-microregs.R` generate replicated tables. `04-microreg_extension.R` executes the extension analysis.
