micro_data <- read_csv(file.path(data_path, "micro_dataset_original.csv")) |> 
  arrange(date)

micro_data <- micro_data |> 
  mutate(across(c(routeside, routedate), as.factor))

cov_list <- list(
  NA,
  c("routeside"),
  c("routeside"),
  c("routeside * tmax", "routeside * valuePRCP_MIDWAY")
)
fe <- c(FALSE, FALSE, TRUE, TRUE)
treatment <- "treatment"


violent_data <- micro_data |> filter(violent == 1)
property_data <- micro_data |> filter(violent == 0)

fit_ols <- function(df, dv, treatment, cov, fe = FALSE) {
  formula_str <- paste0(dv, "~", treatment)
  if (!is.na(cov[[1]])) {
    formula_str <- paste0(formula_str, "+",
                          paste0(cov, collapse = "+")
                          )
  }
  if (fe) {
    formula_str <- paste0(formula_str, "| routedate")
  }
  print(formula_str)
  formula <- as.formula(formula_str)
  res <- feols(formula, data = df, vcov = "HC1")
}

# violent ----------------------------------------------------------------------
names <- c("(1)", "(2)", "(3)", "(4)")

violent_results <- list()
for (i in seq_along(names)) {
  res <- fit_ols(violent_data, "stand_crimes", "treatment", cov_list[[i]], fe[i])
  violent_results[[i]] <- res
}

# property ---------------------------------------------------------------------
property_results <- list()
for (i in seq_along(names)) {
  res <- fit_ols(property_data, "stand_crimes", "treatment", cov_list[[i]], fe[i])
  property_results[[i]] <- res
}


# Generate table ---------------------------------------------------------------
panels <- list(
  "Panel A: Violent crime" = violent_results,
  "Panel B: Property crime" = property_results
)

cm <- c("treatment" = "Treatment (downwind)")
gof_f <- function(x) format(round(x, 3), big.mark = ",")
gm <- list(
  list("raw" = "nobs", "clean" = "Observations", "fmt" = gof_f),
  list("raw" = "r.squared", "clean" = "R\U00B2", "fmt" = gof_f)
)

f_rows <- tibble(term = c("First stage F-statistic", "Calendar fixed effects",
                          "Weather controls", "Historical mean temp"),
                 col1 = c(NA, "X", NA, NA),
                 col2 = c(NA, "X", "X", "X"),
                 col3 = c(as.character(f_violent), "X", "X", "X"),
                 col4 = c(NA, "X", NA, NA),
                 col5 = c(NA, "X", "X", "X"),
                 col6 = c(as.character(f_property), "X", "X", "X"))
attr(f_rows, "position") <- c(3, 4, 5, 6)
msummary(panels, fmt = 4, shape = "rbind",
         coef_map = cm, gof_map = gm)


















