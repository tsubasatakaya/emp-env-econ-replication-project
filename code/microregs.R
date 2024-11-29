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
  violent_results,
  property_results
)
panel_a_title <- "Panel A: Violent crime"
panel_b_title <- "Panel B: Property crime"

cm <- c("treatment" = "Treatment (downwind)")
gof_f <- function(x) format(round(x, 3), big.mark = ",")
gm <- list(
  list("raw" = "nobs", "clean" = "Observations", "fmt" = gof_f),
  list("raw" = "r.squared", "clean" = "R\U00B2", "fmt" = gof_f)
)

add_rows <- tibble(term = c("Route \U00D7 side fixed effects", 
                            "Route \U00D7 date fixed effects",
                            "Route \U00D7 side weather interaction",
                            "Route \U00D7 side fixed effects", 
                            "Route \U00D7 date fixed effects",
                            "Route \U00D7 side weather interaction"),
                   col1 = c(NA, NA, NA, NA, NA, NA),
                   col2 = c("X", NA, NA, "X", NA, NA),
                   col3 = c("X", "X", NA, "X", "X", NA),
                   col4 = c("X", "X", "X", "X", "X", "X"))
attr(add_rows, "position") <- c(3:5, 10:12)

footnote <- md("*Notes*: Robust standard errors reported. Dependent variable
               is the number of crimes within one mile of one side of the interstate
               normalized by the mean number of crimes. See Table 4 of the paper
               for more details.")

summary_data <- msummary(panels, fmt = 4, shape = "rbind",
                         coef_map = cm, gof_map = gm,
                         add_rows = add_rows,)
gt(summary_data@data) |> 
  tab_row_group(rows = 1:7, label = panel_a_title) |> 
  tab_row_group(rows = 8:14, label = panel_b_title) |> 
  row_group_order(groups = c(panel_a_title, panel_b_title)) |> 
  tab_footnote(footnote = footnote) |> 
  cols_align(align = "center",
             columns = 2:5) |> 
  gtsave("table_4_rep.tex", path = file.path(output_path, "tables"))


# Make effect plot -------------------------------------------------------------
specs <- c("No controls", "+ Route \U00D7 side fixed effects",
           "+ Route \U00D7 date fixed effects", "+ Route \U00D7 side weather interaction")

coef_df <- tibble()
for (i in seq_along(violent_results)) {
  df <- tidy(violent_results[[i]], conf.int = TRUE) |> 
    filter(str_detect(term, treatment)) |> 
    mutate(dv = "violent",
           spec = specs[i])
  coef_df <- coef_df |> 
    bind_rows(df)
}
for (i in seq_along(property_results)) {
  df <- tidy(property_results[[i]], conf.int = TRUE) |> 
    filter(str_detect(term, treatment)) |> 
    mutate(dv = "property",
           spec = specs[i])
  coef_df <- coef_df |> 
    bind_rows(df)
}

coef_df <- coef_df |> 
  mutate(spec = factor(spec, levels = specs),
         dv = factor(dv, levels = c("violent", "property")))

microreg_plot <- ggplot(coef_df, aes(x = dv, ymin = conf.low, ymax = conf.high)) +
  geom_hline(yintercept = 0, linetype = "dashed") + 
  geom_linerange(aes(col = spec), linewidth = 1, 
                 position = position_dodge(width = 0.15)) +
  geom_point(aes(x = dv, y = estimate, col = spec), size = 3,
             position = position_dodge(width = 0.15)) + 
  scale_x_discrete(labels = c("Violent crimes", "Property crimes")) + 
  scale_color_okabeito(palette = "black_first") +
  theme_minimal() +
  labs(x = "", y = "Treatment effect\n") + 
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        panel.border = element_rect(color = "grey", fill = NA),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.title = element_text(size = 12,),
        axis.text = element_text(size = 11),
        axis.text.x = element_text(face = "bold"),
        legend.text = element_text(size = 10))
ggsave(file.path(output_path, "figures/microreg_coef_plot.svg"), microreg_plot,
       width = 12, height = 7.8, units = "in", dpi = 300
)



















