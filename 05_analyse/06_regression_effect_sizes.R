### Bayesian multilevel regressions, effect sizes (ROPE) and rank correlations (Table 4, Figures 4 + 6)

# Load packages
library(tidyverse)
library(openxlsx)
library(reshape2)
library(psych)
library(data.table)
library(car)
library(brms)
library(bayestestR)   # ROPE, p_direction, p_significance
# effectsize not needed: standardized betas are read directly from posterior_summary()
library(corrplot)     # rank correlation heatmap

df <- read.csv("evaluation_results/dataset_regression_rebuilt.csv")
row.names(df) <- df$X
df <- df[, -which(names(df) == "X")]

df <- df %>% filter(group_number >= 2, group_number <= 33)
stopifnot(nrow(df) == 320, length(unique(df$group_number)) == 32)

####### REGRESSIONS
# Define the list of dependent variable column names
dependent_vars_all <- c("autnes_sentiment", "ffp", "topics", "bestmatch", "mostsimilar", "opposite", "wordintrusion")
dependent_vars_overview <- c("overall_score_v1", "overall_score_v2", "sum")

# ------------------------------------------------------------------------------
# Multicollinearity check and pre-processing
# z-score all three hyperparameter predictors before fitting. This:
#   (a) eliminates the ROPE multicollinearity warning (standardized predictors
#       have bounded correlations that the sampler handles better), and
#   (b) makes coefficients directly comparable across predictors and tasks
#       without needing a separate standardize_parameters() refit in Step 2.
# We keep the original df intact and work on df_scaled throughout.
# ------------------------------------------------------------------------------
cat("=== Pre-fit predictor correlations (Pearson) ===\n")
print(round(cor(df[, c("lower", "mincount", "windows")], use = "complete.obs"), 3))

# Check VIF on a simple lm as a quick diagnostic (VIF > 5 = concern, > 10 = serious)
vif_check <- lm(df[[dependent_vars_all[1]]] ~ lower + mincount + windows, data = df)
cat("\nVIF for predictors (using '", dependent_vars_all[1], "' as reference outcome):\n")
print(vif(vif_check))
cat("\nNote: VIF > 5 indicates meaningful multicollinearity.\n",
    "Predictors are z-scored below to mitigate this for ROPE and standardized-β steps.\n\n")

df_scaled <- df %>%
  mutate(
    lower   = as.numeric(scale(lower)),
    mincount = as.numeric(scale(mincount)),
    windows = as.numeric(scale(windows))
  )

# Empty list to store the model fits
fits_list <- list()

# Loop through each dependent variable — fitted on z-scored predictors
for (dep_var in dependent_vars_all) {
  print(dep_var)
  fit <- brm(
    formula    = as.formula(paste(dep_var, "~ lower + mincount + windows + (1 | group_number)")),
    family     = gaussian(),
    data       = df_scaled,        # z-scored predictors
    save_pars  = save_pars(all = TRUE),  # required for moment_match in LOO
    cores      = 4,
    chains     = 4,
    iter       = 6000,
    warmup     = 1000,
    seed       = 42
  )
  fits_list[[dep_var]] <- fit
}

saveRDS(fits_list, "plots/regression/fits_list.rds")

# Extract summary statistics for the posterior distributions
get_summary <- function(model, dep_var) {
  summary <- posterior_summary(model)
  summary_df <- as.data.frame(summary)
  summary_df$parameter <- rownames(summary_df)
  summary_df$dependent_variable <- dep_var
  summary_df
}

summary_list <- map2_dfr(fits_list, names(fits_list), get_summary)

summary_list <- summary_list %>%
  filter(str_detect(parameter, "b_") & parameter != "b_Intercept") %>%
  mutate(parameter = str_replace(parameter, "b_", ""))

dependent_variable_levels <- c("bestmatch", "opposite", "wordintrusion", "mostsimilar", "ffp", "topics", "autnes_sentiment")
dependent_variable_labels <- c(
  "bestmatch"        = "Intrinsic Task 1: Best Match",
  "opposite"         = "Intrinsic Task 2: Opposite",
  "wordintrusion"    = "Intrinsic Task 3: Word Intrusion",
  "mostsimilar"      = "Intrinsic Task 4: Grammar",
  "ffp"              = "Extrinsic Task 1: Author Prediction",
  "topics"           = "Extrinsic Task 2: Topic Prediction",
  "autnes_sentiment" = "Extrinsic Task 3: Sentiment Prediction"
)

summary_list <- summary_list %>%
  mutate(dependent_variable = factor(dependent_variable, levels = dependent_variable_levels))

min_x <- min(summary_list$Q2.5, na.rm = TRUE)
max_x <- max(summary_list$Q97.5, na.rm = TRUE)

ggplot(summary_list, aes(x = Estimate, y = parameter)) +
  geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey") +
  facet_wrap(~ dependent_variable, scales = "free_y",
             labeller = labeller(dependent_variable = dependent_variable_labels)) +
  scale_x_continuous(limits = c(min_x, max_x)) +
  labs(title = "Posterior Distributions of Model Parameters per Validation Task",
       x = "Coefficient Estimate",
       y = "Parameter") +
  theme_minimal() +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1))

# ------------------------------------------------------------------------------
# Figures 4 and 6: coefficient plots split into intrinsic and extrinsic tasks
# (same x-axis range for both figures, so they can be compared)
# ------------------------------------------------------------------------------
dir.create("plots/regression", recursive = TRUE, showWarnings = FALSE)

# Table 4: posterior means and 95% credible intervals (incl. intercept)
map2_dfr(fits_list, names(fits_list), function(m, dv) {
  ps <- as.data.frame(posterior_summary(m))
  ps$Parameter <- rownames(ps)
  ps %>%
    filter(str_detect(Parameter, "^b_")) %>%
    transmute(Task = dv, Parameter, Estimate, Q2.5, Q97.5)
}) %>%
  write.csv("plots/regression/table4_posterior.csv", row.names = FALSE)
intrinsic_tasks <- c("bestmatch", "opposite", "wordintrusion", "mostsimilar")
extrinsic_tasks <- c("ffp", "topics", "autnes_sentiment")

plot_coefficients <- function(data, title, ncol) {
  ggplot(data, aes(x = Estimate, y = parameter)) +
    geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey") +
    facet_wrap(~ dependent_variable, scales = "free_y", ncol = ncol,
               labeller = labeller(dependent_variable = dependent_variable_labels)) +
    scale_x_continuous(limits = c(min_x, max_x)) +
    labs(title = title,
         x = "Coefficient Estimate",
         y = "Parameter") +
    theme_minimal() +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1))
}

# Figure 4: intrinsic tasks
p_intrinsic <- plot_coefficients(
  summary_list %>% filter(dependent_variable %in% intrinsic_tasks),
  title = "Intrinsic Tasks: Posterior Distributions of Model Parameters",
  ncol  = 2
)
ggsave("plots/regression/plot_intrinsic_tasks.pdf", p_intrinsic,
       width = 8, height = 6, units = "in")

# Figure 6: extrinsic tasks
p_extrinsic <- plot_coefficients(
  summary_list %>% filter(dependent_variable %in% extrinsic_tasks),
  title = "Extrinsic Tasks: Posterior Distributions of Model Parameters",
  ncol  = 3
)
ggsave("plots/regression/plot_extrinsic_tasks.pdf", p_extrinsic,
       width = 10, height = 4, units = "in")

# ==============================================================================
# EFFECT SIZE ANALYSES
# ==============================================================================
# The ROPE is defined per task as ±10% of the observed performance range across
# the 320 self-trained models (reference models excluded). The coefficients
# are in outcome units per 1 SD change in the (z-scored) predictor.
# All five sections below produce per-dep_var summaries that can be combined
# into one master table at the end.
# ==============================================================================

hyperparams <- c("b_lower", "b_mincount", "b_windows")

# ── Helper: pretty section header ─────────────────────────────────────────────
section_header <- function(title) {
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(" ", title, "\n", sep = "")
  cat(strrep("=", 70), "\n\n", sep = "")
}

# ==============================================================================
# STEP 1 — ROPE (Region of Practical Equivalence)
#   Threshold: ±10% of each task's observed performance range across the
#              320 self-trained models (reference models excluded; same
#              ranges as in 08_task_ranges.py)
#   Interpretation: % of posterior falling INSIDE the ROPE → negligible effect
#                   % OUTSIDE the ROPE → practically meaningful effect
# ==============================================================================
section_header("STEP 1: ROPE Analysis (±10% of observed task range)")

# Self-trained models only (model families 2-33) for computing the task ranges
df_models <- df %>% filter(group_number >= 2, group_number <= 33)

rope_results_list <- list()

for (dep_var in dependent_vars_all) {
  
  # Compute ROPE bounds from the observed range of the outcome
  range_outcome <- diff(range(df_models[[dep_var]], na.rm = TRUE))
  rope_lower    <- -0.1 * range_outcome
  rope_upper    <-  0.1 * range_outcome
  
  rope_res <- rope(
    fits_list[[dep_var]],
    range      = c(rope_lower, rope_upper),
    ci         = 0.95,
    parameters = hyperparams
  )
  
  rope_df <- as.data.frame(rope_res) %>%
    mutate(
      dependent_variable = dep_var,
      rope_lower         = rope_lower,
      rope_upper         = rope_upper,
      range_outcome      = range_outcome
    )
  rope_results_list[[dep_var]] <- rope_df
  
  cat("Outcome:", dep_var,
      " | Range =", round(range_outcome, 4),
      " | ROPE = [", round(rope_lower, 4), ",", round(rope_upper, 4), "]\n")
  print(rope_df %>% select(any_of(c("Parameter", "CI", "ROPE_low", "ROPE_high",
                                    "ROPE_Percentage", "Decisions"))))
  cat("\n")
}

rope_results_all <- bind_rows(rope_results_list)


# ==============================================================================
# STEP 2 — Standardized Coefficients and effect-size labels
#   Because predictors were z-scored into df_scaled before fitting, the
#   posterior fixed-effect estimates are standardized with respect to the
#   predictors: change in the outcome (raw units: accuracy or F1) per 1 SD
#   change in the predictor. No refit needed — we read them directly from
#   posterior_summary(), avoiding the "bogus results" warning that
#   standardize_parameters() raises when priors were not scaled to match a
#   re-standardization step.
#
#   The outcome is NOT standardized, so Cohen's benchmarks (|β| < 0.20 =
#   negligible, ...) do not apply. Practical relevance is judged by the
#   task-specific ROPE from Step 1 (share of the 95% HDI inside the ROPE):
#     100% inside       → "negligible (entirely in ROPE)"
#     50% to < 100%     → "mostly negligible"
#     < 50% inside      → "practically relevant"
# ==============================================================================
section_header("STEP 2: Standardized Coefficients and ROPE-based effect-size labels")

std_results_list <- list()

for (dep_var in dependent_vars_all) {
  
  # posterior_summary() returns Estimate, Est.Error, Q2.5, Q97.5 for every param
  ps <- as.data.frame(posterior_summary(fits_list[[dep_var]]))
  ps$Parameter <- rownames(ps)
  
  std_df <- ps %>%
    filter(str_detect(Parameter, "^b_") & Parameter != "b_Intercept") %>%
    rename(Std_Coefficient = Estimate, CI_low = Q2.5, CI_high = Q97.5) %>%
    mutate(
      dependent_variable = dep_var,
      Parameter          = str_replace(Parameter, "b_", "")
    ) %>%
    left_join(
      rope_results_all %>%
        filter(dependent_variable == dep_var) %>%
        transmute(Parameter = str_replace(Parameter, "b_", ""), ROPE_Percentage),
      by = "Parameter"
    ) %>%
    mutate(
      effect_size_label = case_when(
        round(ROPE_Percentage, 3) == 1 ~ "negligible (entirely in ROPE)",
        ROPE_Percentage >= 0.5         ~ "mostly negligible",
        TRUE                           ~ "practically relevant"
      )
    ) %>%
    select(Parameter, Std_Coefficient, CI_low, CI_high,
           effect_size_label, dependent_variable)
  
  std_results_list[[dep_var]] <- std_df
  
  cat("Outcome:", dep_var, "\n")
  print(std_df %>% select(Parameter, Std_Coefficient, CI_low, CI_high, effect_size_label))
  cat("\n")
}

std_results_all <- bind_rows(std_results_list)


# ==============================================================================
# STEP 3 — Bayesian R² and Marginal Predictor Contribution (LOO comparison)
#   bayes_R2() gives overall model variance explained.
#   LOO comparison quantifies how much each hyperparameter contributes:
#   elpd_diff > 0 means the full model is better; effect sizes via SE ratio.
# ==============================================================================
section_header("STEP 3: Bayesian R² and LOO-based Predictor Contributions")

# Session guard: recreate df_scaled if this section is run independently
# (e.g. after loading saved fits_list from disk without re-running the full script)
if (!exists("df_scaled")) {
  cat("Note: df_scaled not found in session — recreating from df.\n\n")
  df_scaled <- df %>%
    mutate(
      lower    = as.numeric(scale(lower)),
      mincount = as.numeric(scale(mincount)),
      windows  = as.numeric(scale(windows))
    )
}

r2_results_list  <- list()
loo_results_list <- list()

for (dep_var in dependent_vars_all) {
  
  # --- Overall R² ---
  r2_full <- bayes_R2(fits_list[[dep_var]])
  r2_df   <- as.data.frame(r2_full) %>%
    mutate(dependent_variable = dep_var)
  r2_results_list[[dep_var]] <- r2_df
  
  cat("Outcome:", dep_var, "| Bayesian R²\n")
  print(r2_df)
  cat("\n")
  
  # --- LOO contribution per predictor ---
  # reloo = TRUE refits the model for the small number of observations where
  # pareto_k > 0.7, giving exact LOO for those points. This is the correct
  # fallback when save_pars(all=TRUE) was not set at fit time (which moment_match
  # requires). It is slower but always works.
  loo_full <- loo(fits_list[[dep_var]], reloo = TRUE)
  
  loo_drop_list <- list()
  for (hp in c("lower", "mincount", "windows")) {
    
    # Use update() rather than brm() for reduced models.
    # brm() can incorrectly reuse the cached Stan binary from the full model,
    # causing "variable z_1 does not exist" initialization errors when the
    # random-effects structure changes. update() forces a clean recompile.
    fit_reduced <- update(
      fits_list[[dep_var]],
      formula. = as.formula(
        paste(dep_var, "~",
              paste(setdiff(c("lower", "mincount", "windows"), hp), collapse = " + "),
              "+ (1 | group_number)")
      ),
      newdata = df_scaled,
      recompile = TRUE,
      refresh   = 0
    )
    loo_reduced <- loo(fit_reduced, reloo = TRUE)
    
    cmp <- loo_compare(loo_full, loo_reduced)
    # loo_compare puts best model first; elpd_diff for the worse model is negative
    elpd_diff <- cmp[2, "elpd_diff"]
    se_diff   <- cmp[2, "se_diff"]
    
    loo_drop_list[[hp]] <- data.frame(
      dependent_variable = dep_var,
      dropped_predictor  = hp,
      elpd_diff          = elpd_diff,   # negative = dropping hurts model fit
      se_diff            = se_diff,
      z_score            = elpd_diff / se_diff
    )
    cat("  Drop '", hp, "': elpd_diff =", round(elpd_diff, 2),
        " (SE =", round(se_diff, 2), ", z =", round(elpd_diff / se_diff, 2), ")\n")
  }
  
  loo_results_list[[dep_var]] <- bind_rows(loo_drop_list)
  cat("\n")
}

r2_results_all  <- bind_rows(r2_results_list)
loo_results_all <- bind_rows(loo_results_list)


# ==============================================================================
# STEP 4 — Probability of Direction (pd) + Practical Significance
#   pd: probability that the effect is strictly positive or negative.
#       pd > 97.5% ≈ p < 0.05 analogue.
#   p_significance: combines pd with ROPE → effect is both directional AND
#       outside the negligible zone (same ROPE as in Step 1: ±10% of range).
# ==============================================================================
section_header("STEP 4: Probability of Direction & Practical Significance")

# Self-trained models only (model families 2-33) for computing the task ranges
df_models <- df %>% filter(group_number >= 2, group_number <= 33)

pd_results_list <- list()

for (dep_var in dependent_vars_all) {
  
  range_outcome <- diff(range(df_models[[dep_var]], na.rm = TRUE))
  rope_range    <- c(-0.1 * range_outcome, 0.1 * range_outcome)
  
  # Probability of direction
  pd_res <- p_direction(fits_list[[dep_var]], parameters = hyperparams)
  
  # Practical significance (pd outside ROPE)
  ps_res <- p_significance(fits_list[[dep_var]],
                           threshold  = rope_range,
                           parameters = hyperparams)
  
  pd_df <- as.data.frame(pd_res) %>%
    rename(pd = pd) %>%
    left_join(as.data.frame(ps_res) %>% rename(ps = ps),
              by = "Parameter") %>%
    mutate(
      dependent_variable = dep_var,
      Parameter          = str_replace(Parameter, "b_", ""),
      pd_interpretation  = case_when(
        pd >= 0.999 ~ "extremely strong",
        pd >= 0.975 ~ "strong (≈ p<0.05)",
        pd >= 0.950 ~ "moderate",
        TRUE        ~ "weak"
      )
    )
  pd_results_list[[dep_var]] <- pd_df
  
  cat("Outcome:", dep_var, "\n")
  print(pd_df %>% select(Parameter, pd, pd_interpretation, ps))
  cat("\n")
}

pd_results_all <- bind_rows(pd_results_list)


# ==============================================================================
# STEP 5 — Rank Correlation Across Validation Tasks
#   Quantifies how much the different evaluation tasks AGREE on which model
#   configurations perform best. Low correlations indicate that the tasks
#   measure fundamentally different model properties.
# ==============================================================================
section_header("STEP 5: Rank Correlation of Models Across Validation Tasks")

# Build a wide table: one row per model configuration, one col per task
# We use the model's predicted mean (fitted values) as the performance score
# so all tasks are on a comparable scale.
predicted_scores <- map2_dfc(fits_list, names(fits_list), function(fit, dv) {
  tibble(!!dv := fitted(fit)[, "Estimate"])
})

# Rank within each task
ranked_scores <- predicted_scores %>%
  mutate(across(everything(), rank))

# Pairwise Spearman rank correlations
rank_cor_matrix <- cor(ranked_scores, method = "spearman")

cat("Pairwise Spearman rank correlations across validation tasks:\n\n")
print(round(rank_cor_matrix, 3))

mean_rank_cor <- mean(rank_cor_matrix[upper.tri(rank_cor_matrix)])
min_rank_cor  <- min(rank_cor_matrix[upper.tri(rank_cor_matrix)])
max_rank_cor  <- max(rank_cor_matrix[upper.tri(rank_cor_matrix)])

cat("\nSummary of off-diagonal correlations:\n")
cat("  Mean ρ =", round(mean_rank_cor, 3),
    "| Min ρ =", round(min_rank_cor, 3),
    "| Max ρ =", round(max_rank_cor, 3), "\n")
cat("\nInterpretation:\n")
cat("  ρ > 0.80 → tasks largely agree on best models\n")
cat("  ρ 0.50–0.80 → moderate agreement\n")
cat("  ρ < 0.50 → substantial disagreement (tasks capture different model properties)\n\n")


# save the rank correlations
write.csv(round(rank_cor_matrix, 3), "plots/regression/rank_correlations.csv")
cat("Saved to: plots/regression/rank_correlations.csv\n")

# Visualize as a heatmap
corrplot(
  rank_cor_matrix,
  method      = "color",
  type        = "upper",
  addCoef.col = "black",
  tl.cex      = 0.8,
  cl.cex      = 0.8,
  number.cex  = 0.75,
  col         = colorRampPalette(c("#d73027", "#fee090", "#4575b4"))(200),
  title       = "Spearman Rank Correlation of Models Across Validation Tasks",
  mar         = c(0, 0, 2, 0)
)


# cleaner labels
dep_labels <- c(
  "bestmatch"        = "Intrinsic Task 1:\nBest Match",
  "opposite"         = "Intrinsic Task 2:\nOpposite",
  "wordintrusion"    = "Intrinsic Task 3:\nWord Intrusion",
  "mostsimilar"      = "Intrinsic Task 4:\nGrammar",
  "ffp"              = "Extrinsic Task 1:\nAuthor Prediction",
  "topics"           = "Extrinsic Task 2:\nTopic Prediction",
  "autnes_sentiment" = "Extrinsic Task 3:\nSentiment Prediction"
)

pub_cols <- colorRampPalette(c("#ffffff", "#0063A6"))(200)

# safely replace column names BEFORE plotting
colnames(rank_cor_matrix) <- dependent_variable_labels[colnames(rank_cor_matrix)]
rownames(rank_cor_matrix) <- dependent_variable_labels[rownames(rank_cor_matrix)]

corrplot(
  rank_cor_matrix,
  method = "color",
  type = "upper",
  
  col = pub_cols,
  
  tl.col = "black",
  tl.cex = 0.85,
  tl.srt = 45,
  
  addCoef.col = "black",
  number.cex = 0.5,   # smaller numbers
  cl.cex = 0.85,
  
  mar = c(0, 0, 2, 0),
  
  main = "Spearman Rank Correlation of Models Across Validation Tasks",
  cex.main = 1.1
)

# ==============================================================================
# MASTER SUMMARY TABLE
# Combines standardized β, ROPE %, pd, and ps into one printable table
# per dep_var.
# ==============================================================================
section_header("MASTER SUMMARY: Effect Sizes per Task and Hyperparameter")

# Prepare each piece
rope_summary <- rope_results_all %>%
  select(Parameter, dependent_variable, ROPE_Percentage) %>%
  mutate(Parameter = str_replace(Parameter, "b_", ""))

std_summary <- std_results_all %>%
  select(Parameter, dependent_variable, Std_Coefficient, effect_size_label)

pd_summary <- pd_results_all %>%
  select(Parameter, dependent_variable, pd, pd_interpretation, ps)

master_table <- std_summary %>%
  left_join(rope_summary, by = c("Parameter", "dependent_variable")) %>%
  left_join(pd_summary,   by = c("Parameter", "dependent_variable")) %>%
  left_join(
    r2_results_all %>% select(dependent_variable, Estimate) %>%
      rename(bayes_R2 = Estimate),
    by = "dependent_variable"
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
  rename(
    Task           = dependent_variable,
    Hyperparameter = Parameter,
    `Std β`        = Std_Coefficient,
    `Effect size`  = effect_size_label,
    `% in ROPE`    = ROPE_Percentage,
    `pd`           = pd,
    `pd label`     = pd_interpretation,
    `p(significance)` = ps,
    `Bayes R²`     = bayes_R2
  ) %>%
  arrange(Task, Hyperparameter)

cat("Full effect size summary (one row per task × hyperparameter):\n\n")
print(master_table)

# Optionally write to CSV
write.csv(master_table,
          file = "plots/regression/effect_size_summary.csv",
          row.names = FALSE)
cat("\nSaved to: plots/regression/effect_size_summary.csv\n")
