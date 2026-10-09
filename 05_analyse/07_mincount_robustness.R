### Robustness check: effect of min_count by task type (Appendix, Figure 11)

library(tidyverse)
library(openxlsx)
library(reshape2)
library(psych)
library(data.table)
library(car)
library(brms)
library(patchwork)

df <- read.csv("evaluation_results/dataset_regression_rebuilt.csv")
row.names(df) <- df$X
df <- df[, -which(names(df) == "X")]

df <- df %>% filter(group_number >= 2, group_number <= 33)
stopifnot(nrow(df) == 320, length(unique(df$group_number)) == 32)

# ── 1. Create df_long ───────────────────────────────────────────────────
df_long <- df %>%
  rownames_to_column("model_id") %>%
  pivot_longer(
    cols      = all_of(c("autnes_sentiment", "ffp", "topics",
                         "bestmatch", "mostsimilar", "opposite", "wordintrusion")),
    names_to  = "task",
    values_to = "score"
  ) %>%
  mutate(
    task_type    = case_when(
      task %in% c("autnes_sentiment", "ffp", "topics") ~ "extrinsic",
      TRUE                                              ~ "intrinsic"
    ),
    task_type    = factor(task_type, levels = c("extrinsic", "intrinsic")),
    mincount_num = as.numeric(as.character(mincount)),
    mincount_z   = scale(mincount_num)[,1],
    lower_z      = scale(as.numeric(as.character(lower)))[,1],
    windows_z    = scale(as.numeric(as.character(windows)))[,1],
    score_z      = scale(score)[,1]
  )

# ── 2. Interaction model (existing) ───────────────────────────────────────────
fit_interaction <- brm(
  formula = score ~ mincount_z * task_type + lower_z + windows_z +
    (1 | group_number) + (1 | task),
  family  = gaussian(),
  data    = df_long,
  cores   = 4, chains = 4, iter = 6000, warmup = 1000,
  prior   = c(
    prior(normal(0, 1), class = b),
    prior(exponential(1), class = sd)
  )
)

posterior_summary(fit_interaction) %>%
  as.data.frame() %>%
  rownames_to_column("parameter") %>%
  filter(str_detect(parameter, "mincount|task_type"))

as_draws_df(fit_interaction) %>%
  transmute(
    effect_extrinsic = b_mincount_z,
    effect_intrinsic = b_mincount_z + `b_mincount_z:task_typeintrinsic`,
    interaction      = `b_mincount_z:task_typeintrinsic`
  ) %>%
  summarise(across(everything(),
                   list(mean = mean,
                        lo   = ~ quantile(.x, 0.025),
                        hi   = ~ quantile(.x, 0.975))))

posterior_samples <- as_draws_df(fit_interaction)

effects <- posterior_samples %>%
  transmute(
    effect_extrinsic = b_mincount_z,
    effect_intrinsic = b_mincount_z + `b_mincount_z:task_typeintrinsic`
  ) %>%
  pivot_longer(everything(), names_to = "task_type", values_to = "estimate") %>%
  mutate(task_type = str_remove(task_type, "effect_"))

ggplot(effects, aes(x = estimate, fill = task_type)) +
  geom_density(alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values = c("extrinsic" = "#0072B2", "intrinsic" = "#E69F00")) +
  labs(
    title    = "Effect of min_count on Performance by Task Type",
    x        = "Posterior Estimate of min_count Effect",
    y        = "Density",
    fill     = "Task Type"
  ) +
  theme_minimal()

hypothesis(fit_interaction, "mincount_z:task_typeintrinsic = 0")

posterior_summary(fit_interaction) %>%
  as.data.frame() %>%
  rownames_to_column("parameter") %>%
  filter(str_detect(parameter, "mincount"))


# ── OPTION A: Pooled model, min_count as fixed effect, no interaction ─────────
# If min_count has a credible effect when
# pooling across ALL tasks (intrinsic + extrinsic), it reflects genuine quality.
# Task-level variance is absorbed by (1 | task) random effect.

fit_pooled <- brm(
  formula = score ~ mincount_z + lower_z + windows_z +
    (1 | group_number) + (1 | task),
  family  = gaussian(),
  data    = df_long,
  cores   = 4, chains = 4, iter = 6000, warmup = 1000,
  prior   = c(
    prior(normal(0, 1), class = b),
    prior(exponential(1), class = sd)
  )
)

summary(fit_pooled)

# Extract min_count estimate for reporting
posterior_summary(fit_pooled) %>%
  as.data.frame() %>%
  rownames_to_column("parameter") %>%
  filter(str_detect(parameter, "mincount|lower|windows"))


# ── OPTION B: Separate model per task, then plot min_count estimates ──────────
# Re-runs one brm per task and extracts only the min_count coefficient.
# The key visual: does min_count have a credible effect in extrinsic tasks too?

dependent_vars_all <- c("autnes_sentiment", "ffp", "topics",
                        "bestmatch", "mostsimilar", "opposite", "wordintrusion")

fits_list <- list()

for (dep_var in dependent_vars_all) {
  message("Fitting model for: ", dep_var)
  fits_list[[dep_var]] <- brm(
    formula = as.formula(paste(dep_var, "~ mincount_z + lower_z + windows_z + (1 | group_number)")),
    family  = gaussian(),
    data    = df %>% mutate(
      mincount_z = scale(as.numeric(as.character(mincount)))[,1],
      lower_z    = scale(as.numeric(as.character(lower)))[,1],
      windows_z  = scale(as.numeric(as.character(windows)))[,1]
    ),
    cores   = 4, chains = 4, iter = 6000, warmup = 1000
  )
}

# Extract min_count coefficient from each task model
mincount_estimates <- map2_dfr(fits_list, names(fits_list), function(model, dep_var) {
  posterior_summary(model) %>%
    as.data.frame() %>%
    rownames_to_column("parameter") %>%
    filter(parameter == "b_mincount_z") %>%
    mutate(
      task      = dep_var,
      task_type = case_when(
        dep_var %in% c("autnes_sentiment", "ffp", "topics") ~ "Extrinsic",
        TRUE                                                 ~ "Intrinsic"
      )
    )
})

# Custom labels for plot
task_labels <- c(
  "bestmatch"       = "Intrinsic 1: Best Match",
  "opposite"        = "Intrinsic 2: Opposite",
  "wordintrusion"   = "Intrinsic 3: Word Intrusion",
  "mostsimilar"     = "Intrinsic 4: Grammar",
  "ffp"             = "Extrinsic 1: Author Prediction",
  "topics"          = "Extrinsic 2: Topic Prediction",
  "autnes_sentiment" = "Extrinsic 3: Sentiment Prediction"
)

mincount_estimates <- mincount_estimates %>%
  mutate(task = factor(task,
                       levels  = c("bestmatch", "opposite", "wordintrusion", "mostsimilar",
                                   "ffp", "topics", "autnes_sentiment"),
                       labels  = task_labels[c("bestmatch", "opposite", "wordintrusion",
                                               "mostsimilar", "ffp", "topics", "autnes_sentiment")]))

# Plot: if CI excludes zero for extrinsic tasks → supports genuine quality argument
ggplot(mincount_estimates,
       aes(x = Estimate, y = task, color = task_type)) +
  geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5), size = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c("Extrinsic" = "#0072B2", "Intrinsic" = "#E69F00")) +
  labs(
    title    = "Effect of min_count per Task (Standardized)",
    x        = "Posterior Estimate of min_count Effect (standardized)",
    y        = NULL,
    color    = "Task Type"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")





# ── 1. Density plot (interaction model) ─────────────────────────────
p_density <- ggplot(effects, aes(x = estimate, fill = task_type)) +
  geom_density(alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values = c("extrinsic" = "#0072B2", "intrinsic" = "#E69F00")) +
  labs(
    title = "Posterior Distribution of min_count Effects",
    x     = "Posterior Estimate",
    y     = "Density",
    fill  = "Task Type"
  ) +
  theme_minimal()

# ── 2. Task-level coefficient plot ───────────────────────────────────
p_taskwise <- ggplot(mincount_estimates,
                     aes(x = Estimate, y = task, color = task_type)) +
  geom_pointrange(aes(xmin = Q2.5, xmax = Q97.5), size = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c("Extrinsic" = "#0072B2", "Intrinsic" = "#E69F00")) +
  labs(
    title = "Task-level Effects of min_count",
    x     = "Posterior Estimate (standardized)",
    y     = NULL,
    color = "Task Type"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# ── 3. Combine into one multi-panel figure ──────────────────────────
combined_plot <- (p_density | p_taskwise) +
  plot_annotation(
    title = "Effects of min_count on Performance Across Task Types",
    tag_levels = "A"
  )

# ── 4. Save to PDF ───────────────────────────────────────────────────
dir.create("plots/regression", recursive = TRUE, showWarnings = FALSE)

ggsave(
  filename = "plots/regression/figure_combined_mincount_analysis.pdf",
  plot     = combined_plot,
  device   = pdf,
  width    = 11,
  height   = 6,
  units    = "in"
)
