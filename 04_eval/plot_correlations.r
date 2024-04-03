library(RcppSimdJson)
library(dplyr)
library(forcats)
library(ggplot2)
library(ggthemes)
library(viridis)
library(stringr)
library(reshape2)
library(Matrix)

if (!dir.exists("plots")) {
  dir.create("plots")
}

model_meta <- RcppSimdJson::fload(Sys.glob("tmp_models/*/*.json"))
model_meta <- dplyr::bind_rows(model_meta)

# keep only models trained on VSC

filt <- grepl("fs72169", model_meta$model_path)
model_meta <- model_meta[filt, ]

model_meta$model_id <- paste(model_meta$name, model_meta$parameter_string, sep = "_")

model_meta$lowercase <- grepl("_lower", model_meta$training_corpus)
model_meta$computation_time_hours <- model_meta$computation_time / 60 / 60

model_families <- model_meta |>
  select(model_type, min_count, dimensions, window_size, word_ngrams, epochs, learning_rate, lowercase, parameter_string) |>
  distinct()

# Computation Time
p <- model_meta |>
  mutate(window_size = as.factor(window_size)) |>
  rename(
    `Window Size` = window_size,
    `Computation Time (hours)` = computation_time_hours,
    `Minimum Count` = min_count
  ) |>
  ggplot(aes(x = `Window Size`, y = `Computation Time (hours)`, fill = lowercase)) +
  geom_boxplot() +
  facet_wrap(~`Minimum Count`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.4, end = 0.6) +
  ggtitle("Training Duration")

ggsave("plots/training_duration.png", p, width = 1920, height = 1080, units = "px", scale = 2)

# Correlations: Within

if (!dir.exists("plots/within_correlation")) {
  dir.create("plots/within_correlation")
}

correlations_within <- RcppSimdJson::fload(Sys.glob("evaluation_results/within_correlations/*.json"))
correlations_within <- dplyr::bind_rows(correlations_within)
correlations_within <- left_join(correlations_within, model_families, by = "parameter_string")

p <- correlations_within |>
  mutate(window_size = as.factor(window_size)) |>
  rename(
    `Window Size` = window_size,
    `Within-Correlation` = correlation,
    `Minimum Count` = min_count
  ) |>
  ggplot(aes(y = `Within-Correlation`, x = `Window Size`, fill = lowercase)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(0.93, 1.0)) +
  facet_wrap(~`Minimum Count`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.4, end = 0.6) +
  ggtitle("Within Correlations (All Cues)")

ggsave("plots/within_correlation/within_correlation.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/within_correlation/within_correlation.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)

p <- correlations_within |>
  mutate(window_size = as.factor(window_size)) |>
  rename(
    `Window Size` = window_size,
    `Within-Correlation` = correlation,
    `Minimum Count` = min_count
  ) |>
  ggplot(aes(y = `Within-Correlation`, x = `Window Size`, fill = lowercase)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(0.93, 1.0)) +
  facet_wrap(~`cues`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.4, end = 0.6) +
  ggtitle("Within Correlations")

ggsave("plots/within_correlation/within_correlation_cues.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/within_correlation/within_correlation_cues.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)


# Correlations: Across

if (!dir.exists("plots/across_correlation")) {
  dir.create("plots/across_correlation")
}

add_flipped <- function(x) {
  # create duplicate that is flipped
  y <- x
  y$model_a_short <- x$model_b_short
  y$model_b_short <- x$model_a_short
  x <- rbind(x, y)
  return(x)
}

correlations_across <- RcppSimdJson::fload(Sys.glob("evaluation_results/across_correlations/*.json"),
  parse_error_ok = TRUE
)

correlations_across <- bind_rows(correlations_across)
correlations_across$lowercase <- str_detect(correlations_across$model_a_family, "_lower_") | str_detect(correlations_across$model_b_family, "_lower_")

# Boxplots of cues

p <- correlations_across |>
  group_by(cues) |>
  ggplot(aes(y = correlation, x = cues, fill = lowercase)) +
  geom_boxplot() +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.4, end = 0.6) +
  ggtitle("Across Correlations")

ggsave("plots/across_correlation/across_correlation_variation.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/across_correlation/across_correlation_variation.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)


# rename families to short names

filta <- correlations_across$model_a_family == "cc_de_300"
filtb <- correlations_across$model_b_family == "cc_de_300"

correlations_across$model_a_min_count <- str_extract(correlations_across$model_a_family, "mincount(\\d{1,3})_", group = 1)
correlations_across$model_a_min_count[filta] <- 5
correlations_across$model_a_min_count <- str_pad(correlations_across$model_a_min_count, 3, "left", pad = "0")

correlations_across$model_b_min_count <- str_extract(correlations_across$model_b_family, "mincount(\\d{1,3})_", group = 1)
correlations_across$model_b_min_count[filtb] <- 5
correlations_across$model_b_min_count <- str_pad(correlations_across$model_b_min_count, 3, "left", pad = "0")

correlations_across$model_a_window_size <- str_extract(correlations_across$model_a_family, "ws(\\d{1,3})_", group = 1)
correlations_across$model_a_window_size[filta] <- 5
correlations_across$model_a_window_size <- str_pad(correlations_across$model_a_window_size, 2, "left", pad = "0")

correlations_across$model_b_window_size <- str_extract(correlations_across$model_b_family, "ws(\\d{1,3})_", group = 1)
correlations_across$model_b_window_size[filtb] <- 5
correlations_across$model_b_window_size <- str_pad(correlations_across$model_b_window_size, 2, "left", pad = "0")

correlations_across$model_a_short <- paste(correlations_across$model_a_min_count, correlations_across$model_a_window_size, sep = "_")
correlations_across$model_b_short <- paste(correlations_across$model_b_min_count, correlations_across$model_b_window_size, sep = "_")
correlations_across$model_a_short[filta] <- "CC_DE_300"
correlations_across$model_b_short[filtb] <- "CC_DE_300"


# Nested loop:
# we need correlation plots for all kinds of cues (including a mean)
# and additionally every time need plots for the lowercased models

for (cue in c("mean", unique(correlations_across$cues))) {
  if (cue == "mean") {
    # Calculate mean for all Cues
    correlations_across_cue <- correlations_across |>
      group_by(model_a_short, model_b_short, lowercase) |>
      summarise(correlation = mean(correlation), correlation_sd = mean(correlation_sd))
  } else {
    correlations_across_cue <- correlations_across |>
      filter(cues == cue)
  }

  for (lowercasing in c(TRUE, FALSE)) {
    filt <- correlations_across_cue$lowercase == lowercasing
    correlations_across_cue_subset <- correlations_across_cue[filt, ]

    correlations_across_cue_subset <- add_flipped(correlations_across_cue_subset)

    models_names <- unique(correlations_across_cue_subset$model_a_short)
    models_names <- models_names[order(models_names)]

    # reformat the data to get a pretty plot
    # we only want the values of the lower triangle in a matrix
    # so we make the data wide and then long again
    m <- Matrix(
      data = NA,
      nrow = length(models_names),
      ncol = length(models_names),
      dimnames = list(models_names, models_names)
    )

    # I hate myself for doing this

    for (i in 1:nrow(correlations_across_cue_subset)) {
      model_a <- correlations_across_cue_subset[i, "model_a_short"][[1]]
      model_b <- correlations_across_cue_subset[i, "model_b_short"][[1]]
      m[model_a, model_b] <- correlations_across_cue_subset[i, "correlation"][[1]]
    }

    m[upper.tri(m)] <- NA
    diag(m) <- 1
    correlations <- reshape2::melt(as.matrix(m))

    casing <- if (lowercasing) "lowercase" else "regular"
    plot_title <- paste0("Across Correlations (", cue, " ", casing, ")")

    p <- correlations |>
      filter(!is.na(value)) |>
      rename(`Model Group A` = Var1, `Model Group B` = Var2, `Across Correlations` = value) |>
      ggplot(aes(`Model Group A`, `Model Group B`, fill = `Across Correlations`)) +
      geom_tile() +
      coord_fixed() +
      # theme_clean() +
      theme(
        axis.text.x = element_text(
          angle = 45, vjust = 1,
          size = 12, hjust = 1
        ),
        panel.grid.major.y = element_blank(),
        legend.position = "bottom",
        panel.background = element_blank()
      ) +
      scale_fill_viridis("Pearson's Rho",
        option = "mako",
        direction = -1
      ) +
      ggtitle(plot_title)

    ggsave(paste0("plots/across_correlation/across_correlation_", cue, "_", casing, ".png"),
      p,
      width = 1080,
      height = 1080,
      units = "px",
      scale = 2
    )
    ggsave(paste0("plots/across_correlation/across_correlation_", cue, "_", casing, ".pdf"),
      p,
      width = 1080,
      height = 1080,
      units = "px",
      scale = 2
    )
  }
}

# syntactic / semantic

if (!dir.exists("plots/semantic_syntactic")) {
  dir.create("plots/semantic_syntactic")
}

syntactic <- RcppSimdJson::fload(Sys.glob("evaluation_results/semantic_syntactic/*/*.json"))
syntactic <- dplyr::bind_rows(syntactic)

syntactic$model_id <- paste(syntactic$name, syntactic$parameter_string, sep = "_")
syntactic <- left_join(syntactic, model_meta, by = "model_id")
syntactic$`Coverage (%)` <- round((syntactic$coverage / syntactic$total_questions) * 100, digits = 2)

# All models achieved 100% coverage!

syntactic$`Correct (%)` <- round((syntactic$correct / syntactic$coverage) * 100, digits = 2)
syntactic$`Correct (Top 10, %)` <- round((syntactic$top_n / syntactic$coverage) * 100, digits = 2)
syntactic$`Sub-Task` <- as.factor(syntactic$task)
syntactic$`Task` <- as.factor(syntactic$task_group)
syntactic$`Task` <- fct_recode(syntactic$`Task`, "Best match (Semantic)" = "most_similar", "Syntactic" = "most_similar_groups", "Word Intrusion (Semantic)" = "word intrusion")

p <- syntactic |>
  filter(task != "total") |>
  filter(!is.na(name.x)) |>
  mutate(window_size = as.factor(window_size)) |>
  rename(
    `Window Size` = window_size,
    `Computation Time (hours)` = computation_time_hours,
    `Minimum Count` = min_count
  ) |>
  ggplot(aes(x = `Window Size`, y = `Correct (%)`, fill = lowercase)) +
  geom_boxplot() +
  facet_wrap(~`Task`, labeller = "label_both", ncol = 3) +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.4, end = 0.6) +
  ggtitle("Syntactic / Semantic (strict)")

ggsave("plots/semantic_syntactic/results_strict.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/semantic_syntactic/results_strict.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)


p <- syntactic |>
  filter(task != "total") |>
  filter(!is.na(name.x)) |>
  filter(!is.na(`Correct (Top 10, %)`)) |>
  mutate(window_size = as.factor(window_size)) |>
  rename(
    `Window Size` = window_size,
    `Computation Time (hours)` = computation_time_hours,
    `Minimum Count` = min_count
  ) |>
  ggplot(aes(x = `Window Size`, y = `Correct (Top 10, %)`, fill = lowercase)) +
  geom_boxplot() +
  facet_wrap(~`Task`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.4, end = 0.6) +
  ggtitle("Syntactic / Semantic (Top 10)")

ggsave("plots/semantic_syntactic/results_top10.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/semantic_syntactic/results_top10.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)
