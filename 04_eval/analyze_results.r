library(RcppSimdJson)
library(dplyr)
library(forcats)
library(ggplot2)
library(ggthemes)
library(viridis)
library(stringr)
library(reshape2)

model_meta <- RcppSimdJson::fload(Sys.glob('tmp_models/*/*.json'))
model_meta <- dplyr::bind_rows(model_meta)


# keep only models trained on VSC

filt <- grepl('fs72169', model_meta$model_path)
model_meta <- model_meta[filt, ]

model_meta$model_id <- paste(model_meta$name, model_meta$parameter_string, sep = '_')

model_meta$lowercase <- grepl('_lower', model_meta$training_corpus)
model_meta$computation_time_hours <- model_meta$computation_time / 60 / 60

model_families <- model_meta |> select(model_type, min_count, dimensions, window_size, word_ngrams, epochs, learning_rate, lowercase, parameter_string) |>
  distinct()

# Computation Time

p <- model_meta |>
  mutate(window_size = as.factor(window_size)) |>
  rename(`Window Size` = window_size,
         `Computation Time (hours)` = computation_time_hours,
         `Minimum Count` = min_count) |>
  ggplot(aes(x=`Window Size`, y=`Computation Time (hours)`, fill=lowercase)) +
  geom_boxplot() +
  facet_wrap(~`Minimum Count`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = 'mako', begin = 0.4, end = 0.6) +
  ggtitle('Training Duration')

ggsave('plots/training_duration.png', p, width = 1920, height = 1080, units = 'px', scale = 2)

# Correlations: Within

correlations_within <- RcppSimdJson::fload(Sys.glob('evaluation_results/within_correlations/*.json'))
correlations_within <- dplyr::bind_rows(correlations_within)
correlations_within <- left_join(correlations_within, model_families, by = 'parameter_string')

p <- correlations_within |>
  filter(cues == 'random') |>
  mutate(window_size = as.factor(window_size)) |>
  rename(`Window Size` = window_size,
         `Within-Correlation`= correlation,
         `Minimum Count` = min_count) |>
  ggplot(aes(y=`Within-Correlation`, x=`Window Size`, fill=lowercase)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(0.9, 1.0)) +
  facet_wrap(~`Minimum Count`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = 'mako', begin = 0.4, end = 0.6) +
  ggtitle('Within Correlations (Random Cues)')

ggsave('plots/within_correlation.png', p, width = 1920, height = 1080, units = 'px', scale = 2)


# Correlations: Across

correlations_across <- RcppSimdJson::fload(Sys.glob('evaluation_results/across_correlations/results.json'))

# create duplicate that is flipped
correlations_across_b <- correlations_across
correlations_across_b$model_a_family <- correlations_across$model_b_family
correlations_across_b$model_b_family <- correlations_across$model_a_family

correlations_across <- rbind(correlations_across, correlations_across_b)

correlations_across$model_a_min_count <- str_extract(correlations_across$model_a_family, "mincount(\\d{1,3})_", group = 1)
correlations_across$model_a_min_count <- str_pad(correlations_across$model_a_min_count, 3, "left", pad = "0")

correlations_across$model_b_min_count <- str_extract(correlations_across$model_b_family, "mincount(\\d{1,3})_", group = 1)
correlations_across$model_b_min_count <- str_pad(correlations_across$model_b_min_count, 3, "left", pad = "0")

correlations_across$model_a_window_size <- str_extract(correlations_across$model_a_family, "ws(\\d{1,3})_", group = 1)
correlations_across$model_a_window_size <- str_pad(correlations_across$model_a_window_size, 2, "left", pad = "0")

correlations_across$model_b_window_size <- str_extract(correlations_across$model_b_family, "ws(\\d{1,3})_", group = 1)
correlations_across$model_b_window_size <- str_pad(correlations_across$model_b_window_size, 2, "left", pad = "0")


correlations_across$model_a_short <- paste(correlations_across$model_a_min_count, correlations_across$model_a_window_size, sep = "_")
correlations_across$model_b_short <- paste(correlations_across$model_b_min_count, correlations_across$model_b_window_size, sep = "_")

models_names <- unique(correlations_across$model_a_short)
models_names <- models_names[order(models_names)]

m <- Matrix(data = NA,
       nrow = length(models_names),
       ncol = length(models_names),
       dimnames = list(models_names, models_names))

# I hate myself for doing this

for (i in 1:nrow(correlations_across)) {
  model_a <- correlations_across[i, "model_a_short"]
  model_b <- correlations_across[i, "model_b_short"]
  m[model_a, model_b] <- correlations_across[i, 'correlation']
}

m[upper.tri(m)] <- NA

correlations <- reshape2::melt(as.matrix(m))

p <- correlations |>
  filter(!is.na(value)) |>
  rename(`Model Group A` = Var1, `Model Group B` = Var2, `Across Correlations` = value) |>
  ggplot(aes(`Model Group A`, `Model Group B`, fill = `Across Correlations`)) +
  geom_tile() +
  coord_fixed() +
  theme_clean() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                     size = 12, hjust = 1),
        panel.grid.major.y = element_blank()) +
  scale_fill_viridis(option = 'mako') +
  ggtitle('Across Correlations (Random Cues)')

ggsave('plots/across_correlation.png', p, width = 1920, height = 1080, units = 'px', scale = 2)

# syntactic / semantic

syntactic <- RcppSimdJson::fload(Sys.glob('evaluation_results/semantic_syntactic/*/*.json'))
syntactic <- dplyr::bind_rows(syntactic)

syntactic$model_id <- paste(syntactic$name, syntactic$parameter_string, sep = '_')
syntactic <- left_join(syntactic, model_meta, by = 'model_id')
syntactic$`Coverage (%)` <- round((syntactic$coverage / syntactic$total_questions) * 100, digits = 2)

# All models achieved 100% coverage!

syntactic$`Correct (%)` <- round((syntactic$correct / syntactic$coverage) * 100, digits = 2)
syntactic$`Correct (Top 10, %)` <- round((syntactic$top_n / syntactic$coverage) * 100, digits = 2)
syntactic$`Sub-Task` <- as.factor(syntactic$task)
syntactic$`Task` <- as.factor(syntactic$task_group)
syntactic$`Task` <- fct_recode(syntactic$`Task`, "Best match (Semantic)" = "most_similar", "Syntactic" = "most_similar_groups", "Word Intrusion (Semantic)" = "word intrusion")

p <- syntactic |>
  filter(task != 'total') |>
  mutate(window_size = as.factor(window_size)) |>
  rename(`Window Size` = window_size,
         `Computation Time (hours)` = computation_time_hours,
         `Minimum Count` = min_count) |>
  ggplot(aes(x=`Window Size`, y=`Correct (%)`, fill=lowercase)) +
  geom_boxplot() +
  facet_wrap(~`Task`, labeller = "label_both", ncol = 3) +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = 'mako', begin = 0.4, end = 0.6) +
  ggtitle('Syntactic / Semantic (strict)')

ggsave('plots/results_syntactic_semantic_strict.png', p, width = 1920, height = 1080, units = 'px', scale = 2)


p <- syntactic |>
  filter(task != 'total') |>
  filter(!is.na(`Correct (Top 10, %)`)) |>
  mutate(window_size = as.factor(window_size)) |>
  rename(`Window Size` = window_size,
         `Computation Time (hours)` = computation_time_hours,
         `Minimum Count` = min_count) |>
  ggplot(aes(x=`Window Size`, y=`Correct (Top 10, %)`, fill=lowercase)) +
  geom_boxplot() +
  facet_wrap(~`Task`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = 'mako', begin = 0.4, end = 0.6) +
  ggtitle('Syntactic / Semantic (Top 10)')

ggsave('plots/results_syntactic_semantic_top10.png', p, width = 1920, height = 1080, units = 'px', scale = 2)


# Classification
#
# classification <- RcppSimdJson::fload(Sys.glob('evaluation_results/classification/*.json'))
# syntactic <- dplyr::bind_rows(syntactic)
#
# syntactic$model_id <- paste(syntactic$name, syntactic$parameter_string, sep = '_')
# syntactic <- left_join(syntactic, model_meta, by = 'model_id')
