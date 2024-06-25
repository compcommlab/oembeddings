library(RcppSimdJson)
library(arrow)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(viridis)
library(forcats)

if (!dir.exists("plots")) {
  dir.create("plots")
}


model_meta <- read_feather("evaluation_results/fasttext_models_meta.feather")
model_families <- read_feather("evaluation_results/fasttext_model_families.feather")


if (!dir.exists("plots/semantic_syntactic")) {
  dir.create("plots/semantic_syntactic")
}

syntactic <- RcppSimdJson::fload(c(
    Sys.glob("evaluation_results/*/semantic_syntactic/*/*.json"),
    Sys.glob("evaluation_results/*/semantic_syntactic/*.json")))
syntactic <- dplyr::bind_rows(syntactic)

syntactic$model_id <- paste(syntactic$name, syntactic$parameter_string, sep = "_")
syntactic <- left_join(syntactic, model_meta, by = "model_id", relationship = "many-to-many")
syntactic$`Coverage (%)` <- round((syntactic$coverage / syntactic$total_questions) * 100, digits = 2)

syntactic$`Correct (%)` <- round((syntactic$correct / syntactic$coverage) * 100, digits = 2)
syntactic$`Correct (Top 10, %)` <- round((syntactic$top_n / syntactic$coverage) * 100, digits = 2)
syntactic$`Sub-Task` <- as.factor(syntactic$task)
syntactic$`Task` <- case_when(syntactic$task == "opposite" ~ "Opposite (Semantic)",
                              .default = syntactic$task_group)
syntactic$`Task` <- as.factor(syntactic$`Task`)
syntactic$`Task` <- forcats::fct_recode(syntactic$`Task`, 
                                        "Best match (Semantic)" = "most_similar", 
                                        "Syntactic" = "most_similar_groups", 
                                        "Word Intrusion (Semantic)" = "word intrusion")


p <- syntactic |> 
    mutate(window_size = as.factor(window_size)) |>
    rename(
      `Window Size` = window_size,
      `Minimum Count` = min_count
    ) |>
  ggplot(aes(x = `Task`, y = `Coverage (%)`, fill = `Model Group`))+
  geom_boxplot() +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.2, end = 0.8) +
  ggtitle("Vocabulary coverage for Syntactic / Semantic Tasks")

ggsave("plots/semantic_syntactic/coverage.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/semantic_syntactic/coverage.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)

p <- syntactic |>
  filter(task != "total") |>
  filter(!is.na(name.x)) |>
  mutate(`Window Size` = as.factor(window_size)) |>
  ggplot(aes(x = `Window Size`, y = `Correct (%)`, fill = `Model Group`)) +
  geom_boxplot() +
  facet_wrap(~`Task`, labeller = "label_both", ncol = 4) +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako") +
  theme(legend.position = "top", plot.background = element_blank())

ggsave("plots/semantic_syntactic/results_strict.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/semantic_syntactic/results_strict.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)


p <- syntactic |>
  filter(task == "best match") |>
  filter(!is.na(name.x)) |>
  filter(`Model Group` %in% c("OEmbeddings Cased", "OEmbeddings Lowercase")) |> 
  mutate(`Window Size` = as.factor(window_size),
         `Minimum Count` = as.factor(min_count)) |>
  ggplot(aes(x = `Window Size`, y = `Correct (%)`, fill = `Minimum Count`)) +
  geom_boxplot() +
  facet_wrap(~`Model Group`, labeller = "label_both", ncol = 3) +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako") +
  theme(legend.position = "top", plot.background = element_blank())

ggsave("plots/semantic_syntactic/oembeddings_best_match.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/semantic_syntactic/oembeddings_best_match.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)


p <- syntactic |>
  filter(task == "opposite") |>
  filter(!is.na(name.x)) |>
  filter(`Model Group` %in% c("OEmbeddings Cased", "OEmbeddings Lowercase")) |> 
  mutate(`Window Size` = as.factor(window_size),
         `Minimum Count` = as.factor(min_count)) |>
  ggplot(aes(x = `Window Size`, y = `Correct (%)`, fill = `Minimum Count`)) +
  geom_boxplot() +
  ylim(c(0, 20)) +
  facet_wrap(~`Model Group`, labeller = "label_both", ncol = 3) +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.2, end = 0.8) +
  theme(legend.position = "top", plot.background = element_blank())

ggsave("plots/semantic_syntactic/oembeddings_opposite.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/semantic_syntactic/oembeddings_opposite.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)


p <- syntactic |>
  filter(task == "doesnt fit") |>
  filter(!is.na(name.x)) |>
  filter(`Model Group` %in% c("OEmbeddings Cased", "OEmbeddings Lowercase")) |> 
  mutate(`Window Size` = as.factor(window_size),
         `Minimum Count` = as.factor(min_count)) |>
  ggplot(aes(x = `Window Size`, y = `Correct (%)`, fill = `Minimum Count`)) +
  geom_boxplot() +
  ylim(c(80, 100)) +
  facet_wrap(~`Model Group`, labeller = "label_both", ncol = 3) +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.2, end = 0.8) +
  theme(legend.position = "top", plot.background = element_blank())


p <- syntactic |>
  filter(task_group == "most_similar_groups") |>
  filter(task != "total") |> 
  filter(!is.na(name.x)) |>
  filter(`Model Group` %in% c("OEmbeddings Cased", "OEmbeddings Lowercase")) |> 
  mutate(`Window Size` = as.factor(window_size),
         `Minimum Count` = as.factor(min_count)) |>
  ggplot(aes(x = `Window Size`, y = `Correct (%)`, fill = `Minimum Count`)) +
  geom_boxplot() +
  facet_wrap(~`Model Group`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.2, end = 0.8) +
  theme(legend.position = "top", plot.background = element_blank())


ggsave("plots/semantic_syntactic/oembeddings_syntactic.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/semantic_syntactic/oembeddings_syntactic.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)



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
  ggplot(aes(x = `Window Size`, y = `Correct (Top 10, %)`, fill = `Model Group`)) +
  geom_boxplot() +
  facet_wrap(~`Task`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako") +
  theme(legend.position = "top", plot.background = element_blank())

ggsave("plots/semantic_syntactic/results_top10.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/semantic_syntactic/results_top10.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)

syntactic |> 
  filter(!grepl("OEmbeddings", `Model Group`)) |> 
  mutate(correct_percent = correct / coverage) |> 
  group_by(`name.x`, `Task`) |> 
  summarise(`Correct` = mean(correct_percent)) |> 
  tidyr::pivot_wider(names_from = `name.x`, values_from = `Correct`) |> 
  readr::write_csv("plots/offtheshelf_semantic_syntactic.csv")
