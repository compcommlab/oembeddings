library(RcppSimdJson)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(viridis)
library(arrow)

if (!dir.exists("plots")) {
  dir.create("plots")
}

model_meta <- RcppSimdJson::fload(Sys.glob("models/*/*.json"))
model_meta <- dplyr::bind_rows(model_meta)

# keep only models trained on HPC, Facebook models & BERT models

filt <- grepl("fs72169|cc_de_300|wiki_de_300", model_meta$model_path) | model_meta$model_type == "bert"
model_meta <- model_meta[filt, ]

model_meta$model_id <- paste(model_meta$name, model_meta$parameter_string, sep = "_")

model_meta$lowercase <- grepl("_lower", model_meta$training_corpus)
model_meta$computation_time_hours <- model_meta$computation_time / 60 / 60

# get a normalized corpus name
model_meta$training_corpus <- basename(model_meta$training_corpus)
model_meta$training_corpus <- gsub(".txt", "", model_meta$training_corpus, fixed = TRUE)
model_meta$`Training Data` <- case_match(model_meta$training_corpus,
                                        "training_data_lower" ~ "Lowercase",
                                        "training_data" ~ "Cased",
                                        .default = model_meta$training_corpus
                                        )

model_meta$`Model Group` <- case_when(model_meta$`Training Data` == "Lowercase" ~ "OEmbeddings Lowercase",
                                      model_meta$`Training Data` == "Cased" ~ "OEmbeddings Cased",
                                      grepl("cc_de_300|wiki_de_300", model_meta$model_path) ~ "Facebook Embeddings",
                                      model_meta$model_type == "bert" ~ "BERT"
                                    )

model_families <- model_meta |>
  select(`Model Group`, training_corpus, `Training Data`, model_type, min_count, dimensions, window_size, word_ngrams, epochs, learning_rate, lowercase, parameter_string) |>
  distinct()

# cache results

write_feather(model_meta, "evaluation_results/fasttext_models_meta.feather", compression = "uncompressed")
write_feather(model_families, "evaluation_results/fasttext_model_families.feather", compression = "uncompressed")

# Computation Time
p <- model_meta |>
  filter(!is.na(window_size)) |> 
  mutate(window_size = as.factor(window_size)) |>
  rename(
    `Window Size` = window_size,
    `Computation Time (hours)` = computation_time_hours,
    `Minimum Count` = min_count
  ) |>
  ggplot(aes(x = `Window Size`, y = `Computation Time (hours)`, fill = `Training Data`)) +
  geom_boxplot() +
  facet_wrap(~`Minimum Count`, labeller = "label_both") +
  theme_clean() +
  scale_fill_viridis(discrete = TRUE, option = "mako", begin = 0.4, end = 0.6) +
  ggtitle("Training Duration")

ggsave("plots/training_duration.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/training_duration.pdf", p, width = 1920, height = 1080, units = "px", scale = 2)
