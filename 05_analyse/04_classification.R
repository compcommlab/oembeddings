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


if (!dir.exists("plots/classification")) {
  dir.create("plots/classification")
}

classification <- RcppSimdJson::fload(c(
    Sys.glob("evaluation_results/oembeddings/classification/*.json"),
    Sys.glob("evaluation_results/facebook/classification/*.json")))
classification <- dplyr::bind_rows(classification)

classification$model_id <- paste(classification$model_name, classification$parameter_string, sep = "_")

classification <- left_join(classification, model_meta, by = "model_id")

bert_results <- RcppSimdJson::fload(
  Sys.glob("evaluation_results/bert_results/classification/*.json"))

bert_results <- dplyr::bind_rows(bert_results)

bert_results <- bert_results |> 
                rename(task = dataset,
                       precision = eval_precision,
                       recall = eval_recall,
                       f1score = eval_f1,
                       duration = eval_runtime,
                       model_path = model) |> 
                       select(-epoch, -eval_samples_per_second, -eval_steps_per_second, -eval_loss) |> 
                left_join(model_meta, by = "model_path") |> 
                mutate(model_id = paste(name, parameter_string, sep = "_"))

classification <- bind_rows(classification, bert_results)

classification$Task <- case_match(classification$task,
  "twitter" ~ "Party Prediction: Twitter",
  "nationalrat" ~ "Party Prediction: Parliament",
  "pressreleases" ~ "Party Prediction: Press Releases",
  "facebook" ~ "Party Prediction: Facebbok",
  "autnes_automated_2017" ~ "AUTNES Topics 2017",
  "autnes_automated_2019" ~ "AUTNES Topics 2019",
  "autnes_sentiment" ~ "AUTNES Sentiment")
  
classification$`F1 Score` <- classification$f1score

p <- classification |> 
    mutate(window_size = as.factor(window_size)) |>
    rename(
      `Window Size` = window_size,
      `Minimum Count` = min_count
    ) |>
  ggplot(aes(x = `Window Size`, y = `F1 Score`, color = `Model Group`))+
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  theme_clean() +
  scale_color_viridis(discrete = TRUE, option = "mako", begin = 0.2, end = 0.8) +
  facet_wrap(~Task, axes = "all") +
  ggtitle("Classification")

ggsave("plots/classification/classification.png", p, width = 1920, height = 1080, units = "px", scale = 2)
ggsave("plots/classification/classification.pdf", p, width = 1920, height = 1080, units = "px", scale = 1.5)
