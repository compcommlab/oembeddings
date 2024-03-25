library(RcppSimdJson)
library(dplyr)
library(stringr)
library(ggplot2)
syntactic_files <- Sys.glob("evaluation_results/semantic_syntactic/*/*.json")


j <- RcppSimdJson::fload(syntactic_files)

df_syntactic <- dplyr::bind_rows(j)

df_syntactic$model_name <- paste(df_syntactic$name, df_syntactic$parameter_string, sep = "_")

df_syntactic$top_n[is.na(df_syntactic$top_n)] <- 0

# calculate scores for tasks
df_syntactic <- df_syntactic |>
    filter(task != "total") |> 
    group_by(model_name, task) |>
    mutate(
        score_correct_strict = correct / total_questions,
        # score_correct = top_n / total_questions
        score_correct = 0
    ) |> 
    mutate(score = score_correct + score_correct_strict) |> 
    select(model_name, name, parameter_string, task, score)

j <- c()

for (task in c("twitter", "nationalrat", "autnes_automated_2019", "pressreleases", "autnes_automated_2017", "facebook", "autnes_sentiment")) {
    query_pointer <- paste0("/", task, "/metrics")
    tmp_j <- RcppSimdJson::fload(classification_files, query = query_pointer)
    j <- c(j, tmp_j)
}

df_classification <- dplyr::bind_rows(j)

df_classification <- df_classification |> filter(label == 'overall (macro)') |> 
    rename(score = f1score, name = model_name) |>
    mutate(model_name = paste(name, parameter_string, sep = '_')) |> 
    select(model_name, name, parameter_string, task, score)
    

df <- bind_rows(list(df_classification, df_syntactic))

# parse parameter string
df$min_count <- str_extract(df$parameter_string, "mincount(\\d+)_", group = 1)
df$window_size <- str_extract(df$parameter_string, "_ws(\\d+)_", group = 1)
df$lowercase <- grepl("training_data_lower", df$parameter_string)

df$min_count <- ordered(df$min_count, levels = sort(as.integer(unique(df$min_count))))
df$window_size <- ordered(df$window_size, levels = sort(as.integer(unique(df$window_size))))


# impact of lowercasing
df |> ggplot(aes(y=score, fill=lowercase)) +
    geom_boxplot() +
    facet_wrap(~task)

# impact of mincount
df |> ggplot(aes(y=score, fill=min_count)) +
    geom_boxplot() +
    facet_wrap(~task)


# impact of window size
df |> ggplot(aes(y=score, fill=window_size)) +
    geom_boxplot() +
    facet_wrap(~task)


model <- glm(score ~ min_count + window_size + lowercase, data = df)


df_condensed <- df |> group_by(model_name) |> 
    mutate(overall_score = sum(score)) |> 
    distinct(model_name, .keep_all = TRUE) |> 
    select(model_name, name, parameter_string, min_count, window_size, lowercase, overall_score) 

model <- glm(overall_score ~ min_count:lowercase + window_size:lowercase, data = df_condensed)


df |> ggplot(aes(y = score, x = min_count)) +
    geom_point()