library(arrow)
library(dplyr)
library(tidyr)

autnes2017 <- arrow::read_feather(
    "evaluation_data/classification/autnes_automated_2017.feather"
) |>
    select(where(is.numeric)) |>
    select(-id, -starts_with("_"))

autnes2017_n <- nrow(autnes2017)

autnes2017_summary <-
    autnes2017 |>
    summarise(across(where(is.numeric), sum)) |>
    t() |>
    as.data.frame() |>
    tibble::rownames_to_column("label") |>
    rename(N = V1) |>
    mutate(proportion = round(N / autnes2017_n, 2))

autnes2017_no_label <- autnes2017 |>
    mutate(row_sum = rowSums(pick(everything()))) |>
    filter(row_sum == 0) |>
    nrow()

autnes2017_summary |>
    add_row(
        label = "no label",
        N = autnes2017_no_label,
        proportion = round(autnes2017_no_label / autnes2017_n, 2)
    ) |>
    knitr::kable()


autnes2019 <- arrow::read_feather(
    "evaluation_data/classification/autnes_automated_2019.feather"
) |>
    select(where(is.numeric)) |>
    select(-id, -starts_with("_"), -topic_senti)

autnes2019_n <- nrow(autnes2019)

autnes2019_summary <-
    autnes2019 |>
    summarise(across(where(is.numeric), sum)) |>
    t() |>
    as.data.frame() |>
    tibble::rownames_to_column("label") |>
    rename(N = V1) |>
    mutate(proportion = round(N / autnes2019_n, 2))

autnes2019_no_label <- autnes2019 |>
    mutate(row_sum = rowSums(pick(everything()))) |>
    filter(row_sum == 0) |>
    nrow()

autnes2019_summary |>
    add_row(
        label = "no label",
        N = autnes2019_no_label,
        proportion = round(autnes2019_no_label / autnes2019_n, 2)
    ) |>
    knitr::kable()


datasets <- list(
    "evaluation_data/classification/autnes_sentiment.feather",
    "evaluation_data/classification/facebook.feather",
    "evaluation_data/classification/nationalrat.feather",
    "evaluation_data/classification/pressreleases.feather",
    "evaluation_data/classification/twitter.feather"
)

for (dataset in datasets) {
    print(dataset)

    df <- arrow::read_feather(
        dataset
    ) |>
        select(label)

    print(df |> group_by(label) |> count() |> knitr::kable())

    print(nrow(df))
    print("========================")
}
