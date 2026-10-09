# ÖMbeddings (Österreichische Media Embeddings)

# Overview

This repository contains the code used for the study *When Validation Disagrees: How Validation Choices Shape Word Embedding Selection in Computational Communication Science* published in *Computational Communication Research*. 

We trained embedding models on a large corpus of Austrian online news texts. We used the *fastText* software library to train our embeddings (Bojanowski et al, 2017), instead of the often-used word2vec (Mikolov et al, 2013) or GloVe (Pennington et al, 2014). Architecturally, *fastText* is a direct extension of word2vec, utilizing identical underlying optimization objectives and negative sampling mechanics. However, *fastText* extends this logic by learning representations at the sub-word level. This sub-word tokenization is a morphological necessity for our corpus, as the German language relies heavily on complex compound words (Rodriguez et al, 2023). While traditional word-level architectures like word2vec or GloVe treat these compounds as isolated, unique tokens (leading to severe data sparsity for less frequent terms) *fastText* effectively captures the shared semantic roots of compound structures.

## Corpus

The training data for our models consists of N = 5,495,185 online news articles from scraped from nine Austrian media outlets, collected between 2010 and 2022.

*The original data cannot be shared due to copyright restrictions. However, we include a small wikipedia corpus for testing purposes.*

| News Outlet          | 2010  | 2011  | 2012  | 2013  | 2014  | 2015  | 2016  | 2017  | 2018   | 2019   | 2020   | 2021  | 2022  | 2023 |
|----------------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|--------|-------|-------|------|
| www.derstandard.at   | 0     | 0     | 80501 | 73076 | 60819 | 66268 | 67408 | 66939 | 60793  | 52344  | 45287  | 43214 | 46321 | 0    |
| www.diepresse.com    | 54642 | 54728 | 56109 | 58897 | 59438 | 56904 | 55353 | 48397 | 49740  | 44502  | 37829  | 39064 | 36577 | 0    |
| www.gmx.at           | 0     | 27    | 18    | 12    | 4608  | 2743  | 6801  | 9803  | 11873  | 14918  | 15627  | 15381 | 20562 | 0    |
| www.heute.at         | 0     | 0     | 1     | 0     | 2     | 2     | 1     | 73    | 225    | 20754  | 42971  | 45484 | 49875 | 0    |
| www.kleinezeitung.at | 65038 | 71598 | 74287 | 80890 | 78039 | 86606 | 75104 | 94853 | 104817 | 107699 | 107451 | 91881 | 83869 | 0    |
| www.krone.at         | 1     | 0     | 23621 | 25824 | 28081 | 32140 | 34496 | 42106 | 51396  | 72433  | 77127  | 88570 | 85580 | 0    |
| www.kurier.at        | 3034  | 23094 | 41181 | 43151 | 44646 | 43713 | 46280 | 43298 | 43594  | 49835  | 53213  | 54624 | 52089 | 0    |
| www.oe24.at          | 48708 | 47347 | 48540 | 55186 | 55903 | 41772 | 41506 | 36117 | 32311  | 34091  | 50903  | 51223 | 37846 | 54   |
| www.orf.at           | 18139 | 44921 | 50945 | 51167 | 52737 | 54578 | 57312 | 54773 | 52333  | 48219  | 51451  | 49989 | 47695 | 0    |
| www.sn.at            | 0     | 25    | 31955 | 35486 | 35655 | 39162 | 41311 | 48851 | 47027  | 53129  | 50091  | 47403 | 47153 | 1    |

We ran the following preprocessing steps:

- remove hyperlinks 
- remove email addresses 
- remove emojis 
- remove punctuation
- replace numbers with words (e.g., "7" -> "sieben")
- normalize genderstar to common format
- remove all non-latin characters (e.g., Greek, Hebrew, etc)

## Computation Time & Carbon Footprint

Training all models took about 1335 hours (wall clock) of computation time (mean = 4.17 hours, see also "plots/training_duration.png") on the Vienna Scientific Cluster (VSC, see: https://asc.ac.at/systems/vsc-5/). The training ran on VSC-5 where each node is equipped with with two AMD EPYC Milan 7713 processors (64 cores per CPU) with a base frequency of 2GHz and have 512GB of memory. Each training run was performed on one node, submitted to the VSC queue. According to estimations of [Green Algorithms calculator](https://calculator.green-algorithms.org/) the training emitted 107.68 kgCO2e and needed 866.10 kWh.

Analogously we performed the evaluation of all models on the VSC as well:

- Syntactic / Semantic Evaluation: total 57.4 hours (wall clock, mean = 26 seconds); Carbon footprint 4.64 kgCO2e; Energy needed 37.29 kWh
- Classification tasks: total 53.6 hours (wall clock, mean = 85 seconds); Carbon footprint, 4.31 kgCO2e; Energy needed 34.7 kWh

Unfortunately, exact timings for correlation calculation runs were not recorded.

## Evaluation Data

Data for syntactic and semantic evaluation was taken from project [GermanWordEmbeddings](https://github.com/devmount/GermanWordEmbeddings), Copyright (c) 2015 Andreas Müller. These files are licensed under the MIT license. See DEVMOUNT-LICENSE.md for additional details. Redistribution permitted by MIT license.

Data for classification tasks cannot be shared publicly due to copyright issues. Data is part of the AUTNES project where media coverage as well as party messages are collected and manually coded. For access to the data please contact us.

"One Million Posts" Corpus (`evaluation_data/classification/million_posts_sentiment.feather`) by Schabus et al (2017) available at: [https://ofai.github.io/million-post-corpus/](https://ofai.github.io/million-post-corpus/). Redistributed granted by Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (see: `MillionPosts-LICENSE.md`).

For more details refer to `evaluation_data/README.md`

## References

- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. Transactions of the Association for Computational Linguistics, 5, 135–146. https://doi.org/10.1162/tacl_a_00051
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013, September). Efficient Estimation of Word Representations in Vector Space. https://doi.org/10.48550/arXiv.1301.3781
- Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532–1543. https://doi.org/10.3115/v1/D14-1162
- Rodriguez, P. L., Spirling, A., & Stewart, B. M. (2023). Embedding Regression: Models for Context-Specific Description and Inference. American Political Science Review, 117(4), 1255–1274. https://doi.org/10.1017/S0003055422001228
- Dietmar Schabus, Marcin Skowron, Martin Trapp. One Million Posts: A Data Set of German Online Discussions. Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), pp. 1241-1244. Tokyo, Japan, August 2017. DOI: [10.1145/3077136.3080711](https://doi.org/10.1145/3077136.3080711)
- Dietmar Schabus and Marcin Skowron. Academic-Industrial Perspective on the Development and Deployment of a Moderation System for a Newspaper Website. Proceedings of the 11th International Conference on Language Resources and Evaluation (LREC 2018), pp. 1602-1605. Miyazaki, Japan, May 2018. [Full paper available for download from LREC](http://www.lrec-conf.org/proceedings/lrec2018/summaries/8885.html)

# Replication

## Configuration & Installation

- Install `requirements.txt`
- Drop raw feather files for training data into `raw_data`
    - we provide a sample dataset of wikipedia articles for demonstration purposes (`raw_data/wikipedia.sample.xz`, stored in IPC feather format)
    - You need to decompress the wikipedia sample data before loading it (using this command on linux: `xz -d -k raw_data/wikipedia.sample.xz`)
- Drop evaluation data files in `evaluation_data/classification`
    - we cannot share the evaluation data derived from the AUTNES studies due to copyright 
- Copy `.env.template` to `.env`
- Set your SQL Connect string in the `.env`
    - For testing, just use the default which is a sqlite database with the filename `database.db`
    - We recommend using PostgreSQL, because it can easily handle concurrent connections
- Install latest version of [fasttext](https://github.com/facebookresearch/fastText/) (see their documetnation)
- *Important*: Set the path to the `fasttext` binary in your `.env` file
- *Optional*: Install spaCy for sentence splitting. Download spacy model: `python -m spacy download de_core_news_lg`


## Scripts

### 01_dataquality

- Load raw data (news articles) from feather files into SQL database. Use the `--debug` flag to only load a small sample of the full dataset (1000 articles per feather file). Use the `--wikipedia` flag to only load the wikipedia sample (skipped by default)
- Plot descriptive statistics of raw data

### 02_preprocessing

There are different ways to segment the corpus into training units for fasttext:

- ✅ Retain whole articles / paragraphs (minimal segmentation; this is the default approach for fasttext)
- 🧪 Split articles into sentences (smaller training units)

#### General Notes for Text Cleaning

All text cleaning is handled by the function `clean_text()` (`utils/cleaning.py`). The module also contains all regular expressions as well as simple tests.

- hyphenated terms where the first component is longer than one character get separated: 
    - "Ex-ÖVP-Chef" -> "Ex ÖVP Chef"
    - "Pamela Rendi-Wagner" -> "Pamela Rendi Wagner"
- but preserves:
    - "E-Mail" -> "E-Mail"
    - "E-Mobilität" -> "E-Mobilität"
    - "E-Auto-Boom" -> "E-Auto Boom"
- Genderstar ("Gendersternchen") are normalized and preserved:
    - "Patient*innen" -> "Patient_innen"
    - "Rentner:innen" -> "Rentner_innen"
    - "LehrerInnen" -> "Lehrer_innen"
- All non-Latin scripts are removed by default:
    - Hebrew
    - Arabic
    - Cyrillic
    - Chinese (traditional and simplified)
- Unicode symbols are removed by default (e.g., `≈ ≠ ≤ ≥ Ⓒ © − ☆`)

- If numbers are removed then fixed compounds with numbers get truncated
    - "G7-Gipfel" -> "G Gipfel"
    - "Formel-1" -> "Formel"
    - "F1" -> "F"
- Adjustable via `replace_numbers` (this is the recommended setting):
    - "G7-Gipfel" -> "G sieben Gipfel"
    - "Formel-1" -> "Formel eins"
    - "F1" -> "F eins"


#### Retain Whole Articles

`02_preprocess/01_cleanarticles.py`: take a whole article, clean it and add each paragraph as separate row to the DB (table `processed_articles`). 

- Treats headlines as paragraphs. 
- Ensures there are no duplicates with md5 sum.
- Paragraph splitting by double line break characters (`\n\n`)
- Recommended settings (used in the published manuscript): `python3 02_preprocess/01_cleanarticles.py --remove_links --remove_emails --remove_emojis --remove_punctuation --replace_numbers --genderstar --threads 12`
- Parameters are documented, use `python3 02_preprocess/01_cleanarticles.py --help` to get a description of each parameter.

#### Use single sentences (unused / experimental)

`02_preprocess/x_01_splitsentences.py`: split articles into sentences (uses spacy) and store them as raw sentences. Ensures each sentence is unique.

`02_preprocess/x_02_cleansentences.py`: clean every sentence; runs all kinds of text cleaning. Each cleaning parameter can be controlled with arguments. E.g., `--lowercase` makes all text lowercase. Use `--help` for a complete list of parameters.

- Example usage: `python3 02_preprocess/02_cleansentences.py --remove_links --remove_emails --remove_emojis --remove_punctuation --remove_numbers --threads 8`
- use `clean_database` to delete all previously processed sentences (deletes all rows from table `sentences`).

#### Generate Training Corpus

`02_preprocess/03_generate_training_corpus.py`: dumps text to single `.txt` file (preferred format for fasttext). One line per training unit. You can specify these options:

- debug: only use a random sample
- min_length: only use sentence with a minimum number of tokens (default: 5 tokens)
- corpus_name: file name for `txt` file. Training corpora files are always located in the `data` directory (is created automatically)
- lowercase: apply lowercasing to corpus
    - It is recommended to include `lower` in the file name of the training corpus. This way, the evaluation scripts can infer whether the model is lowercased or not.
- seed: set a random seed for exporting the sentences (i.e., shuffle the dataset)  

### Training

`03_train/01_train.py`: a wrapper around the fasttext library. You can adjust any training parameter. Example usage: `python3 03_train/01_train.py cbow data/training_data.txt --window_size 10 --min_count 50 --dimensions 300 --threads 12`

- The script automatically assigns a random name to the model and stores it in the `tmp_models` directory
- It creates a subdirectory based on the model parameters.
    - for the example above, it will create the directory `tmp_models/training_data_cbow_lr0.1_epochs5_mincount50_dims300` and store the model in this path.
    - With this we conveniently sort all model parameter families in seaparate directories 
- Model parameters are stored in JSON files alongside their meta information
    - the JSON file has the same name as the model, and is stored in the same directory
- The training should result in three files:
    - Model as `.bin`
    - Model as `.vec`
    - Model meta as `.json` 

### Evaluation

- All results are stored in the directory `evaluation_results`. 
- The subdirectories correspond to each specific task.
- Results are stored as JSON files, can later be read with pandas for running statistical analysis

#### Cosine Similarity

The scripts `01_eval_cosine_across.py` and `01_eval_cosine_within.py` evaluate the stability of the models. For each model, they calculate the cosine distance of cue words against the every word in the entire vocabulary of the model. Then they compare the cosine distances pairwise with other models

For example, there is Model A and Model B. First we take the cue word "Politik" and calculate its cosine similarity to all other words of the Model A's vocabulary. Next, we do the same for Model B. So we get two sets of cosine similarity metrics. Finally, we measure the correlation between both sets of cosine similarities. If the correlation is high, it means the models are stable, and not subject to randomness. If the correlation is low, it means that random factors strongly influence the models.

The two scripts scan the `tmp_models` directory and automatically make the pairwise comparisons based on the directory structure and also the JSON metadata files.

Model pairs where one model is lowercased and the other is not are not compared!

#### Semantic & Syntactic Tasks

Data files in the directory `evaluation_data/devmount` were taken from project [GermanWordEmbeddings](https://github.com/devmount/GermanWordEmbeddings), Copyright (c) 2015 Andreas Müller. These files are licensed under the MIT license. See DEVMOUNT-LICENSE.md for additional details.

The script was enhanced to automatically scan the `tmp_models` directory for all models and to evaluate them one by one. The results are stored in `evaluation_results/semantic_syntactic` with subdirectories for each sub-task.

The evaluation also handles models with lowercased training data automatically (as long as `lower` is in the training data file name)

#### Classification

Datasets for classification tasks cannot be shared because a) file size is too large, and b) copyright issues (e.g., press releases by parties).

- Drop all feather files in the directory `evaluation_data/classification`
- Run the script `04_eval/classification.py`
    - automatically generates data format required for fasttext
    - evaluates all models one by one
    - stores results in `evaluation_results/classification`
    
## Analysis and Reproduction of the Paper

The raw evaluation results of all models are included in `evaluation_results/` (JSON files, one per model and task). All tables and figures in the paper can be reproduced from these files without the training corpus or the trained models. The only exceptions are Figure 1 / Table 2 (need the corpus database) and step 1 of the case study (needs the self-trained models); their outputs are included in the repository.

### Requirements

- **R** (scripts in `05_analyse/*.R`): `tidyverse`, `dplyr`, `stringr`, `ggplot2`, `ggthemes`, `viridis`, `forcats`, `cowplot`, `patchwork`, `RcppSimdJson`, `arrow`, `brms`, `bayestestR`, `corrplot`, `data.table`, `reshape2`, `psych`, `car`, `openxlsx`, `Matrix`. `brms` needs a working Stan installation (`rstan` or `cmdstanr`). Package versions: see *R session info* below.
- **Python** (`*.py`, `*.ipynb`): see `requirements.txt`.

### Working directory

Unless stated otherwise, run all scripts **from the repository root**, e.g. `Rscript 05_analyse/06_regression_effect_sizes.R` or `python 05_analyse/08_task_ranges.py`. Exceptions: `05_analyse/09_red-flag-analysis.ipynb` is run from inside `05_analyse/`, and the case-study notebooks 00–02 from inside `06_casestudy/`.

### Scripts in `05_analyse/`

Run in this order (scripts 02–04 need the output of 01; 06–09 need the output of 00).

| Script | What it does | Input | Output |
|---|---|---|---|
| `00_create_dataset_regression.py` | Builds the model-level results table (one row per model, one column per task) from the raw results; all analyses (06–09) use this file | `evaluation_results/{oembeddings,facebook,bert_results}/` | `evaluation_results/dataset_regression_rebuilt.csv` (compared with the original `dataset_regression.csv`, see *Known differences*) |
| `01_model_meta.R` | Collects model metadata (hyperparameters, model families, training time) | `models/*/*.json` | `evaluation_results/fasttext_models_meta.feather`, `evaluation_results/fasttext_model_families.feather`, `plots/training_duration.pdf` |
| `02_correlations.R` | Stability: within- and across-family correlations of cue-word similarities | `evaluation_results/*/within_correlations/`, `evaluation_results/*/across_correlations/` | `plots/within_correlation/`, `plots/across_correlation/` |
| `03_syntactic_semantic.R` | Intrinsic tasks (Best Match, Opposite, Word Intrusion, Grammar) and vocabulary coverage | `evaluation_results/*/semantic_syntactic/` | `plots/semantic_syntactic/`, `plots/offtheshelf_semantic_syntactic.csv` |
| `04_classification.R` | Extrinsic tasks (author, topic and sentiment prediction) | `evaluation_results/*/classification/` | `plots/classification/`, `plots/offtheshelf_classification.csv` |
| `05_casestudy_replication.ipynb` | Case study step 1: 100 nearest neighbours per keyword in the 32 selected self-trained models (**needs the models, not shared**) | `models/oembeddings/`, `06_casestudy/selected_models.csv` | `06_casestudy/table_allneighbours_{Keyword}.csv` |
| `06_regression_effect_sizes.R` | Bayesian multilevel regressions (320 self-trained models), ROPE, probability of direction, effect sizes, rank correlations between tasks | `evaluation_results/dataset_regression_rebuilt.csv` | `plots/regression/table4_posterior.csv`, `effect_size_summary.csv`, `plot_intrinsic_tasks.pdf`, `plot_extrinsic_tasks.pdf`, `rank_correlations.csv`, `fits_list.rds` (not committed) |
| `07_mincount_robustness.R` | Robustness check for minimum count: interaction with task type and per-task models | `evaluation_results/dataset_regression_rebuilt.csv` | `plots/regression/figure_combined_mincount_analysis.pdf` |
| `08_task_ranges.py` | Observed performance range per task (basis of the ROPE: ±10% of the range) | `evaluation_results/dataset_regression_rebuilt.csv` | `plots/regression/task_outcome_ranges.csv` |
| `09_red-flag-analysis.ipynb` | Red-flag analysis (consistently underperforming hyperparameter values; Kruskal–Wallis and Mann–Whitney tests), once with all tasks and once without Grammar | `evaluation_results/dataset_regression_rebuilt.csv` | `plots/red_flags/all_tasks/`, `plots/red_flags/without_grammar/` |

The regressions in 06 and 07 use the 320 self-trained models only (model families 2–33); the scripts check this with `stopifnot()`. Predictors are z-scored, outcomes are in raw units (accuracy or F1). Model fits use `seed = 42`.

### Case study (`06_casestudy/`)

See `06_casestudy/README.md` for the full pipeline. To reproduce Figures 8 and 9 and the overlap statistics without the models, run `06_casestudy/02_lemmatize_overlap.ipynb` once per keyword (Frau, Femizid, Mann, Mord, Opfer, Täter); the neighbour tables and the manual lemmatization dictionaries are included.

### Reproducing the tables and figures

| Paper | Script | Output file |
|---|---|---|
| Figure 1: Sources in the training data | `01_dataquality/02_descriptives.ipynb` (needs the corpus database) | `01_dataquality/article_descriptives.pdf` |
| Table 2: Articles per year and outlet | `01_dataquality/02_descriptives.ipynb` (needs the corpus database) | `01_dataquality/article_descriptives.csv` |
| Table 3: Cue words | – (defined in `evaluation_data/cues.py`) | – |
| Figure 2: Within-family correlations | `05_analyse/02_correlations.R` | `plots/within_correlation/within_correlation_cues.pdf` |
| Figure 3: Across-family correlations | `05_analyse/02_correlations.R` | `plots/across_correlation/across_correlation_variation.pdf` |
| Figure 5: Intrinsic tasks | `05_analyse/03_syntactic_semantic.R` | `plots/semantic_syntactic/big_plot.pdf` |
| Figure 10: Vocabulary coverage | `05_analyse/03_syntactic_semantic.R` | `plots/semantic_syntactic/coverage.pdf` |
| Figure 7: Extrinsic tasks | `05_analyse/04_classification.R` | `plots/classification/classification.pdf` |
| Table 5: Off-the-shelf models | `05_analyse/03_syntactic_semantic.R` (intrinsic rows), `05_analyse/04_classification.R` (extrinsic rows) | `plots/offtheshelf_semantic_syntactic.csv`, `plots/offtheshelf_classification.csv` |
| Table 4: Posterior distributions | `05_analyse/06_regression_effect_sizes.R` | `plots/regression/table4_posterior.csv` |
| Figure 4: Coefficients, intrinsic tasks | `05_analyse/06_regression_effect_sizes.R` | `plots/regression/plot_intrinsic_tasks.pdf` |
| Figure 6: Coefficients, extrinsic tasks | `05_analyse/06_regression_effect_sizes.R` | `plots/regression/plot_extrinsic_tasks.pdf` |
| ROPE, pd and effect sizes reported in *Results* | `05_analyse/06_regression_effect_sizes.R` | `plots/regression/effect_size_summary.csv` |
| Task ranges and ROPE half-widths | `05_analyse/08_task_ranges.py` | `plots/regression/task_outcome_ranges.csv` |
| Rank correlations between tasks (*Results*, *Discussion*) | `05_analyse/06_regression_effect_sizes.R` | `plots/regression/rank_correlations.csv` |
| Red-flag analysis (*Results*) | `05_analyse/09_red-flag-analysis.ipynb` | `plots/red_flags/all_tasks/`, robustness check: `plots/red_flags/without_grammar/` |
| Figure 8: Nearest-neighbour overlap, "Frau" | `06_casestudy/02_lemmatize_overlap.ipynb` (keyword = Frau) | `06_casestudy/heatmap_pct_overlap_Frau.pdf` |
| Figure 9: Nearest-neighbour overlap, "Femizid" | `06_casestudy/02_lemmatize_overlap.ipynb` (keyword = Femizid) | `06_casestudy/heatmap_pct_overlap_Femizid.pdf` |
| Overlap statistics for Mann, Mord, Opfer, Täter (*Qualitative Validation*) | `06_casestudy/02_lemmatize_overlap.ipynb` | printed in the notebook; `06_casestudy/heatmap_pct_overlap_{Keyword}.pdf` |
| Figure 11: Robustness of the minimum-count effect (Appendix) | `05_analyse/07_mincount_robustness.R` | `plots/regression/figure_combined_mincount_analysis.pdf` |
| Computation time (this README) | `05_analyse/01_model_meta.R` | `plots/training_duration.pdf` |

### Known differences between `dataset_regression_rebuilt.csv` and `dataset_regression.csv`

All analyses (06–09) use `dataset_regression_rebuilt.csv`, which `00_create_dataset_regression.py` builds from the raw results in this repository. The original `dataset_regression.csv` is kept because the case-study model selection (`06_casestudy/selected_models.csv`) is based on it. For the 320 self-trained models, both files contain the same task scores and hyperparameters. They differ in three respects: (1) the raw One Million Posts sentiment results are not included in the repository, so `sentiment`, `sum`, `overall_score_v2` and `mean_overall` (the case-study selection score) cannot be rebuilt; (2) the wiki.de fastText model is only in the original file, as it has no raw results in the repository; (3) the BERT topic scores differ in four rows. None of these columns or rows enter the analyses in 06–09.

### R session info

<!-- TODO: paste the output of sessionInfo() after running 06 and 07 -->


### Utilities

- `get_third_party_embeddings.py`: automatically downloads fastText pre-trained models (German)
- `datamodel.py`: use SQLAlchemy to declare SQL tables
- `sql.py`: helper functions to start SQL sessions automatically
