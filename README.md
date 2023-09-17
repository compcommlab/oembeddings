![ömbeddings](misc/oembeddings-white.svg)

ÖMbeddings (Österreichische Media Embeddings)

# Configuration & Installation

- Install `requirements.txt`
- Drop raw feather files for training data into `raw_data`
- Drop evaluation data files in `evaluation_data/classification`
- Copy `.env.template` to `.env`
- Set your SQL Connect string in the `.env`
    - For testing, just use the default which is a sqlite database with the filename `database.db`
- Install latest version of [fasttext](https://github.com/facebookresearch/fastText/) (see their documetnation)
- *Important*: Set the path to the `fasttext` binary in your `.env` file
- *Optional*: Install spaCy for sentence splitting. Download spacy model: `python -m spacy download de_core_news_lg`


# Scripts

## 01_dataquality

- Load raw data (news articles) from feather files into SQL database. Use the `--debug` flag to only load a small sample of the full dataset (1000 articles per feather file).
- Plot descriptive statistics of raw data

## 02_preprocessing

There are different ways to segment the corpus into training units for fasttext:

- ✅ Retain whole articles / paragraphs (minimal segmentation; this is the default approach for fasttext)
- 🧪 Split articles into sentences (smaller training units)

### General Notes for Text Cleaning

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


### Retain Whole Articles

`02_preprocess/01_cleanarticles.py`: take a whole article, clean it and add each paragraph as separate row to the DB (table `processed_articles`). 

- Treats headlines as paragraphs. 
- Ensures there are no duplicates with md5 sum.
- Paragraph splitting by double line break characters (`\n\n`)
- Recommended settings: `python3 02_preprocess/01_cleanarticles.py --remove_links --remove_emails --remove_emojis --remove_punctuation --replace_numbers --genderstar --threads 12`
- Parameters are documented, use `python3 02_preprocess/01_cleanarticles.py --help` to get a description of each parameter.

### Use single sentences (unused)

`02_preprocess/x_01_splitsentences.py`: split articles into sentences (uses spacy) and store them as raw sentences. Ensures each sentence is unique.

`02_preprocess/x_02_cleansentences.py`: clean every sentence; runs all kinds of text cleaning. Each cleaning parameter can be controlled with arguments. E.g., `--lowercase` makes all text lowercase. Use `--help` for a complete list of parameters.

- Example usage: `python3 02_preprocess/02_cleansentences.py --remove_links --remove_emails --remove_emojis --remove_punctuation --remove_numbers --threads 8`
- use `clean_database` to delete all previously processed sentences (deletes all rows from table `sentences`).

### Generate Training Corpus

`02_preprocess/03_generate_training_corpus.py`: dumps text to single `.txt` file (preferred format for fasttext). One line per training unit. You can specify these options:

- debug: only use a random sample
- min_length: only use sentence with a minimum number of tokens (default: 5 tokens)
- corpus_name: file name for `txt` file. Training corpora files are always located in the `data` directory (is created automatically)
- lowercase: apply lowercasing to corpus
    - It is recommended to include `lower` in the file name of the training corpus. This way, the evaluation scripts can infer whether the model is lowercased or not.
- seed: set a random seed for exporting the sentences (i.e., shuffle the dataset)  

## Training

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

## Evaluation

- All results are stored in the directory `evaluation_results`. 
- The subdirectories correspond to each specific task.
- Results are stored as JSON files, can later be read with pandas for running statistical analysis

### Cosine Similarity

The scripts `01_eval_cosine_across.py` and `01_eval_cosine_within.py` evaluate the stability of the models. For each model, they calculate the cosine distance of cue words against the every word in the entire vocabulary of the model. Then they compare the cosine distances pairwise with other models

For example, there is Model A and Model B. First we take the cue word "Politik" and calculate its cosine similarity to all other words of the Model A's vocabulary. Next, we do the same for Model B. So we get two sets of cosine similarity metrics. Finally, we measure the correlation between both sets of cosine similarities. If the correlation is high, it means the models are stable, and not subject to randomness. If the correlation is low, it means that random factors strongly influence the models.

The two scripts scan the `tmp_models` directory and automatically make the pairwise comparisons based on the directory structure and also the JSON metadata files.

Model pairs where one model is lowercased and the other is not are not compared!

### Semantic & Syntactic Tasks

Data files in the directory `evaluation_data/devmount` were taken from project [GermanWordEmbeddings](https://github.com/devmount/GermanWordEmbeddings), Copyright (c) 2015 Andreas Müller. These files are licensed under the MIT license. See DEVMOUNT-LICENSE.md for additional details.

The script was enhanced to automatically scan the `tmp_models` directory for all models and to evaluate them one by one. The results are stored in `evaluation_results/semantic_syntactic` with subdirectories for each sub-task.

The evaluation also handles models with lowercased training data automatically (as long as `lower` is in the training data file name)

### Classification

Datasets for classification tasks cannot be shared because a) file size is too large, and b) copyright issues (e.g., press releases by parties).

- Drop all feather files in the directory `evaluation_data/classification`
- Run the script `04_eval/classification.py`
    - automatically generates data format required for fasttext
    - evaluates all models one by one
    - stores results in `evaluation_results/classification`

## Utilities

- `datamodel.py`: use SQLAlchemy to declare SQL tables
- `sql.py`: helper functions to start SQL sessions automatically
