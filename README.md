# oembeddings

ÖMbeddings (Österreichische Media Embeddings)

# Configuration & Installation

- Install `requirements.txt`
- Drop raw feather files into `raw_data`
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

### Use single sentences

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
- seed: set a random seed for exporting the sentences (i.e., shuffle the dataset)  

## Training

`03_train/01_train.py`: a wrapper around the fasttext library. You can adjust any training parameter. Example usage: `python3 03_train/01_train.py cbow data/training_data.txt --window_size 10 --min_count 50 --dimensions 300 --threads 12`

- The script automatically assigns a random name to the model and stores it in the `tmp_models` directory
- Model parameters are stored in the SQL database alongside their meta information  

# Utilities

- `datamodel.py`: use SQLAlchemy to declare SQL tables
- `sql.py`: helper functions to start SQL sessions automatically
