# oembeddings

ÖMbeddings (Österreichische Media Embeddings)

# Configuration & Installation

- Install `requirements.txt`
- Drop raw feather files into `raw_data`
- Copy `.env.template` to `.env`
- Set your SQL Connect string in the `.env`
- For testing, just use the default which is a sqlite database with the filename `database.db`
- Download spacy model: `python -m spacy download de_core_news_lg` (for sentence splitting)


# Scripts

## 01_dataquality

- Load raw data (news articles) from feather files into SQL database. Use the `--debug` flag to only load a small sample of the full dataset (1000 articles per feather file).
- Plot descriptive statistics of raw data

## 02_preprocessing

- split articles into sentences (uses spacy) and store them as raw sentences. Ensures each sentence is unique.
- clean sentences (`02_preprocess/02_cleansentences.py`) runs all kinds of text cleaning. Each cleaning parameter can be controlled with arguments. E.g., `--lowercase` makes all text lowercase. Use `--help` for a complete list of parameters.
    - Example usage: `python3 02_preprocess/02_cleansentences.py --remove_links --remove_emails --remove_emojis --remove_punctuation --remove_numbers --threads 8`
    - use `clean_database` to delete all previously processed sentences (deletes all rows from table `sentences`).

# Utilities

- `datamodel.py`: use SQLAlchemy to declare SQL tables
- `sql.py`: helper functions to start SQL sessions automatically
