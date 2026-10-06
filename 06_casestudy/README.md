# 06_casestudy: Nearest-neighbour case study

Code and data for the qualitative validation (nearest-neighbour analysis) in the paper, Figures 8 and 9 and the overlap statistics reported in the section *Qualitative Validation*.

We compare the 100 nearest neighbours of six keywords across 32 self-trained models (one per model family) and the off-the-shelf German fastText model `cc.de.300` (Grave et al., 2018):

| Keyword | Translation |
|---|---|
| Frau | woman |
| Femizid | femicide |
| Mann | man |
| Mord | murder |
| Opfer | victim |
| Täter | perpetrator |

## Pipeline

| Step | File | Needs the models? | Input | Output |
|---|---|---|---|---|
| 1 | `../05_analyse/05_casestudy_replication.ipynb` | yes (self-trained, not shared) | `models/oembeddings/*.vec` | `table_allneighbours_{Keyword}.csv` |
| 2 | `00_download_facebook_model.ipynb` | – | – | `cc.de.300.bin` (≈7 GB, not committed) |
| 3 | `01_extract_neighbours_facebook.ipynb` | yes (`cc.de.300.bin`) | `cc.de.300.bin`, `table_allneighbours_{Keyword}.csv` | cc.de.300 neighbours added to `table_allneighbours_{Keyword}.csv` |
| 4 | `02_lemmatize_overlap.ipynb` | **no** | `table_allneighbours_{Keyword}.csv`, `dict_{Keyword}.xlsx` | counts, pivot tables, heatmaps, statistics |

Step 1 is run from the repository root or from `05_analyse/`. Steps 2–4 are run from inside `06_casestudy/`.

**Reproducing the paper without the models:** the neighbour tables and the lemmatization dictionaries are included in this folder, so only step 4 is needed. Run `02_lemmatize_overlap.ipynb` once per keyword (set `keyword` in the second cell).

## Model selection

The self-trained models were trained with 32 hyperparameter combinations (casing × minimum count 5/10/50/100 × window size 5/6/12/24), each estimated ten times (= one model family). For the case study we use one model per family: the one with the highest `mean_overall` (average across all seven validation tasks) in `../evaluation_results/dataset_regression.csv`. The selected models are listed in `selected_models.csv`.

For lowercased models the lowercase keyword is used (e.g. `frau`), for cased models the cased keyword (`Frau`). All neighbours are lowercased so that cased and lowercased models can be compared.

## Manual lemmatization

Embedding models return different grammatical forms of the same word as separate neighbours (e.g. *Frau* / *Frauen*). Automated lemmatizers (spaCy, NLTK) did not work well for these words, so all neighbours were lemmatized manually:

1. `02_lemmatize_overlap.ipynb` writes `table_allneighbours_counts_{Keyword}.csv` (every neighbour and how often it occurs).
2. This table was copied to `dict_{Keyword}.xlsx` and a column `Lemma` was filled in by hand. Empty `Lemma` = the word is kept as is.
3. The notebook maps every neighbour to its lemma and computes the overlap on the lemmatized neighbours.

Columns of `dict_{Keyword}.xlsx`: `Neighbor`, `Count`, `Lemma`, `Count_Lemma`.

## Files

| File | Description |
|---|---|
| `selected_models.csv` | the 32 selected models (`model`, `name`, `group_number`, `lower`, `mincount`, `windows`, `mean_overall`) |
| `00_download_facebook_model.ipynb` | downloads `cc.de.300.bin` |
| `01_extract_neighbours_facebook.ipynb` | adds the cc.de.300 neighbours to the tables |
| `02_lemmatize_overlap.ipynb` | lemmatization, overlap statistics, heatmaps |
| `table_allneighbours_{Keyword}.csv` | 100 nearest neighbours per model (`Model`, `Neighbor`, `Similarity`), 33 models |
| `dict_{Keyword}.xlsx` | manual lemmatization dictionaries |

## Figures in the paper

| Figure | File |
|---|---|
| Figure 8: overlap for *Frau* | `heatmap_pct_overlap_Frau.pdf` |
| Figure 9: overlap for *Femizid* | `heatmap_pct_overlap_Femizid.pdf` |

The heatmaps for Mann, Mord, Opfer and Täter are not shown in the paper; their statistics are reported in the text.

## Requirements

Python packages: `gensim` (steps 1, 3), `fasttext` (step 2), `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl` (step 4).
