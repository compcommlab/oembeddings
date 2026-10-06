"""
Create dataset_regression.csv
=============================
Builds the model-level results table used by the analyses in 05_analyse/
(06_regression_effect_sizes.R, 07_mincount_robustness.R, 08_task_ranges.py,
09_red-flag-analysis.ipynb) and by the model selection of the case study
(06_casestudy/selected_models.csv).

Reads the raw evaluation results in evaluation_results/:

  oembeddings/      320 self-trained fastText models
  facebook/         pre-trained fastText model(s) (cc.de.300)
  bert_results/     4 BERT models  -> always the LAST 4 rows (08/09 drop them via iloc[:-4])

One row per model (index = model ID = <name>_<parameter_string>), one column per task.

Scores
  intrinsic   correct / coverage  (share of answerable questions solved);
              mostsimilar = "total" over the 20 grammar groups
  extrinsic   macro F1 ("f1score" for fastText, "eval_f1" for BERT)

Derived columns
  ffp              mean(twitter, facebook, pressreleases, nationalrat)
  topics           mean(autnes_automated_2017, autnes_automated_2019)
  sentiment        mean(million_posts_sentiment, autnes_sentiment) for fastText models,
                   autnes_sentiment for BERT. If million_posts_sentiment is not in the
                   raw results, autnes_sentiment only (a warning is printed).
  mean_Sem         mean(bestmatch, wordintrusion, opposite)
  mean_Synt        mostsimilar
  mean_SemSynt     mean of the four intrinsic tasks (= overall_score_v1)
  overall_score_v2 mean(ffp, topics, sentiment)
  mean_overall     mean(overall_score_v1, overall_score_v2)
  sum              sum of the 7 extrinsic tasks, ffp, topics, sentiment and 4 intrinsic tasks
  group_number     model family = identical (lower, mincount, windows), numbered 1..k in
                   sorted order; BERT models have 99 for all hyperparameters (one group)

Run from the repository root:
    python 05_analyse/00_create_dataset_regression.py

By default the result is written to evaluation_results/dataset_regression_rebuilt.csv
and compared with the published evaluation_results/dataset_regression.csv.
Set OUT_PATH = PUBLISHED_PATH below to overwrite the published file.
"""

import json
import unicodedata
from pathlib import Path
import pandas as pd

# ── PATHS ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation_results"
PUBLISHED_PATH = RES / "dataset_regression.csv"
OUT_PATH = RES / "dataset_regression_rebuilt.csv"

OWN_DIR = RES / "oembeddings"
PRETRAINED_DIR = RES / "facebook"
BERT_DIR = RES / "bert_results"

# Two different self-trained models were both named "pfiffiger_dylan".
# The second one is renamed to "pfiffiger_paul".
RENAME = {
    "pfiffiger_dylan_training_data_lower_cbow_lr0.05_epochs1_mincount5_ws6_dims300":
        "pfiffiger_paul_training_data_lower_cbow_lr0.05_epochs1_mincount5_ws6_dims300",
}

# output column -> task name inside the JSON (= sub-folder name in semantic_syntactic/)
INTRINSIC = {
    "bestmatch": "best match",
    "mostsimilar": "total",
    "opposite": "opposite",
    "wordintrusion": "doesnt fit",
}
EXTRINSIC = ["twitter", "autnes_automated_2019", "pressreleases", "facebook",
             "autnes_sentiment", "nationalrat", "autnes_automated_2017"]
MILLION_POSTS = "million_posts_sentiment"   # only used inside `sentiment`, dropped afterwards

BERT = {  # file prefix in bert_results/semantic_syntactic -> model name in the table
    "uklfr_gottbert-base": "uklfr/gottbert-base",
    "xlm-roberta-base": "xlm-roberta-base",
    "deepset_gbert-base": "deepset/gbert-base",
    "distilbert-base-multilingual-cased": "distilbert-base-multilingual-cased",
}

COLUMNS = EXTRINSIC + [
    "lower", "mincount", "windows", "dimensions", "name",
    "ffp", "topics", "sentiment",
    "bestmatch", "mostsimilar", "opposite", "wordintrusion",
    "mean_SemSynt", "mean_Sem", "mean_Synt",
    "sum", "overall_score_v1", "overall_score_v2", "mean_overall", "group_number",
]


# ── HELPERS ────────────────────────────────────────────────────────────────────
def nfc(s):
    """Model names with umlauts can be NFD-encoded (macOS file names) -> NFC."""
    return unicodedata.normalize("NFC", s)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def as_list(x):
    return x if isinstance(x, list) else [x]


def model_id(entry):
    """<name>_<parameter_string>, identical to the classification file name."""
    return nfc(f"{entry['name']}_{entry['parameter_string']}")


def score(entry):
    return entry["correct"] / entry["coverage"]


# ── 1. fastText models ─────────────────────────────────────────────────────────
def fasttext_results(base):
    rows = {}
    # extrinsic: one file per model, one entry per classification task
    for path in sorted((base / "classification").glob("*.json")):
        for e in load(path):
            if e["task"] in EXTRINSIC + [MILLION_POSTS]:
                rows.setdefault(nfc(path.stem), {})[e["task"]] = e["f1score"]
    # intrinsic: one sub-folder per task, one file per model
    for col, task in INTRINSIC.items():
        for path in sorted((base / "semantic_syntactic" / col).glob("*.json")):
            for e in as_list(load(path)):
                if e["task"] == task:
                    rows.setdefault(model_id(e), {})[col] = score(e)
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


own = fasttext_results(OWN_DIR)
pretrained = fasttext_results(PRETRAINED_DIR)
ft = pd.concat([own, pretrained]).rename(index=RENAME)

# hyperparameters and short name from the model ID
ids = ft.index.to_series()
ft["lower"] = ids.str.contains("_lower_").astype(int)
ft["mincount"] = ids.str.extract(r"mincount(\d+)", expand=False).astype(int)
ft["windows"] = ids.str.extract(r"_ws(\d+)", expand=False).astype(int)
ft["dimensions"] = ids.str.extract(r"dims(\d+)", expand=False).astype(int)
ft["name"] = ids.str.extract(r"^(.+?)_(?:training_data|\d+_[a-z]+_lr)", expand=False)
# self-trained: "aktive_celine"; pre-trained: "cc_de" / "wiki_de"

if MILLION_POSTS in ft and ft[MILLION_POSTS].notna().all():
    ft["sentiment"] = ft[[MILLION_POSTS, "autnes_sentiment"]].mean(axis=1)
else:
    print(f"WARNING: '{MILLION_POSTS}' not in the raw classification results -> "
          "sentiment = autnes_sentiment only (the published file averaged both).")
    ft["sentiment"] = ft["autnes_sentiment"]

# ── 2. BERT models ─────────────────────────────────────────────────────────────
bert_rows = {}
sem = BERT_DIR / "semantic_syntactic"
for prefix, name in BERT.items():
    r = {}
    for e in as_list(load(sem / f"{prefix}_semantic.json")):          # opposite, best match
        r[{"best match": "bestmatch", "opposite": "opposite"}[e["task"]]] = score(e)
    r["wordintrusion"] = score(load(sem / f"{prefix}_intrusion.json"))
    groups = as_list(load(sem / f"{prefix}_syntactic.json"))          # 20 grammar groups
    r["mostsimilar"] = sum(g["correct"] for g in groups) / sum(g["coverage"] for g in groups)
    for path in (BERT_DIR / "classification").glob("classification_*.json"):
        e = load(path)
        if e["model"].split("/")[-1] == name.split("/")[-1] and e["dataset"] in EXTRINSIC:
            r[e["dataset"]] = e["eval_f1"]
    bert_rows[name] = r

bert = pd.DataFrame.from_dict(bert_rows, orient="index")
bert[["lower", "mincount", "windows", "dimensions"]] = 99
bert["name"] = bert.index
bert["sentiment"] = bert["autnes_sentiment"]

# ── 3. Combine and derive aggregate scores ─────────────────────────────────────
df = pd.concat([ft, bert])    # BERT last

df["ffp"] = df[["twitter", "facebook", "pressreleases", "nationalrat"]].mean(axis=1)
df["topics"] = df[["autnes_automated_2019", "autnes_automated_2017"]].mean(axis=1)

intrinsic = ["bestmatch", "mostsimilar", "opposite", "wordintrusion"]
df["mean_SemSynt"] = df[intrinsic].mean(axis=1)
df["mean_Sem"] = df[["bestmatch", "wordintrusion", "opposite"]].mean(axis=1)
df["mean_Synt"] = df["mostsimilar"]
df["sum"] = df[EXTRINSIC + ["ffp", "topics", "sentiment"] + intrinsic].sum(axis=1)
df["overall_score_v1"] = df[intrinsic].mean(axis=1)
df["overall_score_v2"] = df[["ffp", "topics", "sentiment"]].mean(axis=1)
df["mean_overall"] = (df["overall_score_v1"] + df["overall_score_v2"]) / 2

hp = list(zip(df["lower"], df["mincount"], df["windows"]))
group_no = {g: i for i, g in enumerate(sorted(set(hp)), start=1)}
df["group_number"] = [group_no[g] for g in hp]

df = df[COLUMNS]

# ── 4. Sanity checks and save ──────────────────────────────────────────────────
assert len(own) == 320, f"expected 320 self-trained models, found {len(own)}"
assert len(bert) == 4, f"expected 4 BERT models, found {len(bert)}"
assert df.index.is_unique, "duplicate model IDs"
assert not df.isna().any().any(), df[df.isna().any(axis=1)]
assert (df.tail(4)["lower"] == 99).all(), "BERT models must be the last 4 rows"

df.to_csv(OUT_PATH)
print(f"Saved {len(df)} models ({len(own)} self-trained, {len(pretrained)} pre-trained "
      f"fastText, {len(bert)} BERT; {len(group_no)} groups) -> {OUT_PATH.relative_to(ROOT)}")

# ── 5. Compare with the published dataset_regression.csv ───────────────────────
if OUT_PATH != PUBLISHED_PATH and PUBLISHED_PATH.exists():
    pub = pd.read_csv(PUBLISHED_PATH, index_col=0)
    pub.index = pub.index.map(nfc)
    print(f"\nComparison with {PUBLISHED_PATH.relative_to(ROOT)}:")
    for label, ix in [("only in published", pub.index.difference(df.index)),
                      ("only in rebuilt", df.index.difference(pub.index))]:
        if len(ix):
            print(f"  {label}: {list(ix)}")
    common = df.index.intersection(pub.index)
    num = [c for c in COLUMNS if c != "name"]
    diff = (df.loc[common, num].astype(float) - pub.loc[common, num].astype(float)).abs() > 1e-9
    same_names = (df.loc[common, "name"] == pub.loc[common, "name"].map(nfc)).all()
    print(f"  {len(common)} common models; names identical: {same_names}")
    for col in num:
        n_diff = diff[col].sum()
        if n_diff:
            kinds = sorted({"BERT" if df.at[i, "lower"] == 99 else "fastText"
                            for i in common[diff[col].values]})
            print(f"  {col:22s} differs in {n_diff:3d} rows ({', '.join(kinds)})")
    print("  all other columns identical" if diff.any().any() else "  identical")
