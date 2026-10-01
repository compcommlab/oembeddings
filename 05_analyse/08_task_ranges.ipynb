"""
Task Outcome Ranges
====================
Prints descriptive statistics (range, mean, SD, IQR) for each task
over all models used in the red-flag analysis (same exclusions applied).
"""

import pandas as pd
import numpy as np

# ── CONFIG (keep in sync with red_flags_analysis.py) ──────────────────────────
DATA_PATH = "/Users/janabernhard-harrer/Documents/Dokumente/4_projects/2023_Embedding/Round1/analysis/Regression/dataset_regression.csv"                      # ← path to your CSV

HP_COLS  = ["lower", "mincount", "windows"]
ID_VARS  = ["group_number"] + HP_COLS

MODEL_COL        = "name"
EXCLUDE_EXACT    = [
    "wiki_de_300_wikipedia_lr0.1_epochs5_mincount1_ws5_dims300",
]
EXCLUDE_CONTAINS = ["cc_de", "wiki_de"]

TASK_LABELS = {
    "bestmatch":        "Intrinsic Task 1: Best Match",
    "opposite":         "Intrinsic Task 2: Opposite",
    "wordintrusion":    "Intrinsic Task 3: Word Intrusion",
    "mostsimilar":      "Intrinsic Task 4: Grammar",
    "ffp":              "Extrinsic Task 1: Author Prediction",
    "topics":           "Extrinsic Task 2: Topic Prediction",
    "autnes_sentiment": "Extrinsic Task 3: Sentiment Prediction",
}

INTRINSIC_TASKS = ["bestmatch", "opposite", "wordintrusion", "mostsimilar"]
EXTRINSIC_TASKS = ["ffp", "topics", "autnes_sentiment"]
ALL_TASKS       = INTRINSIC_TASKS + EXTRINSIC_TASKS

# ── LOAD & APPLY EXCLUSIONS ────────────────────────────────────────────────────
df_wide = pd.read_csv(DATA_PATH)
df_wide = df_wide.iloc[:-4].copy()

exact_mask    = df_wide[MODEL_COL].isin(EXCLUDE_EXACT)
contains_mask = df_wide[MODEL_COL].str.contains("|".join(EXCLUDE_CONTAINS), na=False)
df_wide = df_wide[~(exact_mask | contains_mask)].copy()

available_tasks = [t for t in ALL_TASKS if t in df_wide.columns]
print(f"Models included: {len(df_wide)}\n")

# ── COMPUTE RANGES ─────────────────────────────────────────────────────────────
rows = []
for task in available_tasks:
    col    = df_wide[task].dropna()
    q1, q3 = col.quantile([0.25, 0.75])
    rows.append({
        "Task":    TASK_LABELS.get(task, task),
        "N":       int(col.count()),
        "Min":     round(col.min(),  3),
        "Max":     round(col.max(),  3),
        "Range":   round(col.max()  - col.min(), 3),
        "Mean":    round(col.mean(), 3),
        "SD":      round(col.std(),  3),
        "Q1":      round(q1, 3),
        "Median":  round(col.median(), 3),
        "Q3":      round(q3, 3),
        "IQR":     round(q3 - q1, 3),
    })

results = pd.DataFrame(rows)

# ── PRINT ──────────────────────────────────────────────────────────────────────
print("=" * 90)
print("  OUTCOME RANGES PER TASK")
print("=" * 90)
print(results.to_string(index=False))

results.to_csv("task_outcome_ranges.csv", index=False)
print("\n→ Saved to task_outcome_ranges.csv")