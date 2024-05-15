import os
import sys

sys.path.append(".")

from utils.misc import get_data_dir
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
import json
from pathlib import Path
import fasttext

# Supress Fasttext warnings when loading a model
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from argparse import ArgumentParser
from time import time

p = Path.cwd()

DATA_DIR = p / "evaluation_data" / "classification"

data_partition = get_data_dir()

PROCESSED_DIR = data_partition / "evaluation_data" / "classification" / "fasttext"
VALIDATION_DIR = data_partition / "evaluation_data" / "classification" / "validation"

RESULTS_DIR = data_partition / "evaluation_results" / "classification"


def evaluate(
    model_path: str, training_file: Path, dims: int, threads: int = 12
) -> dict:
    print("Using corpus", training_file)
    test_file = training_file.name.split(".")[0] + ".feather"
    validation_corpus = pd.read_feather(VALIDATION_DIR / test_file)
    validation_corpus["text"] = (
        validation_corpus["text"]
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
    )
    labels = validation_corpus.fasttext_label.str.split().explode().unique().tolist()
    # for multi-label classification, we need to tell fasttext how
    # many labels to predict per sample (`k` parameter)
    # we take the validation sample with the highest number of labels
    max_num_labels = max(
        validation_corpus.fasttext_label.str.findall("__label_").apply(lambda x: len(x))
    )

    if max_num_labels > 1:
        # multi-label classification problem
        loss = "ova"
        threshold = 0.5
    else:
        # single-label classification problem
        loss = "softmax"
        threshold = 0.0

    mlb = MultiLabelBinarizer(classes=labels)

    if "lower" in model_path:
        validation_sentences = validation_corpus.text.str.lower().to_list()
    else:
        validation_sentences = validation_corpus.text.to_list()

    t = time()
    model = fasttext.train_supervised(
        str(training_file),
        pretrainedVectors=model_path,
        thread=threads,
        dim=dims,
        loss=loss,
    )

    predictions = model.predict(
        validation_sentences, k=max_num_labels, threshold=threshold
    )
    duration = time() - t

    # results is a tuple with len 2: first are the predicted labels
    # second are the probabilities for the labels; we only need the first one
    # get first element of each list
    if max_num_labels > 1:
        _predicted_labels = predictions[0]
        predicted_labels = mlb.fit_transform(_predicted_labels)
        _true_labels = validation_corpus.fasttext_label.str.split().to_list()
        true_labels = mlb.fit_transform(_true_labels)
    else:
        predicted_labels = list(map(lambda x: x[0], predictions[0]))
        true_labels = validation_corpus.fasttext_label.to_list()
        _predicted_labels = predicted_labels
        _true_labels = true_labels

    scores = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        #  labels=labels,
        average="macro",
    )

    results = {
        "task": training_file.name.replace(".train", "").replace("_lower", ""),
        "label": "overall (macro)",
        "precision": scores[0],
        "recall": scores[1],
        "f1score": scores[2],
        "duration": duration,
    }

    return {
        "metrics": results,
        "predicted_labels": _predicted_labels,
        "true_labels": _true_labels,
    }


if __name__ == "__main__":

    if not RESULTS_DIR.exists():
        try:
            RESULTS_DIR.mkdir(parents=True)
        except FileExistsError:
            pass

    if not PROCESSED_DIR.exists():
        raise Exception(
            "Could not find pre-processed data. Run `evaluation_data/prepare_classification_data.py` first"
        )

    arg_parser = ArgumentParser(
        description="Evaluate fasttext models on a classification task"
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Debug flag: only load a random sample"
    )
    arg_parser.add_argument(
        "--threads",
        type=int,
        default=12,
        help="Number of parallel processes (default: 12)",
    )
    arg_parser.add_argument(
        "--seed", type=int, default=1234, help="Seed for random state (default: 1234)"
    )
    arg_parser.add_argument(
        "--modelfamily",
        type=str,
        default=None,
        help="Specificy a directory of models to evaluate",
    )
    arg_parser.add_argument(
        "--glob",
        type=str,
        default="tmp_models/*/*.json",
        help="Provide a glob pattern for models to evaluate",
    )

    arg_parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Filename of corpus to evaluate. (needs to end with '.train')",
    )

    input_args = arg_parser.parse_args()

    if input_args.modelfamily:
        model_dir = Path(input_args.modelfamily)
        glob_pattern = "*.json"
    else:
        model_dir = get_data_dir()
        glob_pattern = input_args.glob

    for model_info in model_dir.glob(glob_pattern):
        model_meta = json.load(model_info.open())
        print("Evaluating:", model_meta["name"])

        model_path = model_meta["model_path"] + ".vec"
        results = {}

        if input_args.corpus:
            corpus_path = Path(input_args.corpus)
            assert (
                corpus_path.exists()
            ), f"Training corpus not found at this location: {corpus_path}"
            r = evaluate(
                model_path,
                corpus_path,
                model_meta["dimensions"],
                threads=input_args.threads,
            )

            results[corpus_path.name.split(".")[0]] = r

            results_file = (
                RESULTS_DIR
                / f"{model_meta['name']}_{model_meta['parameter_string']}_{corpus_path.name.removesuffix('.train')}.json"
            )

        else:
            if "lower" in model_meta["parameter_string"]:
                data_glob_pattern = "*.train_lower"
            else:
                data_glob_pattern = "*.train"
            results = {}
            for training_file in PROCESSED_DIR.glob(data_glob_pattern):
                r = evaluate(
                    model_path,
                    training_file,
                    model_meta["dimensions"],
                    threads=input_args.threads,
                )
                results[training_file.name.split(".")[0]] = r

            results_file = (
                RESULTS_DIR
                / f"{model_meta['name']}_{model_meta['parameter_string']}.json"
            )

        for result in results.values():
            result["metrics"]["model_name"] = model_meta["name"]
            result["metrics"]["parameter_string"] = model_meta["parameter_string"]
            result["metrics"]["model_path"] = model_meta["model_path"]

        # delete results file first
        if results_file.exists():
            print("Found existing results file, deleting it ...")
            os.remove(results_file)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=True)
