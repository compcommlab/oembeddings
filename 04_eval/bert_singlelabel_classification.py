import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Supress warnings when loading a model
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import json
import torch
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

# import wandb

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="OEmbeddings"
#     )

p = Path.cwd()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Metrics
metric_f1 = evaluate.load("f1")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")

# models = ['xlm-roberta-base', 'distilbert-base-multilingual-cased', 'uklfr/gottbert-base', 'microsoft/mdeberta-v3-base']


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    precision = metric_precision.compute(
        predictions=predictions, references=labels, average="macro"
    )
    recall = metric_recall.compute(
        predictions=predictions, references=labels, average="macro"
    )
    results = {
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }
    return results


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="Evaluate Transformer models on a classification task"
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Debug flag: only load a random sample"
    )
    arg_parser.add_argument(
        "--seed", type=int, default=1234, help="Seed for random state (default: 1234)"
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="uklfr/gottbert-base",
        help="Name / Path to the huggingface model",
    )
    arg_parser.add_argument(
        "--dataset",
        type=str,
        default="facebook",
        help="Name of the evaluation dataset",
        choices=[
            "facebook",
            "twitter",
            "nationalrat",
            "pressreleases",
            "autnes_sentiment"
            ],
    )

    arg_parser.add_argument('-l', '--learning_rate', type=float, default=2e-05,
                        help='learning rate', dest='learning_rate')
    arg_parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='epochs', dest='epochs')
    arg_parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='batch size', dest='batch_size')

    input_args = arg_parser.parse_args()

    # Pathing
    results_path = (
        p
        / "evaluation_results"
        / f'classification_{input_args.dataset}_bert_{input_args.model.replace("/", "_")}.json'
    )
    if not results_path.parent.exists():
        results_path.parent.mkdir(parents=True)

    # Dataset
    print("Processing dataset", input_args.dataset)
    ### reformat the data so that it fits the huggingface architecture
    # load dataset
    dataset_path = (
        p / "evaluation_data" / "classification" / f"{input_args.dataset}.feather"
    )
    df = pd.read_feather(dataset_path)

    if "Neutral" in df.label.unique():
        mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
    elif "SPOE" in df.label.unique():
        mapping = {"SPOE": 0, "OEVP": 1, "FPOE": 2, "NEOS": 3, "GRUE": 4}
    df["label"] = df["label"].replace(mapping)
    # drop NAs (we don't have any but just in case)
    df = df.dropna().reset_index(drop=True).loc[:, ["text", "label"]]

    # draw sample for debugging
    if input_args.debug:
        df = df.sample(n=100)

    dataset = Dataset.from_pandas(df)

    # free up memory
    del df

    # Model Preparation
    print("Evaluating model", input_args.model)

    # tokenize the pre-processed data
    tokenizer = AutoTokenizer.from_pretrained(input_args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        input_args.model, num_labels=len(mapping.keys())
    )
    model.to(DEVICE)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(DEVICE)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split the 'train' data into training and validation sets
    train_size = int(0.8 * len(tokenized_dataset))
    validation_size = len(tokenized_dataset) - train_size

    splits = tokenized_dataset.train_test_split(
        test_size=validation_size, shuffle=True, seed=input_args.seed
    )

    training_dataset = splits["train"]
    test_dataset = splits["test"]

    training_args = TrainingArguments(
        output_dir="test_trainer",
        per_device_train_batch_size=input_args.batch_size,
        learning_rate=input_args.learning_rate,
        num_train_epochs=input_args.epochs,
        torch_compile=True,
        save_strategy="no",
        # report_to=["wandb"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    r = trainer.evaluate()
    r["dataset"] = input_args.dataset
    r["model"] = input_args.model
    print("results", r)

    # create one json per model
    with open(results_path, "w") as json_file:
        json.dump(r, json_file, indent=4)
