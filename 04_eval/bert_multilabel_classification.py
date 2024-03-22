import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Supress warnings when loading a model
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
import json
import torch
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# import wandb

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="OEmbeddings"
#     )

p = Path.cwd()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# models = ['xlm-roberta-base', 'distilbert-base-multilingual-cased', 'uklfr/gottbert-base', 'microsoft/mdeberta-v3-base']


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1

    f1_micro_average = f1_score(y_pred=predictions, y_true=labels, average="micro")
    precision = precision_score(y_pred=predictions, y_true=labels, average="micro")
    recall = recall_score(y_pred=predictions, y_true=labels, average="micro")
    return {"f1": f1_micro_average, "precision": precision, "recall": recall}


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


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
        default="autnes_automated_2017",
        help="Name of the evaluation dataset",
        choices=["autnes_automated_2017", "autnes_automated_2019"],
    )

    arg_parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=2e-05,
        help="learning rate",
        dest="learning_rate",
    )
    arg_parser.add_argument(
        "-e", "--epochs", type=int, default=2, help="epochs", dest="epochs"
    )
    arg_parser.add_argument(
        "-b", "--batch_size", type=int, default=8, help="batch size", dest="batch_size"
    )

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

    # fill samples without label
    df.labels = df.labels.apply(lambda x: ["none"] if len(x) == 0 else x)

    labels = df.labels.explode().unique().tolist()
    labels.sort()
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    for l in labels:
        df[l] = df["labels"].apply(lambda x: l in x)

    df = df.drop(columns=["labels"])
    df = df.dropna().reset_index(drop=True)

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
    # "single_label_classification" or "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        input_args.model,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(DEVICE)

    def preprocess_data(examples):
        encoded = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(DEVICE)
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        labels_matrix = np.zeros((len(encoded["input_ids"]), len(labels)))
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoded["labels"] = labels_matrix.tolist()
        return encoded

    tokenized_dataset = dataset.map(
        preprocess_data, batched=True, remove_columns=dataset.column_names
    )

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
