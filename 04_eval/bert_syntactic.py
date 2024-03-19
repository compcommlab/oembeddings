from transformers import pipeline
import time
import torch
import json
from argparse import ArgumentParser

from pathlib import Path

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

arg_parser = ArgumentParser(
    description="Evaluate Transformer models on syntactic tests"
)
arg_parser.add_argument(
    "--model",
    type=str,
    default="uklfr/gottbert-base",
    help="Name / Path to the huggingface model",
)

input_args = arg_parser.parse_args()

p = Path.cwd()

TARGET_SYN = p / "evaluation_data" / "devmount" / "syntactic.questions"

model = input_args.model
results = []


mask_filler = pipeline("fill-mask", model, device=DEVICE)
try:
    mask_token = mask_filler.tokenizer.mask_token
except:
    mask_token = "[MASK]"

with open(TARGET_SYN) as f:
    groups = f.read().split("\n: ")

templates = {
    "nouns: SI/PL": 'Der Plural von "{}" ist "{}". Der Plural von "{}" ist "{}".',
    "nouns: PL/SI": 'Der Singular von "{}" ist "{}". Der Singular von "{}" ist "{}".',
    "adjectives: GR/KOM": 'Der Komparativ von "{}" ist "{}". Der Komparativ von "{}" ist "{}".',
    "adjectives: KOM/GR": 'Die Grundform von "{}" ist "{}". Die Grundform von "{}" ist "{}".',
    "adjectives: GR/SUP": 'Der Superlativ von "{}" ist "{}". Der Superlativ von "{}" ist "{}".',
    "adjectives: SUP/GR": 'Die Grundform von "{}" ist "{}". Die Grundform von "{}" ist "{}".',
    "adjectives: KOM/SUP": 'Der Superlativ von "{}" ist "{}". Der Superlativ von "{}" ist "{}".',
    "adjectives: SUP/KOM": 'Der Komparativ von "{}" ist "{}". Der Komparativ von "{}" ist "{}".',
    "verbs (pres): INF/1SP": 'Die 1. Person Singular von "{}" ist "{}". Die 1. Person Singular von "{}" ist "{}".',
    "verbs (pres): 1SP/INF": 'Der Infinitiv von "{}" ist "{}". Der Infinitiv von "{}" ist "{}".',
    "verbs (pres): INF/2PP": 'Die 2. Person Plural von "{}" ist "{}". Die 2. Person Plural von "{}" ist "{}".',
    "verbs (pres): 2PP/INF": 'Der Infinitiv von "{}" ist "{}". Der Infinitiv von "{}" ist "{}".',
    "verbs (pres): 1SP/2PP": 'Die 2. Person Plural von "{}" ist "{}". Die 2. Person Plural von "{}" ist "{}".',
    "verbs (pres): 2PP/1SP": 'Die 1. Person Singular von "{}" ist "{}". Die 1. Person Singular von "{}" ist "{}".',
    "verbs (past): INF/3SV": 'Die 3. Person Singular Vergangenheitsform von "{}" ist "{}". Die 3. Person Singular Vergangenheitsform von "{}" ist "{}".',
    "verbs (past): 3SV/INF": 'Der Infinitiv von "{}" ist "{}". Der Infinitiv von "{}" ist "{}".',
    "verbs (past): INF/3PV": 'Die 3. Person Plural Vergangenheitsform von "{}" ist "{}". Die 3. Person Plural Vergangenheitsform von "{}" ist "{}".',
    "verbs (past): 3PV/INF": 'Der Infinitiv von "{}" ist "{}". Der Infinitiv von "{}" ist "{}".',
    "verbs (past): 3SV/3PV": 'Die 3. Person Plural Vergangenheitsform von "{}" ist "{}". Die 3. Person Plural Vergangenheitsform von "{}" ist "{}".',
    "verbs (past): 3PV/3SV": 'Die 3. Person Singular Vergangenheitsform von "{}" ist "{}". Die 3. Person Singular Vergangenheitsform von "{}" ist "{}".',
}

for group in groups:
    questions = group.splitlines()
    label = questions.pop(0)
    label = label[2:] if label.startswith(": ") else label  # handle first group
    num_group_lines = len(questions)
    num_group_questions = len(questions)
    num_group_right = 0
    num_group_topn = 0
    template = templates[label]
    texts = []
    for question in questions:
        texts.append(
            template.format(
                question.split()[0],
                question.split()[1],
                question.split()[2],
                mask_token,
            )
        )
    t = time.time()
    result = mask_filler(texts, top_k=10)
    for r, question in zip(result, questions):
        if r[0]["token_str"] == question.split()[3]:
            num_group_right += 1
        if question.split()[3] in map(lambda x: x["token_str"], r):
            num_group_topn += 1
    # calculate result
    correct_group_matches = (
        round(num_group_right / float(num_group_questions) * 100, 1)
        if num_group_questions > 0
        else 0.0
    )
    topn_group_matches = (
        round(num_group_topn / float(num_group_questions) * 100, 1)
        if num_group_questions > 0
        else 0.0
    )
    group_coverage = (
        round(num_group_questions / float(num_group_lines) * 100, 1)
        if num_group_lines > 0
        else 0.0
    )
    # log result
    duration = time.time() - t
    result = {
        "task_group": "most_similar_groups",
        "task": label,
        "correct": num_group_right,
        "top_n": num_group_topn,
        "n": 10,
        "coverage": num_group_questions,
        "total_questions": num_group_lines,
        "duration": duration,
    }
    results.append(result)


with open(
    p / "evaluation_results/" / f"{model.replace('/', '_')}_syntactic.json", "w"
) as f:
    json.dump(results, f, indent=1)
