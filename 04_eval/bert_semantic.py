from transformers import pipeline
import time
import torch
import json
from argparse import ArgumentParser

from pathlib import Path

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

arg_parser = ArgumentParser(description="Evaluate Transformer models on semantic tasks")
arg_parser.add_argument(
    "--model",
    type=str,
    default="uklfr/gottbert-base",
    help="Name / Path to the huggingface model",
)

input_args = arg_parser.parse_args()

p = Path.cwd()

TARGET_SEM_OP = p / "evaluation_data" / "devmount" / "semantic_op.questions"
TARGET_SEM_BM = p / "evaluation_data" / "devmount" / "semantic_bm.questions"

model = input_args.model
results = []

mask_filler = pipeline("fill-mask", model, device=DEVICE)
try:
    mask_token = mask_filler.tokenizer.mask_token
except:
    mask_token = "[MASK]"

# Prompt engineering: https://doi.org/10.3390/fi15070230
# https://github.com/LorenzoSerina-UniBS/Synonyms-Antonyms-GeneralKnowledge-BERT-Heads/tree/main

templates = {
    "opposite": 'Du hast es als "{}" beschrieben, aber ich würde eher das Gegenteil sagen, ich würde es als "{}" beschreiben.',
    "best match": 'Das Verhältnis zwischen "{}" und "{}" ist ähnlich wie das Verhältnis zwischen "{}" und "{}".',
}
tasks = {"opposite": TARGET_SEM_OP, "best match": TARGET_SEM_BM}

for label, dataset in tasks.items():
    with open(dataset) as f:
        questions = [l.strip() for l in f.readlines()]

    num_lines = len(questions)
    num_questions = len(questions)
    num_right = 0
    num_topn = 0
    total_duration = 0
    template = templates[label]

    texts = []
    if label == 'best match':
        for question in questions:
            texts.append(
                template.format(
                    question.split()[0],
                    question.split()[1],
                    question.split()[2],
                    mask_token,
                )
            )
    else:
        for question in questions:
            texts.append(
                template.format(
                    question.split()[2],
                    mask_token,
                )
            )
    t = time.time()
    result = mask_filler(texts, top_k=10)
    for r, question in zip(result, questions):
        if r[0]["token_str"].strip() == question.split()[3]:
            num_right += 1
        if question.split()[3] in map(lambda x: x["token_str"].strip(), r):
            num_topn += 1
    # calculate result
    correct_matches = (
        round(num_right / float(num_questions) * 100, 1) if num_questions > 0 else 0.0
    )
    topn_matches = (
        round(num_topn / float(num_questions) * 100, 1) if num_questions > 0 else 0.0
    )
    # log result
    duration = time.time() - t
    result = {
        "task_group": "most_similar_groups",
        "task": label,
        "correct": num_right,
        "top_n": num_topn,
        "n": 10,
        "coverage": num_questions,
        "total_questions": num_lines,
        "duration": duration,
    }
    results.append(result)
    # total numbers
    num_lines += num_lines
    num_questions += num_questions
    num_right += num_right
    num_topn += num_topn
    total_duration += duration


with open(
    p / "evaluation_results/" / f"{model.replace('/', '_')}_semantic.json", "w"
) as f:
    json.dump(results, f, indent=1)
