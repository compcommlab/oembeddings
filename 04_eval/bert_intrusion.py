import time
import json
import torch
from transformers import AutoTokenizer, AutoModel
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

model_name = input_args.model

p = Path.cwd()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
model.to(DEVICE)
model.eval()

with open("evaluation_data/devmount/semantic_df.questions") as f:
    questions = [l.strip() for l in f.readlines()]

num_right = 0
t = time.time()
for question in questions:
    words = question.split()
    try:
        texts = [tokenizer.bos_token + t + tokenizer.eos_token for t in words]
    except:
        texts = words
    input_ids = tokenizer(
        texts, padding="max_length", return_tensors="pt", max_length=32
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**input_ids)
    embeddings = outputs.hidden_states[-1]
    mean_vector = torch.mean(embeddings, dim=0)
    # Normalize mean vector to unit length
    mean_vector /= torch.norm(mean_vector)
    # Normalize input vectors to unit length
    embeddings_norm = embeddings / torch.norm(embeddings, dim=2, keepdim=True)
    dot_products = torch.sum(embeddings_norm * mean_vector, dim=2)
    # Identify least fitting vector
    i = dot_products.sum(1).argmin()
    if i == 3:
        num_right += 1

duration = time.time() - t

result = {
    "task_group": "word intrusion",
    "task": "doesnt fit",
    "correct": num_right,
    "coverage": len(questions),
    "total_questions": len(questions),
    "duration": duration,
}

with open(
    p / "evaluation_results/" / f"{model_name.replace('/', '_')}_intrusion.json", "w"
) as f:
    json.dump(result, f, indent=1)
