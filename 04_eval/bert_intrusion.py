import torch
from transformers import AutoTokenizer, AutoModel

from argparse import ArgumentParser

from pathlib import Path


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# model_name = "uklfr/gottbert-base"
# model_name = "microsoft/mdeberta-v3-base"
model_name = "FacebookAI/xlm-roberta-base"
model_name = "FacebookAI/xlm-roberta-large"
# model_name = "deepset/gbert-large"
# model_name = "deepset/gbert-base"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
model.to(DEVICE)
model.eval()

with open('evaluation_data/devmount/semantic_df.questions') as f:
    questions = [l.strip() for l in f.readlines()]

num_correct = 0

for question in questions:
    words = question.split()
    try:
        texts = [tokenizer.bos_token + t + tokenizer.eos_token for t in words]
    except:
        texts = words
    input_ids = tokenizer(texts, padding="max_length", return_tensors="pt", max_length=32).to(DEVICE)
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
        num_correct += 1

print('Num correct', num_correct)
print(round((num_correct / len(questions)) * 100, 2), '%')