from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelForMaskedLM.from_pretrained('uklfr/gottbert-base')
model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained('uklfr/gottbert-base')

# ["Abbildung", "Abbildungen", "Arbeit", "Arbeiten"]

input_ids = tokenizer(["Abbildung Abbildungen Arbeit [MASK]"], 
                      return_tensors="pt",
                      padding="max_length",
                      truncation=True,
                      max_length=16).to(DEVICE)

output = model(**input_ids)

from transformers import pipeline
mask_filler = pipeline("fill-mask", "uklfr/gottbert-base", device=DEVICE)

text = 'Der Plural von "Abbildung" ist "Abbildungen". Der Plural von "Arbeit" ist "<mask>".'
text = "Athen und Griechenland. Paris und <mask>."


result = mask_filler(text, top_k=10)

for r in result:
    print(r['sequence'])

with open('evaluation_data/devmount/syntactic.questions') as f:
    groups = f.read().split('\n: ')

group = groups[0]
questions = group.splitlines()
label = questions.pop(0)
label = label[2:] if label.startswith(': ') else label 

question_prompts = [f'Der Plural von "{question.split()[0]}" ist "{question.split()[1]}". Der Plural von "{question.split()[2]}" ist "<mask>".' for question in questions]
answers = [question.split()[3] for question in questions]
results = mask_filler(question_prompts, top_k=10)

num_group_right = 0
num_group_topn = 0

for result, answer in zip(results, answers):
    if result[0]['token_str'] == answer:
        num_group_right += 1
    if answer in [x['token_str'] for x in result]:
        num_group_topn += 1




group = groups[1]
questions = group.splitlines()
label = questions.pop(0)
label = label[2:] if label.startswith(': ') else label 

question_prompts = [f'Der Singular von "{question.split()[0]}" ist "{question.split()[1]}". Der Singular von "{question.split()[2]}" ist "<mask>".' for question in questions]
answers = [question.split()[3] for question in questions]
results = mask_filler(question_prompts, top_k=10)

num_group_right = 0
num_group_topn = 0

for result, answer in zip(results, answers):
    if result[0]['token_str'] == answer:
        num_group_right += 1
    if answer in [x['token_str'] for x in result]:
        num_group_topn += 1


