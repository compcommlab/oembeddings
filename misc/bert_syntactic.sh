#!/bin/bash

models=("xlm-roberta-base" "distilbert-base-multilingual-cased" "uklfr/gottbert-base" "microsoft/mdeberta-v3-base" "deepset/gbert-base")

for model in ${models[@]}; do
    echo "EVALUATING MODEL: ${model}"
    python3 04_eval/bert_intrusion.py --model ${model}
    python3 04_eval/bert_semantic.py --model ${model}
    python3 04_eval/bert_syntactic.py --model ${model}
done
    