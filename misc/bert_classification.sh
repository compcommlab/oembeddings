#!/bin/bash

models=("xlm-roberta-base" "distilbert-base-multilingual-cased" "uklfr/gottbert-base" "microsoft/mdeberta-v3-base" "deepset/gbert-base")

singlelabel=("autnes_sentiment" "facebook" "twitter" "nationalrat" "pressreleases")

multilabel=("autnes_automated_2017" "autnes_automated_2019")

for model in ${models[@]}; do
    echo "EVALUATING MODEL: ${model}"
    for dataset in ${singlelabel[@]}; do
        python3 04_eval/bert_singlelabel_classification.py --model ${model} --dataset ${dataset}
    done
    for dataset in ${mulitlabel[@]}; do
        python3 04_eval/bert_multilabel_classification.py --model ${model} --dataset ${dataset}
    done
        
done
    