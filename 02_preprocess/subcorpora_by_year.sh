#!/bin/bash

years=(2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022)

for i in "${years[@]}"
do 
    echo "Year: ${i}"
    python3 02_preprocess/01_cleanarticles.py --clean_database --threads 12 --before "${i}-12-21" --after "${i}-01-01" --remove_links --remove_emails --remove_emojis --remove_punctuation --replace_numbers --genderstar
    python3 02_preprocess/03_generate_training_corpus.py --corpus_name "${i}_data"
done