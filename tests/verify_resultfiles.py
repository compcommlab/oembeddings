# Script to check whether all models were evaluated

import json
from pathlib import Path
import sys
sys.path.append('.')

from evaluation_data.cues import CUES

p = Path.cwd() / 'evaluation_results'

print('Verifying all tasks are completed')

# Hyperparamters

min_count_values = [5, 10, 50, 100]
window_size_values = [5, 6, 12, 24]
lowercasing_values = ['no', 'yes']
training_runs = 10

total_family_count = len(min_count_values) * len(window_size_values) * len(lowercasing_values)
total_model_count = total_family_count * training_runs

print('Expected number of model families:', total_family_count)
print('Expected number of models:', total_model_count)

# Within Correlations
print('Verifying Within correlations...')

# pairwise combinations, for each CUE (plus one for random cues)
possible_combinations = (training_runs * (training_runs - 1) / 2) * (len(CUES) + 1)
possible_combinations = int(possible_combinations)

families = set()

for file in p.glob('within_correlations/*.json'):
    with open(file) as f:
        j = json.load(f)
    assert len(j) == possible_combinations, f"Problem in within correlations. Number of pairs not correct: {file}"
    family = {val['parameter_string'] for val in j}
    assert len(family) == 1, f"Problem in Within correlations. Paramter strings inconsistent: {file}"
    families = families | family

assert len(families) == total_family_count, f"Problem in within correlations. Number of families not correct: {families}"

print('Passed!')

# Across correlations
print('Verifying Across correlations...')

families_a = set()
families_b = set()
combinations = set()

_no_lowercase_combinations = len(min_count_values) * len(window_size_values)
no_lowercase_combinations = int(_no_lowercase_combinations * (_no_lowercase_combinations - 1) / 2)

_no_cased_combinations = _no_lowercase_combinations + 1 # plus fastText commoncrawl model
no_cased_combinations = int(_no_cased_combinations * (_no_cased_combinations - 1) / 2)

files = p.glob('across_correlations/*.json')
file = list(files)[0]

with open(p / 'across_correlations/results.json') as f:
    j = json.load(f)

for file in p.glob('across_correlations/*.json'):
    with open(file) as f:
        j = json.load(f)
    assert len(j) == (len(CUES) + 1), f"Problem in across correlations. Number of cues not correct: {file}"
    combination = {val['model_a_family'] + '_X_' + val['model_b_family'] for val in j}
    assert len(combination) == 1, f"Problem in across correlations. Combination not correct: {file}"
    combinations = combinations | combination
    family_a = {val['model_a_family'] for val in j}
    assert len(family_a) == 1, f"Problem in across correlations. Family A not correct: {file}"
    family_b = {val['model_b_family'] for val in j}
    assert len(family_b) == 1, f"Problem in across correlations. Family B not correct: {file}"
    families_a = families_a | family_a
    families_b = families_b | family_b

# TODO: Check more here
print('Passed!')
    
# Semantic Syntactic
print('Verifying Semantic/Syntactic tasks...')


tasks = ['bestmatch', 'mostsimilar', 'opposite', 'wordintrusion']

for task in tasks:
    print('Task:', task)
    families = set()
    models = set()
    for file in p.glob(f'semantic_syntactic/{task}/*.json'):
        with open(file) as f:
            j = json.load(f)
        try:
            family = j['parameter_string']
        except:
            family = j[0]['parameter_string']
        try:
            model_name = j['name']
        except:
            model_name = j[0]['name']
        families.add(family)
        models.add(model_name + '_' + family)
    assert len(families) == total_family_count, f"Problem in Semantic syntactic: not all families tested: {task}. No. Tested: {len(families)}"
    assert len(models) == total_model_count, f"Problem in Semantic syntactic: not all families tested: {task}. No. Tested: {len(models)}"


print('Passed!')

# Classification tasks
print('Verifying classification tasks...')

tasks = {'facebook', 'twitter', 'pressreleases', 'nationalrat', 'autnes_automated_2017', 'autnes_automated_2019', 'autnes_sentiment'}

models = set()
families = set()

for file in p.glob('classification/*.json'):
    model_name = set()
    family = set()
    with open(file) as f:
        j = json.load(f)
    assert tasks.issubset(set(j.keys())), f"Problem in classification tasks. Not all tasks are included. File: {file}"
    for task in tasks:
        _name = j[task]['metrics']['model_name']
        _family = j[task]['metrics']['parameter_string']
        model_name.add(_name + '_' + _family)
        family.add(_family)
    assert len(family) == 1, f"Problem in classification tasks. File inconsitent: {file}"
    assert len(model_name) == 1, f"Problem in classification tasks. File inconsitent: {file}"
    models = models | model_name
    families = families | family

assert len(models) == total_model_count + 2, f"Problem in classification task: Total number of models incorrect: {len(models)}"
assert len(families) == total_family_count + 2, f"Problem in classification task: Total number of models incorrect: {len(models)}"

print('Passed!')