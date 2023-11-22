import os
import sys
sys.path.append('.')

from utils.misc import get_data_dir
from sklearn.metrics import precision_recall_fscore_support
import json
from pathlib import Path
import fasttext

# Supress Fasttext warnings when loading a model
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from argparse import ArgumentParser
from time import time

p = Path.cwd()

DATA_DIR = p / 'evaluation_data' / 'classification'

data_partition = get_data_dir()

PROCESSED_DIR = data_partition / 'evaluation_data' / 'classification' / 'fasttext'
VALIDATION_DIR = data_partition / 'evaluation_data' / 'classification' / 'validation'

RESULTS_DIR = data_partition / 'evaluation_results' / 'classification'


def evaluate(model_path: str, 
             training_file: Path,
             dims: int,
             threads: int = 12) -> dict:
    print('Using corpus', training_file)
    t = time()
    model = fasttext.train_supervised(str(training_file),
                                        pretrainedVectors=model_path,
                                        thread=threads,
                                        dim=dims)

    test_file = training_file.name.split('.')[0] + '.feather'
    validation_corpus = pd.read_feather(VALIDATION_DIR / test_file)
    validation_corpus["text"] = validation_corpus['text'].str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False)
    labels = validation_corpus.fasttext_label.unique().tolist()

    if 'lower' in model_path:
        validation_sentences = validation_corpus.text.str.lower().to_list()
    else:
        validation_sentences = validation_corpus.text.to_list()

    predictions = model.predict(validation_sentences)
    duration = time() - t

    # results is a tuple with len 2: first are the predicted labels
    # second are the probabilities for the labels; we only need the first one
    # get first element of each list
    predicted_labels = list(map(lambda x: x[0], predictions[0]))
    true_labels = validation_corpus.fasttext_label.to_list()
    
    scores = precision_recall_fscore_support(true_labels, 
                                             predicted_labels,
                                             labels=labels,
                                             average='macro')

    results = [{'task': training_file.name.replace('.train', '').replace('_lower', ''),
                'label': 'overall (macro)',
                'precision': scores[0],
                'recall': scores[1],
                'f1score': scores[2],
                'duration': duration
            }]
    
    scores_labels = precision_recall_fscore_support(true_labels, 
                                                    predicted_labels,
                                                    labels=labels)
    
    results_labels = {label: dict() for label in labels}
    for metric, values in zip(['precision', 'recall', 'f1'], scores_labels[:3]):
        for label, value in zip(labels, values):
            results_labels[label][metric] = value

    for label, vals in results_labels.items():
        results.append({'task': training_file.name.replace('.train', ''),
                        'label': label.replace('__label__', ''),
                        'duration': duration,
                        **vals})
    return {'metrics': results, 'predicted_labels': predicted_labels, 'true_labels': true_labels}


if __name__ == '__main__':

    if not RESULTS_DIR.exists():
        try:
            RESULTS_DIR.mkdir(parents=True)
        except FileExistsError:
            pass

    if not PROCESSED_DIR.exists():
        raise Exception('Could not find pre-processed data. Run `evaluation_data/prepare_classification_data.py` first')

    arg_parser = ArgumentParser(description="Evaluate fasttext models on a classification task")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
    arg_parser.add_argument('--threads', type=int, default=12, help='Number of parallel processes (default: 12)')
    arg_parser.add_argument('--seed', type=int, default=1234, help='Seed for random state (default: 1234)')
    arg_parser.add_argument('--modelfamily', type=str, default=None, help="Specificy a directory of models to evaluate")
    arg_parser.add_argument('--corpus', type=str, default=None, help="Filename of corpus to evaluate. (needs to end with '.train')")
    
    input_args = arg_parser.parse_args()

    if input_args.modelfamily:
        model_dir = Path(input_args.modelfamily)
        glob_pattern = "*.json"
    else:
        model_dir = get_data_dir()
        glob_pattern = 'tmp_models/*/*.json'

    for model_info in model_dir.glob(glob_pattern):
        model_meta = json.load(model_info.open())
        print('Evaluating:', model_meta['name'])

        model_path = model_meta['model_path'] + '.vec'

        if input_args.corpus:
            corpus_path = Path(input_args.corpus)
            assert corpus_path.exists(), f'Training corpus not found at this location: {corpus_path}'
            r = evaluate(model_path, 
                               corpus_path, 
                               model_meta["dimensions"],
                               threads=input_args.threads)
            
            results[corpus_path.name.split('.')[0]] = r

            results_file = RESULTS_DIR / f"{model_meta['name']}_{model_meta['parameter_string']}_{corpus_path.name.removesuffix('.train')}.json"

        else:
            if 'lower' in model_meta['parameter_string']:
                glob_pattern = '*.train_lower'
            else:
                glob_pattern = '*.train'
            results = {}
            for training_file in PROCESSED_DIR.glob(glob_pattern):
                r = evaluate(model_path, 
                            training_file, 
                            model_meta["dimensions"],
                            threads=input_args.threads)
                results[training_file.name.split('.')[0]] = r

            results_file = RESULTS_DIR / f"{model_meta['name']}_{model_meta['parameter_string']}.json"
            
        for result in results.values():
            for r in result['metrics']:
                # error here
                r['model_name'] = model_meta['name']
                r['parameter_string'] = model_meta['parameter_string']
                r['model_path'] = model_meta['model_path']
        
        # delete results file first
        if results_file.exists():
            print('Found existing results file, deleting it ...')
            os.remove(results_file)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=True)
