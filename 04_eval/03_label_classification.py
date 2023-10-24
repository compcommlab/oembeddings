import sys
sys.path.append('.')

from utils.misc import harmonic_mean, get_data_dir
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
RESULTS_DIR = data_partition / 'evaluation_results' / 'classification'

def preprocess_data():
    print('Preprocessing data')
    if not PROCESSED_DIR.exists():
        PROCESSED_DIR.mkdir(parents=True)

    for feather in DATA_DIR.glob('*.feather'):
        print(feather)
        df = pd.read_feather(feather)

        # convert to fasttext format: e.g., __label__spoe Sentence starts here.
        df['fasttext_str'] = '__label__' + df['label'].str.lower() + ' ' + \
            df['text'].str.replace('\n', ' ') + '\n'
        df['fasttext_lower'] = df.fasttext_str.str.lower()

        training_data = df.sample(frac=0.75, random_state=input_args.seed)
        evaluation_data = df.drop(training_data.index)

        prcocessed_name = PROCESSED_DIR / \
            f"{feather.name.replace('.feather', '.train')}"
        with open(prcocessed_name, 'w') as f:
            f.writelines(training_data.fasttext_str.to_list())

        prcocessed_name = PROCESSED_DIR / \
            f"{feather.name.replace('.feather', '.train_lower')}"
        with open(prcocessed_name, 'w') as f:
            f.writelines(training_data.fasttext_lower.to_list())

        prcocessed_name = PROCESSED_DIR / \
            f"{feather.name.replace('.feather', '.test')}"
        with open(prcocessed_name, 'w') as f:
            f.writelines(evaluation_data.fasttext_str.to_list())

        prcocessed_name = PROCESSED_DIR / \
            f"{feather.name.replace('.feather', '.test_lower')}"
        with open(prcocessed_name, 'w') as f:
            f.writelines(evaluation_data.fasttext_lower.to_list())


if __name__ == '__main__':

    if not RESULTS_DIR.exists():
        try:
            RESULTS_DIR.mkdir(parents=True)
        except FileExistsError:
            pass

    if not PROCESSED_DIR.exists():
        raise Exception('Could not find pre-processed data. Run this script first with the parameter `--preprocess`')

    arg_parser = ArgumentParser(description="Evaluate fasttext models on a classification task")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
    arg_parser.add_argument('--preprocess', action='store_true', help='Preprocess raw data into fasttext format')
    arg_parser.add_argument('--threads', type=int, default=12, help='Number of parallel processes (default: 12)')
    arg_parser.add_argument('--seed', type=int, default=1234, help='Seed for random state (default: 1234)')
    arg_parser.add_argument('--modelfamily', type=str, default=None, help="Specificy a directory of models to evaluate")

    input_args = arg_parser.parse_args()

    if input_args.preprocess:
        preprocess_data()

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

        if 'lower' in model_meta['parameter_string']:
            glob_pattern = '*.train_lower'
        else:
            glob_pattern = '*.train'

        results = []
        for training_file in PROCESSED_DIR.glob(glob_pattern):
            t = time()
            model = fasttext.train_supervised(str(training_file),
                                                pretrainedVectors=model_path,
                                                thread=input_args.threads,
                                                dim=model_meta["dimensions"])

            test_file = str(training_file).replace('.train', '.test')
            # n samples, precision, recall
            results_overall = model.test(test_file)
            results_labels = model.test_label(test_file)
            duration = time() - t

            results.append({'task': training_file.name.replace('.train', '').replace('_lower', ''),
                            'label': 'overall',
                            'precision': results_overall[1],
                            'recall': results_overall[2],
                            'f1score': harmonic_mean(results_overall[1], results_overall[2]),
                            'duation': duration
                            })

            for label, vals in results_labels.items():
                results.append({'task': training_file.name.replace('.train', ''),
                                'label': label.replace('__label__', ''),
                                'duation': duration,
                                **vals})
        for r in results:
            r['model_name'] = model_meta['name']
            r['parameter_string'] = model_meta['parameter_string']
            r['model_path'] = model_meta['model_path']
        
        results_file = RESULTS_DIR / f"{model_meta['name']}_{model_meta['parameter_string']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=True)
