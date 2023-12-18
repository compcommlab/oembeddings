import sys
sys.path.append('.')

from utils.misc import get_data_dir
import re
from pathlib import Path

import pandas as pd
from argparse import ArgumentParser

arg_parser = ArgumentParser()

arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
arg_parser.add_argument('--seed', type=int, default=1234, help='Seed for random state (default: 1234)')

input_args = arg_parser.parse_args()

p = Path.cwd()

DATA_DIR = p / 'evaluation_data' / 'classification'

data_partition = get_data_dir()

FASTTEXT_DIR = data_partition / 'evaluation_data' / 'classification' / 'fasttext'
VALIDATION_DIR = data_partition / 'evaluation_data' / 'classification' / 'validation'
TRAINING_DIR = data_partition / 'evaluation_data' / 'classification' / 'training'

for d in [FASTTEXT_DIR, VALIDATION_DIR, TRAINING_DIR]:
    if not d.exists():
        d.mkdir(parents=True)

print('Preprocessing data')
for feather in DATA_DIR.glob('*.feather'):
    print(feather)
    df = pd.read_feather(feather)

    # convert to fasttext format: e.g., __label__spoe Sentence starts here.
    if 'labels' in df.columns:
        # handle multi-label data, which is a list of labels
        filt = df.labels.apply(lambda x: len(x) == 0)
        df.loc[filt, 'labels'] = df.loc[filt, 'labels'].apply(lambda x: ['none'])
        df['fasttext_label'] = df.labels.apply(lambda x: " ".join(['__label__' + label for label in x]))
        df['label'] = df.labels.apply(lambda x: " ".join([label for label in x]))
    else:
        # regular case: only one label per sample
        df['fasttext_label'] = '__label__' + df['label'].str.lower()

    df['fasttext_str'] = df.fasttext_label  + ' ' + \
        df['text'].str.replace('\n', ' ', regex=False).str.replace('\r', ' ').str.replace('\\n', ' ', regex=False) + '\n'
    df['fasttext_lower'] = df.fasttext_str.str.lower()

    training_data = df.sample(frac=0.75, random_state=input_args.seed)
    evaluation_data = df.drop(training_data.index)

    fasttext_name = FASTTEXT_DIR / \
        f"{feather.name.replace('.feather', '.train')}"
    with open(fasttext_name, 'w') as f:
        f.writelines(training_data.fasttext_str.to_list())

    training_name = TRAINING_DIR / f"{feather.name}"
    training_data.loc[:, ['label', 'fasttext_label', 'text']].reset_index(drop=True).to_feather(training_name, compression='uncompressed')

    fasttext_name = FASTTEXT_DIR / \
        f"{feather.name.replace('.feather', '.train_lower')}"
    with open(fasttext_name, 'w') as f:
        f.writelines(training_data.fasttext_lower.to_list())

    fasttext_name = FASTTEXT_DIR / \
        f"{feather.name.replace('.feather', '.test')}"
    with open(fasttext_name, 'w') as f:
        f.writelines(evaluation_data.fasttext_str.to_list())

    evaluation_name = VALIDATION_DIR / f"{feather.name}"
    evaluation_data.loc[:, ['label', 'fasttext_label', 'text']].reset_index(drop=True).to_feather(evaluation_name, compression='uncompressed')

    fasttext_name = FASTTEXT_DIR / \
        f"{feather.name.replace('.feather', '.test_lower')}"
    with open(fasttext_name, 'w') as f:
        f.writelines(evaluation_data.fasttext_lower.to_list())

uppercase = re.compile(r'[A-ZÄÖÜ]')

# validate generated data
for text_file in FASTTEXT_DIR.glob('*'):
    with open(text_file) as f:
        for line in f.readlines():
            assert line.startswith('__label__'), text_file
            if 'lower' in text_file.name:
                assert not uppercase.match(line), f'File: {text_file}, line: {line}'