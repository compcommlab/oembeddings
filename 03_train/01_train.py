import sys
sys.path.append('.')
import os
from argparse import ArgumentParser
from pathlib import Path
import fasttext

p = Path.cwd()

model_dir = p / 'tmp_models'

if not model_dir.exists():
    model_dir.mkdir()

model_path = model_dir / 'oembeddings.bin'

training_data = p / 'data' / 'training_data.txt'

assert training_data.exists(), f'Could not find training data at: {training_data}'

print('Training model...')
model = fasttext.train_unsupervised(str(training_data), model='skipgram')

print('Saving model...')
model.save_model(str(model_path))