import sys
sys.path.append('.')

from datetime import datetime
import platform
from argparse import ArgumentParser
from pathlib import Path
from utils.random_names import generate_random_name
import os
host = platform.node()
assert host != '', 'Could not get hostname!'

p = Path.cwd()

if __name__ == '__main__':

    print('-'*80)
    print('TESTING: Training Arguments')

    arg_parser = ArgumentParser(description="Train a fasttext model")
    arg_parser.add_argument('model_type', type=str, choices=['cbow', 'skipgram'])
    arg_parser.add_argument('training_corpus', type=str, help='Path to training corpus')
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag')
    arg_parser.add_argument('--threads', type=int, default=12, help='Number of parallel processes (default: 12)')

    arg_parser.add_argument('--learning_rate', type=float, default=0.1)
    arg_parser.add_argument('--epochs', type=int, default=5)
    arg_parser.add_argument('--word_ngrams', type=int, default=1)
    # arg_parser.add_argument('--loss_function', type=str, default="ns")
    arg_parser.add_argument('--min_count', type=int, default=1)
    arg_parser.add_argument('--window_size', type=int, default=5)
    arg_parser.add_argument('--dimensions', type=int, default=100)
    
    input_args = arg_parser.parse_args()

    training_corpus = Path(input_args.training_corpus)

    assert training_corpus.exists(), f'Could not find training data at: {training_corpus}'

    model_basedir = p / 'tmp_models'

    model_name = generate_random_name()

    corpus = training_corpus.name.replace('.txt', '')

    parameter_string = str(corpus) + '_' + \
                        str(input_args.model_type) + \
                        '_lr' + str(input_args.learning_rate) + \
                        '_epochs' + str(input_args.epochs) + \
                        '_mincount' + str(input_args.min_count) + \
                        '_ws' + str(input_args.window_size) + \
                        '_dims' + str(input_args.dimensions) 
    
    model_dir = model_basedir / parameter_string
    model_file_name = model_name + '_' + parameter_string
    model_path = model_dir / model_file_name

    print('Got the following arguments', input_args)
    print('Parameter String:', parameter_string)
    print('Model path:', model_path)

    command = [os.environ.get('FASTTEXT_PATH'), 
               input_args.model_type, 
               '-input', str(training_corpus.absolute()), 
               '-output', str(model_path.absolute()),
               '-minCount', str(input_args.min_count),
               '-wordNgrams', str(input_args.word_ngrams),
               '-lr', str(input_args.learning_rate),
               '-epoch', str(input_args.epochs),
               '-dim', str(input_args.dimensions),
               '-ws', str(input_args.window_size),
               '-thread', str(input_args.threads)]
    
    print('command:', command)

    print('-'*80)

