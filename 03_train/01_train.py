import sys
sys.path.append('.')
import os
import subprocess
import re
import time
from argparse import ArgumentParser
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from utils.sql import start_sqlsession
from utils.datamodel import Model, ModelTrainingProgress
from utils.random_names import generate_random_name

import fasttext

p = Path.cwd()

session, engine = start_sqlsession()

if __name__ == '__main__':

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

    model_dir = p / 'tmp_models'

    if not model_dir.exists():
        model_dir.mkdir()

    model_name = generate_random_name()

    model_file_name = model_name + \
                    '_lr' + str(input_args.learning_rate) + \
                    '_epochs' + str(input_args.epochs) + \
                    '_mincount' + str(input_args.min_count) + \
                    '_ws' + str(input_args.window_size) + \
                    '_dims' + str(input_args.dimensions) 

    model_path = model_dir / model_file_name

    training_corpus = Path(input_args.training_corpus)

    assert training_corpus.exists(), f'Could not find training data at: {training_corpus}'

    model = Model(name=model_name,
                  training_corpus=str(training_corpus.absolute()),
                  model_type=input_args.model_type,
                  learning_rate=input_args.learning_rate,
                  epochs=input_args.epochs,
                  word_ngrams=input_args.word_ngrams,
                  min_count=input_args.min_count,
                  window_size=input_args.window_size,
                  dimensions=input_args.dimensions,
                  model_path=str(model_path.absolute()))
    
    session.add(model)
    session.commit()

    print('Training model...')
    print('Model name:', model_name)
    print('Model path:', model_path)
    print('Model ID:', model.id)

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
    
    training_start_time = time.time()

    if input_args.debug:
        print('calling command:', command)

    p = subprocess.Popen(command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        encoding='utf-8')
    
    loss_value = 100.0 # pre-allocate
    progress = 0.0
    
    while True:
        realtime_output = p.stdout.readline()

        if realtime_output == '' and p.poll() is not None:
            break

        if realtime_output:
            try:
                new_progress = re.search(r"Progress:\s*(\d+.\d)%", realtime_output)[1]
                new_progress = float(new_progress)

                loss_value = re.search(r"avg\.loss: *(\d\.\d+) *ETA", realtime_output)[1]
                loss_value = float(loss_value)

                words_sec_thread = re.search(r"thread:\s*(\d+)\s", realtime_output)[1]
                words_sec_thread = float(words_sec_thread)

                lr = re.search(r"lr:\s*(\d+\.\d+)\s", realtime_output)[1]
                lr = float(lr)

                if new_progress != progress:
                    # only print to console when progress increased
                    print(realtime_output.strip(), flush=True)
                    progress = new_progress
                    # updated = ModelTrainingProgress(model_id=model.id,
                    #                                 progress=progress,
                    #                                 loss=loss_value,
                    #                                 learning_rate=lr,
                    #                                 words_sec_thread=words_sec_thread)
                    # try:
                    #     session.add(updated)
                    #     session.commit()
                    # except:
                    #     session.rollback()
            except:
                print(realtime_output.strip(), flush=True)

    computation_time = time.time() - training_start_time

    # test whether model can be loaded
    loaded_model = fasttext.load_model(str(model_path.absolute()) + '.bin')

    vocab_size = len(loaded_model.words)

    model.computation_time = computation_time
    model.avg_loss = loss_value
    model.vocab_size = vocab_size

    session.commit()
    session.close()