import sys
sys.path.append('.')
import os
from sqlalchemy import text
from utils.sql import start_sqlsession
from utils.datamodel import Sentence
from argparse import ArgumentParser
from tqdm import tqdm
from sqlalchemy import func
from pathlib import Path
from math import ceil
import random

import pandas as pd

# start sql
session, engine = start_sqlsession()

if __name__ == '__main__':
    arg_parser = ArgumentParser(description="Clean raw sentences")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
    arg_parser.add_argument('--min_length', type=int, default=5, help='Minimum length of tokens per sentence (default: 5 tokens)')
    arg_parser.add_argument('--batch_size', type=int, default=10000, help='Number of sentences to load per batch (default: 10,000 sentences)')
    arg_parser.add_argument('--corpus_name', type=str, default="training_data", help='Name of the corpus file; file extension is added automatically (default "taining_data")')
    arg_parser.add_argument('--seed', type=int, default=1234, help='Seed for shuffling (default: 1234)')
    
    input_args = arg_parser.parse_args()

    p = Path.cwd()

    data_dir = p / 'data'

    if not data_dir.exists():
        data_dir.mkdir()

    output_file = data_dir / f'{input_args.corpus_name}.txt'

    if output_file.exists():
        os.remove(output_file)

    output_file.touch()

    total_sentences = session.query(Sentence).filter(Sentence.n_tokens >= input_args.min_length).count()
    print(f'Got {total_sentences} sentences')

    """ If we have PostgreSQL we can leverage its built-in export function """
    if 'postgresql' in engine.dialect.name:
        print('Using postgres export function...')
        import psycopg2
        conn = psycopg2.connect(engine.url.render_as_string(hide_password=False))
        cursor = conn.cursor()
        query_string = f"SELECT setseed(0.{input_args.seed}); "
        query_string += "COPY (SELECT sentence FROM sentences"
        query_string += f" WHERE n_tokens >= {input_args.min_length}"
        query_string += f" ORDER BY RANDOM()"
        if input_args.debug:
            query_string += " LIMIT 10000"
        query_string += ") TO STDOUT"
        # query_string += str(output_file.absolute())
        # query_string += "'"
        cursor.copy_expert(
            query_string,
            open(output_file.absolute(), 'w')
        )

        cursor.close()
        conn.close()
        
    else:

        pages = ceil(total_sentences / input_args.batch_size)

        if input_args.debug:
            pages = 100

        """
            Shuffle batches of training data. Smaller batches == more shuffeling
        """
        pages = list(range(pages))
        random.shuffle(pages)

        base_query = session.query(Sentence.sentence).filter(Sentence.n_tokens >= input_args.min_length)

        for p in tqdm(pages, total=len(pages), unit="batches"):
            query = base_query.limit(input_args.batch_size).offset(input_args.batch_size * p)
            with engine.begin() as con:
                df = pd.read_sql(query.statement, con)
            df.sentence.to_csv(output_file, index=False, header=False, mode="a")
        
    session.close()