import sys
sys.path.append('.')
from utils.sql import start_sqlsession
from utils.datamodel import RawSentence, Sentence
from utils.misc import md5sum
from utils.cleaning import *
from argparse import ArgumentParser
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from multiprocessing import Pool

import emoji
import re

from typing import Union

# start sql
session, engine = start_sqlsession()

def clean_sentence(text: str,
                     lowercase=False,
                     remove_links=True,
                     remove_emails=True,
                     remove_emojis=True,
                     remove_punctuation=True,
                     remove_numbers=True) -> str:
    
    text = HTML_FRAGMENTS.sub(" ", text)
    
    if lowercase:
        text = text.lower()
    if remove_links:
        text = LINK_REGEX.sub(" ", text)
    if remove_emails:
        text = EMAIL_REGEX.sub(" ", text)
    if remove_emojis:
        text = emoji.replace_emoji(text, " ")
    if remove_punctuation:
        text = PUNCTUATION.sub(" ", text)
        text = QUOTATION_MARKS.sub(" ", text)
    else:
        # pad punctuation with whitespace
        text = PUNCTUATION.sub(r' \1 ', text)
        text = QUOTATION_MARKS.sub(r' \1 ', text)
    if remove_numbers:
        text = NUMBERS.sub("", text)

    text = LINE_BREAKS.sub(" ", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()
    
def process_sentence(sentence_id: int, **kwargs) -> None:
    try:
        local_session = sessionmaker(bind=engine)()
        raw = local_session.query(RawSentence).filter(RawSentence.id == sentence_id).first()
        sentence = clean_sentence(raw.sentence, **kwargs)
        if sentence == "":
            return None
        sentence_md5 = md5sum(sentence)
        duplicated_sentence = local_session.query(Sentence).filter(Sentence.sentence_md5 == sentence_md5).first()
        if duplicated_sentence:
            duplicated_sentence.count += 1
        else:
            n_tokens = len(sentence.split())
            new_sentence = Sentence(sentence_md5=sentence_md5, sentence=sentence, n_tokens=n_tokens)
            local_session.add(new_sentence)
        local_session.commit()
    except Exception as e:
        print('couldnt process', e)
    finally:
        local_session.close()


def initializer():
    """ensure the parent proc's database connections are not touched
    in the new connection pool
    see SQL Alchemy documentation: 
    https://docs.sqlalchemy.org/en/20/core/pooling.html
    """
    engine.dispose(close=False)


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="Clean raw sentences")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
    arg_parser.add_argument('--clean_database', action='store_true', help='Remove all previously processed sentences')
    arg_parser.add_argument('--lowercase', action='store_true', help='Lowercase text')
    arg_parser.add_argument('--remove_links', action='store_true', help='Remove hyperlinks')
    arg_parser.add_argument('--remove_emails', action='store_true', help='Remove emails')
    arg_parser.add_argument('--remove_emojis', action='store_true', help='Remove emojis')
    arg_parser.add_argument('--remove_punctuation', action='store_true', help='Remove punctutation')
    arg_parser.add_argument('--remove_numbers', action='store_true', help='Remove numbers')
    arg_parser.add_argument('--threads', type=int, default=1, help='Number of parallel processes (default: 1)')
    input_args = arg_parser.parse_args()

    n_threads = input_args.threads

    if input_args.clean_database:
        print('Deleting all previouls processed sentences...')
        session.query(Sentence).delete()
        session.commit()

    # get all raw sentence ids and reformat to clean list
    raw_sentece_ids = session.query(RawSentence.id).all()
    raw_sentece_ids = [a[0] for a in raw_sentece_ids]
    print('Got', len(raw_sentece_ids), 'raw sentences to process ...')

    settings = {'lowercase': input_args.lowercase,
                "remove_links": input_args.remove_links,
                "remove_emails": input_args.remove_emails,
                "remove_emojis": input_args.remove_emojis,
                "remove_punctuation": input_args.remove_punctuation,
                "remove_numbers": input_args.remove_numbers}

    with Pool(n_threads, initializer=initializer) as p:
        for raw_id in tqdm(raw_sentece_ids, desc="Processing", unit="sentences"):
            p.apply(process_sentence, (raw_id, ), kwds=settings)
