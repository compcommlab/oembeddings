import sys
sys.path.append('.')
from utils.sql import start_sqlsession
from utils.datamodel import Article, RawSentence
from utils.misc import md5sum
from argparse import ArgumentParser
import spacy
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from multiprocessing import Pool

# start sql
session, engine = start_sqlsession()


def add_if_not_duplicated(sentence: str) -> None:
    """ Take a sentence (string) then calc md5 sum 
        check if it is already in the SQL database,
        if not then add new sentence
        if already exists, increment count by 1
    """
    local_session = sessionmaker(bind=engine)()
    sentence_md5 = md5sum(sentence)
    duplicated_sentence = local_session.query(RawSentence).filter(RawSentence.sentence_md5 == sentence_md5).first()
    if duplicated_sentence:
        duplicated_sentence.count += 1
    else:
        new_sentence = RawSentence(sentence_md5=sentence_md5, sentence=sentence)
        local_session.add(new_sentence)
    local_session.commit()
    local_session.close()


def process_headline(article_id: int) -> None:
    """ Get only article headline and pretitle 
        and add them as raw sentences
        also add them as combined string: pretitle + headline
    """
    local_session = sessionmaker(bind=engine)()
    article = local_session.query(Article).filter(Article.id == article_id).first()
    if article.headline and article.headline != "":
        try:
            add_if_not_duplicated(article.headline)
        except Exception as e:
            print("Error when adding headline", article.article_id, e)
    if article.pretitle and article.pretitle != "":
        try:
            add_if_not_duplicated(article.pretitle)
        except Exception as e:
            print("Error when adding pretitle", article.article_id, e)
    if article.pretitle and article.pretitle != "" and article.headline and article.headline != "":
        try:
            add_if_not_duplicated(article.pretitle + " " + article.headline)
        except Exception as e:
            print("Error when adding pretitle + headline", article.article_id, e)
    local_session.close()

def yield_article(column: str, ids: list) -> str:
    """ Generator function to return articles 
        used for spacy to load articles in batches
    """
    for a_id in ids:
        try:
            local_session = sessionmaker(bind=engine)()
            article = local_session.query(Article).filter(Article.id == a_id).first()
            local_session.close()
            text = getattr(article, column)
            if text:
                yield text
        except Exception as e:
            pass

def initializer():
    """ensure the parent proc's database connections are not touched
    in the new connection pool
    see SQL Alchemy documentation: 
    https://docs.sqlalchemy.org/en/20/core/pooling.html
    """
    engine.dispose(close=False)


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="Split articles into sentences")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
    arg_parser.add_argument('--threads', type=int, default=1, help='Number of parallel processes (default: 1)')
    input_args = arg_parser.parse_args()

    n_threads = input_args.threads

    # get all article ids and reformat to clean list
    article_ids = session.query(Article.id).all()
    article_ids = [a[0] for a in article_ids]


    # first add all headlines and pre-titles (without splitting)
    with Pool(4, initializer=initializer) as p:
        r = list(tqdm(p.imap(process_headline, article_ids), 
                      total=len(article_ids),
                      desc="Headlines"))
    

    # load spacy for sentence splitting
    nlp = spacy.load("de_core_news_lg")
    nlp.disable_pipes('ner', 'tagger')
    nlp.enable_pipe('senter')

    docs = nlp.pipe(yield_article("lead_paragraph", article_ids), n_process=4)
    for doc in tqdm(docs, total=len(article_ids), desc="Lead paragraph"):
        for s in doc.sents:
            add_if_not_duplicated(s.text.strip())

    docs = nlp.pipe(yield_article("description", article_ids), n_process=4)
    for doc in tqdm(docs, total=len(article_ids), desc="Description"):
        for s in doc.sents:
            add_if_not_duplicated(s.text.strip())

    docs = nlp.pipe(yield_article("body", article_ids), n_process=4)
    for doc in tqdm(docs, total=len(article_ids), desc="Article body"):
        for s in doc.sents:
            add_if_not_duplicated(s.text.strip())

