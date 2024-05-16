import sys
sys.path.append('.')
from utils.sql import start_sqlsession
from utils.datamodel import Article, ProcessedParagraph
from utils.misc import md5sum
from utils.cleaning import clean_text
from argparse import ArgumentParser
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
import sqlalchemy
from multiprocessing import Pool

# start sql
session, engine = start_sqlsession()

def add_if_not_duplicated(text: str) -> None:
    """ Take a text (string) then calc md5 sum 
        check if it is already in the SQL database,
        if not then add new text
        if already exists, increment count by 1
    """
    if text == "":
        return None
    local_session = sessionmaker(bind=engine)()
    text_md5 = md5sum(text)
    duplicated_text = local_session.query(ProcessedParagraph).filter(ProcessedParagraph.md5 == text_md5).first()
    if duplicated_text:
        duplicated_text.count += 1
    else:
        n_tokens = len(text.split())
        new_text = ProcessedParagraph(md5=text_md5, text=text, n_tokens=n_tokens)
        local_session.add(new_text)
    local_session.commit()
    local_session.close()

def get_article(article_id: int) -> Article:
    local_session = sessionmaker(bind=engine)()
    article = local_session.query(Article).filter(Article.id == article_id).first()
    local_session.close()
    return article

def process_article(article_id: int, **kwargs) -> None:
    try:
        article = get_article(article_id)
        headline = article.pretitle or " "
        headline += " "
        headline += article.headline or ""
        headline = clean_text(headline, **kwargs)
        add_if_not_duplicated(headline)
        lead_paragraph = clean_text(article.lead_paragraph, **kwargs)
        add_if_not_duplicated(lead_paragraph)

        for paragraph in article.body.split('\n\n'):
            paragraph = clean_text(paragraph, **kwargs)
            add_if_not_duplicated(paragraph)

    except Exception as e:
        print('couldnt process', e)


def initializer():
    """ensure the parent proc's database connections are not touched
    in the new connection pool
    see SQL Alchemy documentation: 
    https://docs.sqlalchemy.org/en/20/core/pooling.html
    """
    engine.dispose(close=False)


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="Process Articles and clean text")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
    arg_parser.add_argument('--threads', type=int, default=1, help='Number of parallel processes (default: 1)')
    arg_parser.add_argument('--clean_database', action='store_true', help='Remove all previously processed articles')

    arg_parser.add_argument('--remove_links', action='store_true', help='Remove hyperlinks')
    arg_parser.add_argument('--remove_emails', action='store_true', help='Remove emails')
    arg_parser.add_argument('--remove_emojis', action='store_true', help='Remove emojis')
    arg_parser.add_argument('--remove_punctuation', action='store_true', help='Remove punctutation')
    arg_parser.add_argument('--remove_numbers', action='store_true', help='Remove numbers')
    arg_parser.add_argument('--replace_numbers', action='store_true', help='Replace numbers with words (1 -> "eins")')
    arg_parser.add_argument('--remove_quotations', action='store_true', help='Remove quotation marks')
    arg_parser.add_argument('--genderstar', action='store_true', help='Preserve genderstar (normalize with underscore)')
    arg_parser.add_argument('--before', type=str, default=None, help='Only consider articles on and before the given date (YYYY-MM-DD)')
    arg_parser.add_argument('--after', type=str, default=None, help='Only consider articles on and after the given date (YYYY-MM-DD)')
    
    input_args = arg_parser.parse_args()

    if input_args.clean_database:
        print('Removing all previously processed articles')
        session.query(ProcessedParagraph).delete()
        session.commit()

    
    n_threads = input_args.threads

    # get all article ids and reformat to clean list
    q = session.query(Article.id)
    if input_args.before:
        print(f'Got input argument "before": {input_args.before}')
        q = q.filter(Article.date_published <= input_args.before)
    if input_args.after:
        print(f'Got input argument "after": {input_args.after}')
        q = q.filter(Article.date_published >= input_args.after)
    article_ids = q.all()
    article_ids = [a[0] for a in article_ids]
    print(f"Got {len(article_ids)} articles to parse")
    if input_args.debug:
        import random
        article_ids = random.sample(article_ids, 1000)

    settings = {"remove_links": input_args.remove_links,
                "remove_emails": input_args.remove_emails,
                "remove_emojis": input_args.remove_emojis,
                "remove_punctuation": input_args.remove_punctuation,
                "remove_numbers": input_args.remove_numbers,
                "replace_numbers": input_args.replace_numbers,
                "remove_quotations": input_args.remove_quotations,
                "genderstar": input_args.genderstar}

    print('Start cleaning ...')

    with Pool(n_threads, initializer=initializer) as pool:
        for raw_id in tqdm(article_ids, desc="Processing", unit="articles"):
            pool.apply(process_article, (raw_id, ), kwds=settings)

    if 'postgresql' in engine.dialect.name:
        print('Calculating token counts')
        clear_table = r"""
            DO $$ 
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'token_counts') THEN
                    -- If it exists, delete the table
                    DROP TABLE token_counts;
                END IF;
            END $$;
        """

        count_tokens = """
            CREATE TABLE token_counts AS
            SELECT
                word,
                COUNT(*) AS count
            FROM (
                SELECT regexp_split_to_table(text, E'\\s') AS word
                FROM processed_articles
            ) AS subquery
            GROUP BY word;
        """

        with engine.connect() as con:
            con.begin()
            r1 = con.execute(sqlalchemy.text(clear_table))
            r2 = con.execute(sqlalchemy.text(count_tokens))
            con.commit()