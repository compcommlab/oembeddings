from utils.sql import start_sqlsession
from utils.datamodel import Article
from utils.misc import md5sum
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

if __name__ == '__main__':
    arg_parser = ArgumentParser(description="Load all feather files in 'raw_data' to the SQL Databse")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
    arg_parser.add_argument('--clean', action='store_true', help='Clean the database before loading (delete all articles)')
    arg_parser.add_argument('--db', type=str, dest='db', default="sqlite:///database.db",
                    help='Maximum number of articles to scrape')
    
    input_args = arg_parser.parse_args()

    connect_string = arg_parser.db

    session, engine = start_sqlsession(connect_string)

    if input_args.clean:
        print('Cleaning database...')
        session.query(Article).delete()
        session.commit()

    p = Path.cwd() / "raw_data"

    for feather in p.rglob('*.feather'):
        # get exisiting Articles by md5 id
        article_md5 = session.query(Article.article_md5).all()
        print('Loading feather file', feather)
        df = pd.read_feather(feather)
        if input_args.debug:
            df = df.sample(1000).reset_index(drop=True)

        df['article_md5'] = df.url.apply(md5sum)

        # drop duplicates; make sure article not in db already
        filt = df.article_md5.isin(article_md5)
        df = df[~filt].reset_index(drop=True)

        # save to SQL db
        df.to_sql(Article.__tablename__, session.bind, if_exists="append", index=False)
