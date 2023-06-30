import sys
sys.path.append('.')
from utils.sql import start_sqlsession
from utils.datamodel import Article
from utils.misc import md5sum
from utils.cleaning import wrong_encoding
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import re
# run pandas apply in parallel:
import swifter
# ftfy: fix text for you (solves encoding issues)
import ftfy
from typing import Union, Any

# safe wrapper
def fix_text(text: Any) -> Union[str, None]:
    try:
        return ftfy.fix_text(text)
    except:
        return None


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="Load all feather files in 'raw_data' to the SQL Databse")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: only load a random sample')
    arg_parser.add_argument('--clean', action='store_true', help='Clean the database before loading (delete all articles)')
    
    input_args = arg_parser.parse_args()
    
    session, engine = start_sqlsession()

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
            df = df.sample(100).reset_index(drop=True)

        print('Calculating md5 sum for URL column')
        df['article_md5'] = df.url.swifter.apply(md5sum)

        # some texts might have the wrong encoding
        text_cols = {"headline", "description", "pretitle", "lead_paragraph", "body"}
        text_cols = set(df.columns.tolist()).intersection(text_cols)

        encoding_issues = False
        # if any column has an encoding issue, we assume that the entire dataset is corrupted
        for indicator in wrong_encoding:
            for col in text_cols:
                try:
                    _issues = sum(df[col].str.contains(indicator))
                    if _issues > 0:
                        print('Detected encoding issue in column', col)
                        encoding_issues = True
                        break
                except:
                    continue

        if encoding_issues:
            print('fixing encoding')
            for col in text_cols:
                df[col] = df[col].swifter.apply(fix_text)

        if 'diepresse' in feather.name:
            # there is currently a parsing bug that caused some 
            # sentences not to be separated properly
            # we can fix that with a crude regex
            wrong_sentences = re.compile(r"(\b[A-ZÄÖÜa-zäöü]{3,}?!?)([A-Z])")
            df['body'] = df['body'].swifter.apply(lambda x: wrong_sentences.sub(r"\1. \2", x))
            
            
        # drop duplicates; make sure article not in db already
        filt = df.article_md5.isin(article_md5)
        df = df[~filt].reset_index(drop=True)

        # save to SQL db
        df.to_sql(Article.__tablename__, session.bind, if_exists="append", index=False)
