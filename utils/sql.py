
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, inspect
import os
from utils.datamodel import Base, Article
from dotenv import load_dotenv
load_dotenv()
import os

dbstring = os.environ.get('OEMBEDDINGS_DB', 'sqlite:///database.db')

def start_sqlsession(connect_string=dbstring, echo=False, **kwargs):
    engine = create_engine(connect_string, echo=echo, pool_size=100)

    Session = sessionmaker(bind=engine)
    session = Session()

    # Ensure SQL database exists
    inspector = inspect(engine)
    if len(inspector.get_table_names()) == 0:
        Base.metadata.create_all(engine)

    for table in Base.metadata.tables.keys():
        if table not in inspector.get_table_names():
            print(f"Table <{table}> does not exist, creating it")
            Base.metadata.create_all(engine)

    return session, engine
