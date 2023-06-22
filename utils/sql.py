
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, inspect
import os
from utils.datamodel import Base, Article

def start_sqlsession(connect_string, echo=False, **kwargs):
    engine = create_engine(connect_string, echo=echo)

    Session = sessionmaker(bind=engine)
    session = Session()

    # Ensure SQL database exists
    inspector = inspect(engine)
    if len(inspector.get_table_names()) == 0:
        Base.metadata.create_all(engine)

    for table in Base.metadata.tables.keys():
        if table not in inspector.get_table_names():
            print("Database does not exist, creating tables")
            Base.metadata.create_all(engine)

    return session, engine
