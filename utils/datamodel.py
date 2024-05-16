from typing import Optional
from sqlalchemy import ForeignKey, BigInteger, DateTime, func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from datetime import datetime

class Base(DeclarativeBase):
    
    def _as_dict(self) -> dict:
        """ 
            returns a dictionary representation of an instance
            used for dumping an instance as JSON file 
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class Article(Base):
    __tablename__ = "articles"
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(index=True)
    article_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True) # native id of article (potentially duplicated)
    article_md5: Mapped[str] = mapped_column(index=True, nullable=False, unique=True) # hashed URL of article (md5sum), has to be unique
    url: Mapped[str] = mapped_column(nullable=False, index=True)
    section: Mapped[Optional[str]]
    premium: Mapped[Optional[int]]
    date_published: Mapped[Optional[datetime]]
    date_modified: Mapped[Optional[datetime]]
    has_ticker: Mapped[Optional[int]]
    description: Mapped[Optional[str]]
    headline: Mapped[Optional[str]]
    pretitle: Mapped[Optional[str]]
    lead_paragraph: Mapped[Optional[str]]
    picture_links: Mapped[Optional[str]]
    picture_captions: Mapped[Optional[str]]
    author: Mapped[Optional[str]]
    body: Mapped[Optional[str]]
    comments: Mapped[Optional[int]] = mapped_column(BigInteger)
    tweet_ids: Mapped[Optional[str]]
    slide_show: Mapped[Optional[str]]
    keywords: Mapped[Optional[str]]
    tags: Mapped[Optional[str]]
    scrape_timestamp: Mapped[Optional[datetime]]
    article_uuid: Mapped[Optional[str]] # only used for Krone   

    def __repr__(self) -> str:
        return (f"<Article(article_id={self.article_id}, "
                f"source={self.source}, "
                f"url={self.url}>")

class RawSentence(Base):

    __tablename__ = 'raw_sentences'

    id: Mapped[int] = mapped_column(primary_key=True)
    sentence_md5: Mapped[str] = mapped_column(index=True, nullable=False, unique=True) # hash value of sentence to determine duplicates
    sentence: Mapped[str] = mapped_column(nullable=False) # actual sentence
    count: Mapped[int] = mapped_column(default=1) # count how many times the sentence was found in the dataset

    def __repr__(self) -> str:
        return (f"<RawSentence(sentence_md5={self.sentence_md5})>")


class Sentence(Base):

    __tablename__ = 'sentences'

    id: Mapped[int] = mapped_column(primary_key=True)
    sentence_md5: Mapped[str] = mapped_column(index=True, nullable=False, unique=True) # hash value of sentence to determine duplicates
    sentence: Mapped[str] = mapped_column(nullable=False) # actual sentence
    n_tokens: Mapped[int] = mapped_column(default=0, index=True) # number of tokens
    count: Mapped[int] = mapped_column(default=1) # count how many times the sentence was found in the dataset

    def __repr__(self) -> str:
        return (f"<Sentence(sentence_md5={self.sentence_md5})>")

class ProcessedParagraph(Base):

    __tablename__ = 'processed_articles'

    id: Mapped[int] = mapped_column(primary_key=True)
    md5: Mapped[str] = mapped_column(index=True, nullable=False, unique=True) # hash value of article to determine duplicates
    text: Mapped[str] = mapped_column(nullable=False) # actual sentence
    n_tokens: Mapped[int] = mapped_column(default=0, index=True) # number of tokens
    count: Mapped[int] = mapped_column(default=1) # count how many times the article was found in the dataset

    def __repr__(self) -> str:
        return (f"<ProcessedArticle(md5={self.md5})>")