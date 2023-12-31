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

class Model(Base):

    """ We store model details with training parameters here """

    __tablename__ = 'models'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(index=True, nullable=False)
    training_corpus: Mapped[str] = mapped_column(index=True, nullable=False)
    model_type: Mapped[str] = mapped_column(nullable=False) # cbow or skipgram
    learning_rate: Mapped[Optional[float]]
    epochs: Mapped[Optional[int]]
    word_ngrams: Mapped[Optional[int]] # fasttext parameter: wordNgrams
    loss_function: Mapped[Optional[str]] # type of loss function used
    min_count: Mapped[Optional[int]] # word minimum occurences
    window_size: Mapped[Optional[int]]
    dimensions: Mapped[Optional[int]]
    vocab_size: Mapped[Optional[int]]
    computation_time: Mapped[Optional[float]] # time in seconds
    avg_loss: Mapped[Optional[float]]
    model_path: Mapped[Optional[str]] # path to model file
    parameter_string: Mapped[Optional[str]] # string representation of all parameters for quick grouping
    created_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(), server_default=func.now()
    ) 

    def __repr__(self) -> str:
        return f'<Model: {self.name}>'

class ModelTrainingProgress(Base):

    """" Save the logs for model training progress here """

    __tablename__ = 'model_training_progress'

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    progress: Mapped[Optional[float]]
    loss: Mapped[Optional[float]]
    learning_rate: Mapped[Optional[float]]
    words_sec_thread: Mapped[Optional[float]] # type of loss function used

class Evaluation(Base):

    """ Keeps track on evaluation results """

    __tablename__ = 'evaluations'

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    corpus: Mapped[str]
    task: Mapped[str]
    f1: Mapped[float]
    precision: Mapped[float]
    recall: Mapped[float]

class WithinCorrelationResults(Base):

    """ Stores results for correlation tests within the same parameter settings """

    __tablename__ = 'within_correlation_results'

    id: Mapped[int] = mapped_column(primary_key=True)
    model_a_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    model_b_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    parameter_string: Mapped[str]
    cues: Mapped[str] # what kind of cues were used (e.g., "random", "politics")
    correlation: Mapped[float] # value for correlation between model a and b
    correlation_sd: Mapped[float] # standard deviation
    correlation_type: Mapped[str] # kind of correlation metric (e.g, "Pearson's Rho")

class AcrossCorrelationResults(Base):

    """ Stores results for correlation tests across different parameter settigns """

    __tablename__ = 'across_correlation_results'

    id: Mapped[int] = mapped_column(primary_key=True)
    model_a_family: Mapped[str]
    model_b_family: Mapped[str]
    cues: Mapped[str] # what kind of cues were used (e.g., "random", "politics")
    correlation: Mapped[float] # value for correlation between parameters a and b
    correlation_sd: Mapped[float]
    correlation_type: Mapped[str] # kind of correlation metric (e.g, "Pearson's Rho")


class SyntacticSemanticEvaluation(Base):

    """ Keeps track on evaluation results """

    __tablename__ = 'syntactic_semantic_evaluation'

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    task_group: Mapped[str] # Syntactic or Semantic
    task: Mapped[str] # nouns; adjectives; doesn't fit; etc
    correct: Mapped[int] # count how many are correct
    top_n: Mapped[Optional[int]] # was answer correct if taking top N words?
    n: Mapped[Optional[int]] # N for top_n
    coverage: Mapped[int] # total questions that could be answered (ie. model "knows" the words)
    total_questions: Mapped[int] # total number of questions
    duration: Mapped[float]