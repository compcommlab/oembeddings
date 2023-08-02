"""
    Based on replication files for the publication:
    Rodriguez, P. L., & Spirling, A. (2022). Word Embeddings: 
        What Works, What Doesn't, and How to Tell the Difference for Applied Research. 
        In The Journal of Politics (Vol. 84, Issue 1, pp. 101-115). 
        https://doi.org/10.1086/715162

    R implementation by Rodriguez & Spirling available here:
        https://github.com/prodriguezsosa/EmbeddingsPaperReplication

    We have two evaluations here:
        1. How stable are the embeddings with each initialization?
            (Compare each model with the same parameters with each other)
        2. How stable are the embeddings across initializations?
            (Compare each parameter setting with each other)

    We use a selection of words (cues) to compare the models with each other.
    One pre-defined set of words that we are interested in (politics), and one
    random set of words (we draw 100 random words from the vocabularly list of the model)
            
    We calculate the cosine distance for each word embedding against every other 
    word embedding in the model.

    For 1., we then take the distance measures and pairwise compare them by calculating
    Pearson's Rho. Next, we take the mean and standard deviation of all pairwise combinations.

    For 2., we take all cosine distances from models with the same initialisation 
    (i.e., one set of parameters) and calculate their mean. This means, we get one representation
    for the entire "model family". We then use this mean representation to pairwise compare it
    to initializations with different parameters.

    In 1. all models should have the same vocabulary, but 2. not. 
    Therefore, we need to account for that.

"""

import sys
sys.path.append('.')

import random
from pathlib import Path
import fasttext
import sqlalchemy

from evaluation_data.cues import CUES
from utils.datamodel import Model, WithinCorrelationResults
from utils.sql import start_sqlsession
from utils.similarity import cosine_distance, calc_correlation

p = Path.cwd()

session, engine = start_sqlsession()


if __name__ == '__main__':

    print('Within Correlations')

    # Get models from same family

    sql_query_string = """
        select A.id as model_a, B.id as model_b
        from models A, models B
        where A.id < B.id AND
            A.parameter_string = B.parameter_string
    """

    with engine.connect() as con:
        model_combinations = con.execute(
            sqlalchemy.text(sql_query_string)).all()

    print('Model combinations:', len(model_combinations))

    for combination in model_combinations:

        model_a_meta = session.query(Model).where(
            Model.id == combination[0]).first()
        model_b_meta = session.query(Model).where(
            Model.id == combination[1]).first()

        # delete previous result
        old_result = session.query(WithinCorrelationResults).filter_by(model_a_id=model_a_meta.id,
                                                                       model_b_id=model_b_meta.id).first()
        if old_result:
            print('Found old result, deleting...')
            session.delete(old_result)
            session.commit()

        old_result = session.query(WithinCorrelationResults).filter_by(model_b_id=model_a_meta.id,
                                                                       model_a_id=model_b_meta.id).first()
        if old_result:
            print('Found old result, deleting...')
            session.delete(old_result)
            session.commit()


        model_a = fasttext.load_model(model_a_meta.model_path + '.bin')
        model_b = fasttext.load_model(model_b_meta.model_path + '.bin')

        shared_vocabulary = set(model_a.words) & set(model_b.words)

        words_random = random.sample(shared_vocabulary, 100)

        corr_random = calc_correlation(model_a, model_b,
                                       words_random, vocabulary=shared_vocabulary)

        results = WithinCorrelationResults(model_a_id=model_a_meta.id,
                                           model_b_id=model_b_meta.id,
                                           parameter_string=model_a_meta.parameter_string,
                                           cues='random',
                                           correlation=corr_random.mean(),
                                           correlation_sd=corr_random.std(),
                                           correlation_type="Pearson's Rho")
        session.add(results)
        session.commit()

        for cue, wordlist in CUES.items():

            shared_wordlist = set(wordlist) & shared_vocabulary
            corr = calc_correlation(model_a, model_b,
                                    shared_wordlist, vocabulary=shared_vocabulary)

            results = WithinCorrelationResults(model_a_id=model_a_meta.id,
                                               model_b_id=model_b_meta.id,
                                               parameter_string=model_a_meta.parameter_string,
                                               cues=cue,
                                               correlation=corr.mean(),
                                               correlation_sd=corr.std(),
                                               correlation_type="Pearson's Rho")

            session.add(results)
            session.commit()


    session.close()