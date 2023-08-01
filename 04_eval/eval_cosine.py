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

import typing

from pathlib import Path
import sqlalchemy

import fasttext
import random
import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr

from utils.sql import start_sqlsession
from utils.datamodel import Model, WithinCorrelationResults, AcrossCorrelationResults

p = Path.cwd()

session, engine = start_sqlsession()

def cosine_distance(model: fasttext.FastText._FastText, 
                    cues: typing.List[str],
                    normalize=True) -> np.ndarray:
    """ 
        Given a list of cues (words), this function calculates the 
        cosine distance for each cue word against every word in the 
        model's vocabulary list.
    """
    v = np.array([model.get_word_vector(cue) for cue in cues])
    if normalize:
        # normalize array
        v = v / np.sqrt(np.sum(v**2))
    d = distance.cdist(v, model.get_output_matrix(), metric="cosine")
    return d


def average_cosine_distance(models: typing.List[fasttext.FastText._FastText], 
                              cues: typing.List[str]) -> np.ndarray:
    """ 
        Given a list of models, this returns the mean cosine distance for
        each word in cues.
        
        Returns a numpy array with the dims: (len(cues), len(model.words)).
        E.g, 12 words and vocabulary of 2000 returns an array (12, 2000)

    """
    cos_sim = [cosine_distance(model, cues) for model in models]
    return np.mean(cos_sim, axis=0)


def calc_correlation(model1: fasttext.FastText._FastText, 
                    model2: fasttext.FastText._FastText, 
                    cues: typing.List[str]) -> np.ndarray:
    """ 
    
    Given a list of cue words, this function calculates the corrlations
    between two models.

    First, calculate the cosine distance between every cue word 
    and each word in a model's vocab (pairwise). Yielding two arrays
    of distances for each model.
    Next, calculate the correlation between both distance arrays.

    Returns an array with the same length as `cues` (correlations)
    """
    d1 = cosine_distance(model1, cues)
    d2 = cosine_distance(model2, cues)
    correlation = [pearsonr(d1[i], d2[i])[0] for i in range(len(cues))]
    return np.array(correlation)


if __name__ == '__main__':

    # TODO: 
    #    - load models via sql db
    #    - automatically compare all models with each other
    #    - store output in sql db


    # Within Correlations

    # Get models from same family

    

    sql_query_string = """
    select A.id as model_a, B.id as model_b
    from models A, models B
    where A.id != B.id AND
            A.training_corpus = B.training_corpus AND
            A.model_type = B.model_type and
            A.learning_rate = B.learning_rate and 
            A.epochs = B.epochs AND
            A.word_ngrams = B.word_ngrams AND 
            A.min_count = B.min_count AND
            A.window_size = B.window_size AND 
            A.dimensions = B.dimensions
    """

    with engine.connect() as con:
        model_combinations = con.execute(sqlalchemy.text(sql_query_string)).all()

    print('Model combinations:', len(model_combinations))

    model_a_meta = session.query(Model).where(Model.id == model_combinations[0][0]).first()
    model_b_meta = session.query(Model).where(Model.id == model_combinations[0][1]).first()

    # build a string that identifies the model parameters
    # maybe put that into a separate function
    corpus = Path(model_a_meta.training_corpus).name.replace('.txt', '')
    parameter_string = f"{corpus}_epochs{model_a_meta.epochs}_lr{model_a_meta.learning_rate}_mincount{model_a_meta.min_count}_ws{model_a_meta.window_size}_dims{model_a_meta.dimensions}"

    model_a = fasttext.load_model(model_b_meta.model_path + '.bin')
    model_b = fasttext.load_model(model_b_meta.model_path + '.bin')

    # ensure that we only have words that both models share
    # get intersection of both vocabulary
    shared_vocabulary = set(model_a.words) & set(model_b.words)

    words_politics = ["Demokratie", "Gleichheit", "Gerechtigkeit",
                    "Einwanderung", "Pension", "Sozialstaat", "Bildung",
                    "Steuern", "ÖVP", "SPÖ", "FPÖ", "Grüne", "NEOS"]
    
    words_politics = set(words_politics) & shared_vocabulary

    words_random = random.sample(shared_vocabulary, 100)

    corr_politics = calc_correlation(model_a, model_b, words_politics)
    corr_random = calc_correlation(model_a, model_b, words_random)

    results = WithinCorrelationResults(model_a_id=model_a_meta.id, model_b_id=model_b_meta.id,
                                       cues='politics',
                                       correlation=corr_politics.mean(),
                                       correlation_type="Pearson's Rho")
    session.add(results)
    session.commit()

    # 2. compare different initializations
    avg_sim_vectors1 = average_cosine_distance([model_a, model_b], words_politics)
    # avg_sim_vectors2 = average_cosine_distance([model3, model4], words_politics)

    # correlation = [pearsonr(avg_sim_vectors1[i], avg_sim_vectors2[i])[0] for i in range(len(words_politics))]