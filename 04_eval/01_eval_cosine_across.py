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

import itertools
import random
from pathlib import Path

from scipy.stats import pearsonr
import numpy as np
import fasttext

from evaluation_data.cues import CUES
from utils.datamodel import Model, AcrossCorrelationResults
from utils.sql import start_sqlsession
from utils.similarity import average_cosine_distance

p = Path.cwd()

session, engine = start_sqlsession()



if __name__ == '__main__':

    print('Across Correlations')

    # Get models with different parameters
    parameter_families = session.query(Model.parameter_string).all()
    parameter_families = set([p[0] for p in parameter_families])

    model_combinations = list(itertools.combinations(parameter_families, 2))

    print('Model combinations:', len(model_combinations))

    for combination in model_combinations:

        models_a_meta = session.query(Model).where(
            Model.parameter_string == combination[0]).all()
        models_b_meta = session.query(Model).where(
            Model.parameter_string == combination[1]).all()

        print(
            f'Evaluating combination: {combination[0]} and {combination[1]}')
        
        models_a = [fasttext.load_model(m.model_path + '.bin') for m in models_a_meta]
        models_b = [fasttext.load_model(m.model_path + '.bin') for m in models_b_meta]
        # delete previous result
        old_result = session.query(AcrossCorrelationResults).filter_by(model_a_family=combination[0],
                                                                       model_b_family=combination[1]).first()
        if old_result:
            print('Found old result, deleting...')
            session.delete(old_result)
            session.commit()

        old_result = session.query(AcrossCorrelationResults).filter_by(model_b_family=combination[0],
                                                                       model_a_family=combination[1]).first()
        if old_result:
            print('Found old result, deleting...')
            session.delete(old_result)
            session.commit()

        # ensure that we only have words that both models share
        # get intersection of both vocabularies
        shared_vocabulary = set(models_a[0].words)
        for m in models_a:
            shared_vocabulary = shared_vocabulary & set(m.words)
        for m in models_b:
            shared_vocabulary = shared_vocabulary & set(m.words)

        words_random = random.sample(shared_vocabulary, 100)

        d_a = average_cosine_distance(models_a, words_random,
                                        vocabulary=shared_vocabulary)
        
        d_b = average_cosine_distance(models_b, words_random,
                                        vocabulary=shared_vocabulary)

        corr_random = [pearsonr(d_a[i], d_b[i])[0] for i in range(len(words_random))]
        corr_random = np.array(corr_random)

        results = AcrossCorrelationResults(model_a_family=combination[0], 
                                           model_b_family=combination[1],
                                           cues='random',
                                           correlation=corr_random.mean(),
                                           correlation_sd=corr_random.std(),
                                           correlation_type="Pearson's Rho")
        session.add(results)
        session.commit()

        for cue, wordlist in CUES.items():

            shared_wordlist = set(wordlist) & shared_vocabulary

            d_a = average_cosine_distance(models_a, shared_wordlist,
                                        vocabulary=shared_vocabulary)
        
            d_b = average_cosine_distance(models_b, shared_wordlist,
                                            vocabulary=shared_vocabulary)

            corr_cues = [pearsonr(d_a[i], d_b[i])[0] for i in range(len(shared_wordlist))]
            corr_cues = np.array(corr_cues)

            results = AcrossCorrelationResults(model_a_family=combination[0], 
                                            model_b_family=combination[1],
                                            cues=cue,
                                            correlation=corr_cues.mean(),
                                            correlation_sd=corr_cues.std(),
                                            correlation_type="Pearson's Rho")
            session.add(results)
            session.commit()

    session.close()