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

import fasttext
import random
import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr

# TODO: 
#    - load models via sql db
#    - automatically compare all models with each other
#    - store output in sql db


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



model1 = fasttext.load_model("models/beliebte_annamaria_lr0.05_epochs3_mincount50_ws12_dims300.bin")
model2 = fasttext.load_model("models/reicher_alper.bin")

assert len(model1.words) == len(model2.words)

words_politics = ["Demokratie", "Freiheit", "Gleichheit", "Gerechtigkeit", "Einwanderung", "Schwangerschaftsabbruch", "Sozialstaat", "Steuern", "ÖVP", "SPÖ", "FPÖ", "Grüne", "NEOS"]
words_random = random.sample(model1.words, 100)

corr_politics = calc_correlation(model1, model2, words_politics)
corr_random = calc_correlation(model1, model2, words_random)

# 2. compare different initializations
avg_sim_vectors1 = average_cosine_distance([model1, model2], words_politics)
# avg_sim_vectors2 = average_cosine_distance([model3, model4], words_politics)

# correlation = [pearsonr(avg_sim_vectors1[i], avg_sim_vectors2[i])[0] for i in range(len(words_politics))]