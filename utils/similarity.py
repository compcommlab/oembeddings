import typing
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial import distance
import fasttext

def cosine_distance(model: fasttext.FastText._FastText,
                    cues: typing.Iterable[str],
                    vocabulary: typing.Iterable = None,
                    normalize=True) -> np.ndarray:
    """ 
        Given a list of cues (words), this function calculates the 
        cosine distance for each cue word against every word in the 
        model's vocabulary list.
    """
    
    assert len(cues) > 0, 'List of cues is empty!'

    v = np.array([model.get_word_vector(cue) for cue in cues])
    if vocabulary:
        v_vocab = np.array([model.get_word_vector(word)
                           for word in vocabulary])
    else:
        v_vocab = model.get_output_matrix()
    d = distance.cdist(v, v_vocab, metric="cosine")
    if normalize:
        # normalize array
        d = d / np.sqrt(np.sum(d**2))
    return d


def average_cosine_distance(models: typing.List[fasttext.FastText._FastText],
                            cues: typing.Iterable[str],
                            vocabulary: typing.Iterable = None) -> np.ndarray:
    """ 
        Given a list of models, this returns the mean cosine distance for
        each word in cues.

        Returns a numpy array with the dims: (len(cues), len(model.words)).
        E.g, 12 words and vocabulary of 2000 returns an array (12, 2000)

    """

    assert len(cues) > 0, 'List of cues is empty!'

    cos_sim = [cosine_distance(model, cues, vocabulary=vocabulary) for model in models]
    return np.mean(cos_sim, axis=0)


def calc_correlation(model1: fasttext.FastText._FastText,
                     model2: fasttext.FastText._FastText,
                     cues: typing.Iterable[str],
                     vocabulary: typing.Iterable = None) -> np.ndarray:
    """ 

    Given a list of cue words, this function calculates the corrlations
    between two models.

    First, calculate the cosine distance between every cue word 
    and each word in a model's vocab (pairwise). Yielding two arrays
    of distances for each model.
    Next, calculate the correlation between both distance arrays.

    Returns an array with the same length as `cues` (correlations)
    """
    d1 = cosine_distance(model1, cues, vocabulary=vocabulary)
    d2 = cosine_distance(model2, cues, vocabulary=vocabulary)
    correlation = [pearsonr(d1[i], d2[i])[0] for i in range(len(cues))]
    return np.array(correlation)

