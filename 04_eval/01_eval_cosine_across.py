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
import typing
import random
from pathlib import Path
import json
from multiprocessing import Pool
from argparse import ArgumentParser

from scipy.stats import pearsonr
import numpy as np
import fasttext

# Supress Fasttext warnings when loading a model
import warnings
warnings.filterwarnings('ignore')

from evaluation_data.cues import CUES
from utils.similarity import average_cosine_distance
from utils.misc import get_data_dir

p = Path.cwd()

def load_model(model_path: str) -> fasttext.FastText._FastText:
    """ Wrapper for Multiprocessing """
    if not model_path.endswith('.bin'):
        model_path = model_path + '.bin'
    return fasttext.load_model(model_path)

def compare_model_groups(models: typing.Tuple[Path]) -> typing.List[dict]:
    results = []

    models_a_meta = [json.load(m.open()) for m in models[0].glob('*.json')]
    models_b_meta = [json.load(m.open()) for m in models[1].glob('*.json')]

    with Pool(input_args.threads) as pool:
        models_a = pool.map(load_model, [m['model_path'] for m in models_a_meta])
    
    with Pool(input_args.threads) as pool:
        models_b = pool.map(load_model, [m['model_path'] for m in models_b_meta])
    
    # models_a = [fasttext.load_model(m["model_path"] + '.bin') for m in models_a_meta]
    # models_b = [fasttext.load_model(m["model_path"] + '.bin') for m in models_b_meta]

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

    random_results = dict(model_a_family=models[0].name, 
                    model_b_family=models[1].name,
                    cues='random',
                    correlation=corr_random.mean(),
                    correlation_sd=corr_random.std(),
                    correlation_type="Pearson's Rho")
    
    results.append(random_results)
    
    for cue, wordlist in CUES.items():

        shared_wordlist = set(wordlist) & shared_vocabulary

        d_a = average_cosine_distance(models_a, shared_wordlist,
                                    vocabulary=shared_vocabulary)
    
        d_b = average_cosine_distance(models_b, shared_wordlist,
                                        vocabulary=shared_vocabulary)

        corr_cues = [pearsonr(d_a[i], d_b[i])[0] for i in range(len(shared_wordlist))]
        corr_cues = np.array(corr_cues)

        cue_results = dict(model_a_family=models[0].name, 
                        model_b_family=models[1].name,
                        cues=cue,
                        correlation=corr_cues.mean(),
                        correlation_sd=corr_cues.std(),
                        correlation_type="Pearson's Rho")
        
        results.append(cue_results)

        return results

if __name__ == '__main__':

    arg_parser = ArgumentParser(description="Evaluate correlations across different parameter settings")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag')
    arg_parser.add_argument('--threads', type=int, default=10, help='Number of parallel processes (default: 12)')
    input_args = arg_parser.parse_args()

    print('Across Correlations')

    results_dir = p / 'evaluation_results' / 'across_correlations'
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    results_name = results_dir / 'results.json'

    model_dir = get_data_dir()

    # Get different kinds of parameter settings
    parameter_groups = [d for d in model_dir.glob('tmp_models/*') if d.is_dir()]

    lowercase_groups = [g for g in parameter_groups if 'lower' in g.name]
    parameter_groups = list(set(parameter_groups) - set(lowercase_groups))

    model_combinations = list(itertools.combinations(parameter_groups, 2))
    model_combinations_lowercase = list(itertools.combinations(lowercase_groups, 2))

    print('Model combinations:', len(model_combinations) + len(model_combinations_lowercase))

    # with Pool(input_args.threads) as pool:
    #     results = pool.map(compare_model_groups, model_combinations)

    # results = list(itertools.chain(*results))
    
    results = []

    for combination in model_combinations:
        print('Combination', combination)
        results += compare_model_groups(combination)


    # with Pool(input_args.threads) as pool:
    #     results_lower = pool.map(compare_model_groups, model_combinations_lowercase)

    # results_lower = list(itertools.chain(*results_lower))

    results_lower = []
    for combination in model_combinations_lowercase:
        print('Combination', combination)
        results_lower += compare_model_groups(combination)

    results += results_lower

    with open(results_name, 'w') as f:
        json.dump(results, f)

    
 