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

import typing
import itertools
from multiprocessing import Pool
from argparse import ArgumentParser

import json
import random
from pathlib import Path
import fasttext

from evaluation_data.cues import CUES
from utils.similarity import calc_correlation

p = Path.cwd()


def compare_models(models: typing.Tuple[Path]) -> typing.List[dict]:
    results = []
    model_a_meta = json.load(models[0].open())
    model_b_meta = json.load(models[1].open())

    model_a = fasttext.load_model(model_a_meta["model_path"] + '.bin')
    model_b = fasttext.load_model(model_b_meta["model_path"] + '.bin')

    shared_vocabulary = set(model_a.words) & set(model_b.words)

    words_random = random.sample(shared_vocabulary, 100)

    corr_random = calc_correlation(model_a, model_b,
                                words_random, vocabulary=shared_vocabulary)

    results_random = dict(model_a_name=model_a_meta['name'],
                    model_b_name=model_b_meta['name'],
                    parameter_string=model_a_meta["parameter_string"],
                    cues='random',
                    correlation=corr_random.mean(),
                    correlation_sd=corr_random.std(),
                    correlation_type="Pearson's Rho")
    
    results.append(results_random)

    for cue, wordlist in CUES.items():

        if 'lower' in model_a_meta['parameter_string']:
            wordlist = [w.lower() for w in wordlist]

        shared_wordlist = set(wordlist) & shared_vocabulary
        corr = calc_correlation(model_a, model_b,
                                shared_wordlist, vocabulary=shared_vocabulary)

        cue_results = dict(model_a_name=model_a_meta['name'],
                    model_b_name=model_b_meta['name'],
                    parameter_string=model_a_meta["parameter_string"],
                    cues=cue,
                    correlation=corr.mean(),
                    correlation_sd=corr.std(),
                    correlation_type="Pearson's Rho")
        
        results.append(cue_results)
        return results



if __name__ == '__main__':

    arg_parser = ArgumentParser(description="Evaluate correlations within the same parameter settings")
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag')
    arg_parser.add_argument('--threads', type=int, default=12, help='Number of parallel processes (default: 12)')
    input_args = arg_parser.parse_args()

    print('Within Correlations')

    results_dir = p / 'evaluation_results' / 'within_correlations'
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    # Get different kinds of parameter settings
    parameter_groups = [d for d in p.glob('tmp_models/*') if d.is_dir()]
    for group in parameter_groups:
        print('Evaluating Group:', group.name)
        model_meta = [m for m in group.glob('*.json')]
        model_combinations = itertools.combinations(model_meta, 2)

        results_name = results_dir / f'{group.name}.json'

        with Pool(input_args.threads) as pool:
            group_results = pool.map(compare_models, model_combinations)
        
        group_results = list(itertools.chain(*group_results))

        with open(results_name, 'w') as f:
            json.dump(group_results, f)

