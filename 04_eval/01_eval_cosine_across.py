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
    print('Loading model', model_path)
    return fasttext.load_model(model_path)

def compare_model_groups(models: typing.Tuple[Path],
                         lowercase: bool = False) -> typing.List[dict]:
    results = []

    models_a_meta = [json.load(m.open()) for m in models[0].glob('*.json')]
    models_b_meta = [json.load(m.open()) for m in models[1].glob('*.json')]

    print('Loading Model Group', models[0])
    print('Number of models in Group', len(models_a_meta))
    models_a = [fasttext.load_model(m["model_path"] + '.bin') for m in models_a_meta]
    
    print('Loading Model Group', models[1])
    print('Number of models in Group', len(models_b_meta))
    models_b = [fasttext.load_model(m["model_path"] + '.bin') for m in models_b_meta]
    
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
        if lowercase:
            wordlist = [w.lower() for w in wordlist]

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

    arg_parser = ArgumentParser(description=("Evaluate correlations across different parameter settings. "
                                            "If no further arguments are provided then the script will automatically "
                                            "scan the tmp_models directory and calculate all possible combinations "
                                            "across different model families.\n"
                                            "If you specify parameters --model_a and --model_b, the script will only "
                                            "calculate the across correlations between those two groups"))
    arg_parser.add_argument('--debug', action='store_true', help='Debug flag: dry run the script (do not actually run the calculations)')
    arg_parser.add_argument('--threads', type=int, default=10, help='Number of parallel processes (default: 12)')
    arg_parser.add_argument('--model_a', type=str, help='Specify a directory with Model A group')
    arg_parser.add_argument('--model_b', type=str, help='Specify a directory with Model B group')
    input_args = arg_parser.parse_args()

    print('Across Correlations')

    results_dir = p / 'evaluation_results' / 'across_correlations'
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    model_dir = get_data_dir()

    if input_args.model_a and input_args.model_b:
        model_a_dir = Path(input_args.model_a)
        assert model_a_dir.is_dir(), f'Model A is not a directory! Model A path: {model_a_dir}'
        model_b_dir = Path(input_args.model_b)
        assert model_b_dir.is_dir(), f'Model B is not a directory! Model B path: {model_b_dir}'
        assert model_a_dir.name != model_b_dir.name, f'Both model families are the same! Can only calculate across correlation between different model families'

        print('Comparing models:', model_a_dir.name, 'and', model_b_dir.name)

        results_name = results_dir / f'{model_a_dir.name}_{model_b_dir.name}.json'
        lowercase = "lower" in model_a_dir.name
        try:
            if not input_args.debug:
                results = compare_model_groups((model_a_dir, model_b_dir), lowercase=lowercase)
            else:
                results = []
            with open(results_name, 'w') as f:
                json.dump(results, f)
        except Exception as e:
            print('Could not calculate across correlations')
            print(e)
    else:
        print('Comparing all possible combinations ...')
        results_name = results_dir / f'results.json'

        # Get different kinds of parameter settings
        parameter_groups = [d for d in model_dir.glob('tmp_models/*') if d.is_dir()]

        lowercase_groups = [g for g in parameter_groups if 'lower' in g.name]
        parameter_groups = list(set(parameter_groups) - set(lowercase_groups))

        model_combinations = list(itertools.combinations(parameter_groups, 2))
        model_combinations_lowercase = list(itertools.combinations(lowercase_groups, 2))

        print('Model combinations:', len(model_combinations) + len(model_combinations_lowercase))

        results = []

        for combination in model_combinations:
            print('Combination', combination)
            try:
                if not input_args.debug:
                    results += compare_model_groups(combination)
            except Exception as e:
                print('Could not calculate across correlations')
                print(e)

        results_lower = []
        for combination in model_combinations_lowercase:
            print('Combination', combination)
            try:
                if not input_args.debug:
                    results_lower += compare_model_groups(combination, lowercase=True)
            except Exception as e:
                print('Could not calculate across correlations')
                print(e)


        results += results_lower

        with open(results_name, 'w') as f:
            json.dump(results, f)
