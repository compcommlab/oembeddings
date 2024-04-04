# script to create test-sets for evaluation of word embeddings
# needs the data & src folders to create the test questions
# saves logged results in additional file
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
# 
# Available under MIT License, Copyright (c) 2015 Andreas Müller. 
# For original source and full license details, see:
# https://github.com/devmount/GermanWordEmbeddings
#
# Contributors:
#  Michael Egger <michael.egger@tsn.at>
# Adapted by:
#  Jana Bernhard 
#
# @example: python evaluation.py model/my.model -u -t 10
# where model/my.model is the path to the model 
# model needs to be .vec file 
# to get .vec file from .bin file see the code here: 
#   https://stackoverflow.com/questions/67679162/how-can-i-get-a-vec-file-from-a-bin-file 
#   https://stackoverflow.com/questions/58337469/how-to-save-fasttext-model-in-vec-format 

import sys
sys.path.append('.')
import typing

import gensim
import random
import time
import argparse
import logging
from pathlib import Path
import json
from utils.misc import get_data_dir

# gensim already uses all cores
# should make multiprocessing redundant
# from multiprocessing import Pool

""" Set up pathing """

p = Path.cwd()

RESULTS_DIR = p / "evaluation_results" / "semantic_syntactic"
MOST_SIMILAR_DIR = RESULTS_DIR / 'mostsimilar'
BEST_MATCH_DIR = RESULTS_DIR / 'bestmatch'
OPPOSITE_DIR = RESULTS_DIR / 'opposite'
WORD_INTRUSION_DIR = RESULTS_DIR / 'wordintrusion'


TARGET_SYN = p / 'evaluation_data' / 'devmount' / 'syntactic.questions'
TARGET_SEM_OP = p / 'evaluation_data' / 'devmount' / 'semantic_op.questions'
TARGET_SEM_BM = p / 'evaluation_data' / 'devmount' / 'semantic_bm.questions'
TARGET_SEM_DF = p / 'evaluation_data' / 'devmount' / 'semantic_df.questions'
SRC_NOUNS = p / 'evaluation_data' / 'devmount' / 'nouns.txt'
SRC_ADJECTIVES = p / 'evaluation_data' / 'devmount' / 'adjectives.txt'
SRC_VERBS = p / 'evaluation_data' / 'devmount' / 'verbs.txt'
SRC_BESTMATCH = p / 'evaluation_data' / 'devmount' / 'bestmatch.txt'
SRC_DOESNTFIT = p / 'evaluation_data' / 'devmount' / 'doesntfit.txt'
SRC_OPPOSITE = p / 'evaluation_data' / 'devmount' / 'opposite.txt'
PATTERN_SYN = [
    ('nouns', 'SI/PL', SRC_NOUNS, 0, 1),
    ('nouns', 'PL/SI', SRC_NOUNS, 1, 0),
    ('adjectives', 'GR/KOM', SRC_ADJECTIVES, 0, 1),
    ('adjectives', 'KOM/GR', SRC_ADJECTIVES, 1, 0),
    ('adjectives', 'GR/SUP', SRC_ADJECTIVES, 0, 2),
    ('adjectives', 'SUP/GR', SRC_ADJECTIVES, 2, 0),
    ('adjectives', 'KOM/SUP', SRC_ADJECTIVES, 1, 2),
    ('adjectives', 'SUP/KOM', SRC_ADJECTIVES, 2, 1),
    ('verbs (pres)', 'INF/1SP', SRC_VERBS, 0, 1),
    ('verbs (pres)', '1SP/INF', SRC_VERBS, 1, 0),
    ('verbs (pres)', 'INF/2PP', SRC_VERBS, 0, 2),
    ('verbs (pres)', '2PP/INF', SRC_VERBS, 2, 0),
    ('verbs (pres)', '1SP/2PP', SRC_VERBS, 1, 2),
    ('verbs (pres)', '2PP/1SP', SRC_VERBS, 2, 1),
    ('verbs (past)', 'INF/3SV', SRC_VERBS, 0, 3),
    ('verbs (past)', '3SV/INF', SRC_VERBS, 3, 0),
    ('verbs (past)', 'INF/3PV', SRC_VERBS, 0, 4),
    ('verbs (past)', '3PV/INF', SRC_VERBS, 4, 0),
    ('verbs (past)', '3SV/3PV', SRC_VERBS, 3, 4),
    ('verbs (past)', '3PV/3SV', SRC_VERBS, 4, 3)
]

def replace_umlauts(text: str) -> str:
    """
    Replaces german umlauts and sharp s in given text.

    :param text: text as str
    :return: manipulated text as str
    """
    res = text
    res = res.replace('ä', 'ae')
    res = res.replace('ö', 'oe')
    res = res.replace('ü', 'ue')
    res = res.replace('Ä', 'Ae')
    res = res.replace('Ö', 'Oe')
    res = res.replace('Ü', 'Ue')
    res = res.replace('ß', 'ss')
    return res


def create_syntactic_testset() -> None:
    """
    Creates syntactic test set and writes it into a file.

    :return: None
    """
    if args.umlauts:
        u = open(str(TARGET_SYN) + '.nouml', 'w')
    with open(TARGET_SYN, 'w') as t:
        for label, short, src, index1, index2 in PATTERN_SYN:
            t.write(': {}: {}\n'.format(label, short))
            if args.umlauts:
                u.write(': {}: {}\n'.format(label, short))
            for q in create_questions(src, index1, index2):
                t.write(q + '\n')
                if args.umlauts:
                    u.write(replace_umlauts(q) + '\n')
            logging.info('created pattern ' + short)
    if args.umlauts:
        u.close()

    # Create lowercase version of dataset
    with open(str(TARGET_SYN) + '.lower', 'w') as t:
        for label, short, src, index1, index2 in PATTERN_SYN:
            t.write(': {}: {}\n'.format(label, short))
            for q in create_questions(src, index1, index2):
                t.write(q.lower() + '\n')
            logging.info('created lowercase pattern ' + short)


def create_semantic_testset() -> None:
    """
    Creates semantic test set and writes it into a file.

    :return: None
    """
    # opposite
    with open(TARGET_SEM_OP, 'w') as t:
        for q in create_questions(SRC_OPPOSITE, combinate=10):
            t.write(q + '\n')
            if args.umlauts:
                with open(TARGET_SEM_OP + '.nouml', 'w') as u:
                    u.write(replace_umlauts(q) + '\n')
        logging.info('created opposite questions')
    
    with open(str(TARGET_SEM_OP) + '.lower', 'w') as t:
        for q in create_questions(SRC_OPPOSITE, combinate=10):
            t.write(q.lower() + '\n')
        logging.info('created opposite questions (lowercase)')

    # best match
    with open(TARGET_SEM_BM, 'w') as t:
        groups = open(SRC_BESTMATCH).read().split(':')
        groups.pop(0)  # remove first empty group
        for group in groups:
            questions = group.splitlines()
            _ = questions.pop(0)
            while questions:
                for i in range(1, len(questions)):
                    question = questions[0].split('-') + questions[i].split('-')
                    t.write(' '.join(question) + '\n')
                    if args.umlauts:
                        with open(TARGET_SEM_BM + '.nouml', 'w') as u:
                            u.write(replace_umlauts(' '.join(question)) + '\n')
                questions.pop(0)
        logging.info('created best-match questions')

    # best match (lowercase version)
    with open(str(TARGET_SEM_BM) + '.lower', 'w') as t:
        groups = open(SRC_BESTMATCH).read().split(':')
        groups.pop(0)  # remove first empty group
        for group in groups:
            questions = group.splitlines()
            _ = questions.pop(0)
            while questions:
                for i in range(1, len(questions)):
                    question = questions[0].split('-') + questions[i].split('-')
                    question = [q.lower() for q in question]
                    t.write(' '.join(question) + '\n')
                questions.pop(0)
        logging.info('created best-match questions (lowercase)')

    # doesn't fit
    with open(TARGET_SEM_DF, 'w') as t:
        for line in open(SRC_DOESNTFIT):
            words = line.split()
            for wrongword in words[-1].split('-'):
                question = ' '.join(words[:3] + [wrongword])
                t.write(question + '\n')
                if args.umlauts:
                    with open(TARGET_SEM_DF + '.nouml', 'w') as u:
                        u.write(replace_umlauts(question) + '\n')
        logging.info('created doesn\'t-fit questions')

    # doesn't fit (lowercase version)
    with open(str(TARGET_SEM_DF) + '.lower', 'w') as t:
        for line in open(SRC_DOESNTFIT):
            words = line.split()
            for wrongword in words[-1].split('-'):
                question = ' '.join(words[:3] + [wrongword])
                t.write(question.lower() + '\n')
        logging.info('created doesn\'t-fit questions (lowercase)')


def create_questions(src: Path, index1=0, index2=1, combinate=5) -> typing.List[str]:
    """
    Creates single questions from given source.

    :param src: source file to load words from
    :param index1: index of first word in a line to focus on
    :param index2: index of second word in a line to focus on
    :param combinate: combinate number of combinations with random other lines
    :return: list of question words
    """
    # get source content
    with open(src) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    questions = []

    for line in content:
        for i in range(0, combinate):
            # get current word pair
            question = list(line.split('-')[i] for i in [index1, index2])
            # get random word pair that is not the current
            random_line = random.choice(list(set(content) - {line}))
            random_word = list(random_line.split('-')[i] for i in [index1, index2])
            # merge both word pairs to one question
            question.extend(random_word)
            questions.append(' '.join(question))
    return questions


def test_most_similar(model: gensim.models.KeyedVectors, 
                      src: Path, 
                      label='most similar', 
                      topn=10) -> dict:
    """
    Tests given model to most similar word.

    :param model: model to test
    :param src: source file to load words from
    :param label: label to print current test case
    :param topn: number of top matches
    :return: dict with results
    """
    num_lines = sum(1 for _ in open(src))
    num_questions = 0
    num_right = 0
    num_topn = 0
    t = time.time()
    # get questions
    with open(src) as f:
        questions = f.readlines()
        questions = [x.strip() for x in questions]
    # test each question
    for question in questions:
        words = question.split()
        # check if all words exist in vocabulary
        if all(x in model.index_to_key for x in words):
            num_questions += 1
            best_matches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
            # best match
            if words[3] in best_matches[0]:
                num_right += 1
            # topn match
            for match in best_matches[:topn]:
                if words[3] in match:
                    num_topn += 1
                    break
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions > 0 else 0.0
    topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions > 0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines > 0 else 0.0
    duration = time.time() - t
    # log result
    logging.info(label + ' correct:  {0}% ({1}/{2})'.format(correct_matches, num_right, num_questions))
    logging.info(label + ' top {0}:   {1}% ({2}/{3})'.format(topn, topn_matches, num_topn, num_questions))
    logging.info(label + ' coverage: {0}% ({1}/{2})'.format(coverage, num_questions, num_lines))
    result = {'task_group': 'most_similar',
                      'task': label,
                      'correct': num_right,
                      'top_n': num_topn,
                      'n': topn,
                      'coverage': num_questions,
                      'total_questions': num_lines,
                      'duration': duration}
    return result


def test_most_similar_groups(model: gensim.models.KeyedVectors, 
                             src: Path, 
                             topn=10) -> typing.List[dict]:
    """
    Tests given model to most similar word.

    :param model: model to test
    :param src: source file to load words from
    :param topn: number of top matches
    :return: A list of dicts, where each dict is an evaluation result
    """
    num_lines = 0
    num_questions = 0
    num_right = 0
    num_topn = 0
    total_duration = 0
    results = []
    # test each group
    with open(src) as groups_fp:
        groups = groups_fp.read().split('\n: ')
        for group in groups:
            questions = group.splitlines()
            label = questions.pop(0)
            label = label[2:] if label.startswith(': ') else label  # handle first group
            num_group_lines = len(questions)
            num_group_questions = 0
            num_group_right = 0
            num_group_topn = 0
            # time the duration
            t = time.time()
            # test each question of current group
            for question in questions:
                words = question.split()
                # check if all words exist in vocabulary
                if all(x in model.index_to_key for x in words):
                    num_group_questions += 1
                    best_matches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
                    # best match
                    if words[3] in best_matches[0]:
                        num_group_right += 1
                    # topn match
                    for match in best_matches[:topn]:
                        if words[3] in match:
                            num_group_topn += 1
                            break
            # calculate result
            correct_group_matches = round(num_group_right/float(num_group_questions)*100, 1) if num_group_questions > 0 else 0.0
            topn_group_matches = round(num_group_topn/float(num_group_questions)*100, 1) if num_group_questions > 0 else 0.0
            group_coverage = round(num_group_questions/float(num_group_lines)*100, 1) if num_group_lines > 0 else 0.0
            # log result
            duration = time.time() - t
            logging.info(label + ': {0}% ({1}/{2}), {3}% ({4}/{5}), {6}% ({7}/{8})'.format(
                correct_group_matches, # %-correct
                num_group_right, # int
                num_group_questions, # total questions asked
                topn_group_matches, # %-correct
                num_group_topn, # int
                num_group_questions, # total questions asked
                group_coverage, # %-covered
                num_group_questions, # total questions asked
                num_group_lines # total number of questions
            ))
            result = {'task_group': 'most_similar_groups',
                      'task': label,
                      'correct': num_group_right,
                      'top_n': num_group_topn,
                      'n': topn,
                      'coverage': num_group_questions,
                      'total_questions': num_group_lines,
                      'duration': duration}
            results.append(result)
            # total numbers
            num_lines += num_group_lines
            num_questions += num_group_questions
            num_right += num_group_right
            num_topn += num_group_topn
            total_duration += duration
        # calculate result
        correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions > 0 else 0.0
        topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions > 0 else 0.0
        coverage = round(num_questions/float(num_lines)*100, 1) if num_lines > 0 else 0.0
        result = {'task_group': 'most_similar_groups',
                      'task': "total",
                      'correct': num_right,
                      'top_n': num_topn,
                      'n': topn,
                      'coverage': num_questions,
                      'total_questions': num_lines,
                      'duration': total_duration}
        results.append(result)
        # log result
        logging.info('total correct:  {0}% ({1}/{2})'.format(correct_matches, num_right, num_questions))
        logging.info('total top {0}:   {1}% ({2}/{3})'.format(topn, topn_matches, num_topn, num_questions))
        logging.info('total coverage: {0}% ({1}/{2})'.format(coverage, num_questions, num_lines))
        return results


def test_doesnt_fit(model: gensim.models.KeyedVectors, 
                    src: Path) -> dict:
    """
    Tests given model to most not fitting word.

    :param model: model to test
    :param src: source file to load words from
    :return: dict with results
    """

    num_lines = sum(1 for _ in open(src))
    num_questions = 0
    num_right = 0
    t = time.time()
    # get questions
    with open(src) as f:
        questions = f.readlines()
        questions = [x.strip() for x in questions]
    # test each question
    for question in questions:
        words = question.split()
        # check if all words exist in vocabulary
        if all(x in model.index_to_key for x in words):
            num_questions += 1
            if model.doesnt_match(words) == words[3]:
                num_right += 1
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions > 0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines > 0 else 0.0
    duration = time.time() - t
    # log result
    logging.info('doesn\'t fit correct:  {0}% ({1}/{2})'.format(correct_matches, num_right, num_questions))
    logging.info('doesn\'t fit coverage: {0}% ({1}/{2})'.format(coverage, num_questions, num_lines))
    result = {'task_group': 'word intrusion',
                'task': "doesnt fit",
                'correct': num_right,
                'coverage': num_questions,
                'total_questions': num_lines,
                'duration': duration}
    return result

def evaluate_model(model_path: str, 
                   topn: int, 
                   umlauts = False,
                   lowercase = False,
                   meta: dict = None) -> None:

    if not model_path.endswith('.vec') and not Path(model_path).exists():
        model_path = model_path + '.vec'

    assert Path(model_path).exists(), f"Cannot find model at path {model_path}"

    if model_path.endswith('.vec'):
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
    else:
        model = gensim.models.KeyedVectors.load(model_path)

    # execute evaluation
    logging.info(f'Model: {model_path}')
    logging.info('> EVALUATING SYNTACTIC FEATURES')
    most_similar_results = test_most_similar_groups(model, str(TARGET_SYN) + '.lower' if lowercase else TARGET_SYN, args.topn)
    
    if meta:
        for r in most_similar_results:
            r['name'] = meta['name']
            r['parameter_string'] = meta['parameter_string']

    results_destination = MOST_SIMILAR_DIR / Path(model_path).name.replace('.vec', '_most_similar.json')
    with open(results_destination, "w") as f:
        json.dump(most_similar_results, f, indent=True)

    logging.info('> EVALUATING SEMANTIC FEATURES')
    result = test_most_similar(model, str(TARGET_SEM_OP) + '.lower' if lowercase else TARGET_SEM_OP, 'opposite', topn)
    if meta:
        result['name'] = meta['name']
        result['parameter_string'] = meta['parameter_string']

    results_destination = OPPOSITE_DIR / Path(model_path).name.replace('.vec', '_opposite.json')
    with open(results_destination, "w") as f:
        json.dump(result, f, indent=True)

    result = test_most_similar(model, str(TARGET_SEM_BM) + '.lower' if lowercase else TARGET_SEM_BM, 'best match', topn)
    if meta:
        result['name'] = meta['name']
        result['parameter_string'] = meta['parameter_string']

    results_destination = BEST_MATCH_DIR / Path(model_path).name.replace('.vec', '_best_match.json')
    with open(results_destination, "w") as f:
        json.dump(result, f, indent=True)

    result = test_doesnt_fit(model, str(TARGET_SEM_DF) + '.lower' if lowercase else TARGET_SEM_DF)
    if meta:
        result['name'] = meta['name']
        result['parameter_string'] = meta['parameter_string']

    results_destination = WORD_INTRUSION_DIR / Path(model_path).name.replace('.vec', '_word_intrusion.json')
    with open(results_destination, "w") as f:
        json.dump(result, f, indent=True)

    logging.info('------------------------------')


if __name__ == '__main__':
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)
    if not MOST_SIMILAR_DIR.exists():
        MOST_SIMILAR_DIR.mkdir(parents=True)
    if not OPPOSITE_DIR.exists():
        OPPOSITE_DIR.mkdir(parents=True)
    if not BEST_MATCH_DIR.exists():
        BEST_MATCH_DIR.mkdir(parents=True)
    if not WORD_INTRUSION_DIR.exists():
        WORD_INTRUSION_DIR.mkdir(parents=True)
    # configuration
    parser = argparse.ArgumentParser(description='Script for creating testsets and evaluating word vector models. If no "model" parameter is specified, the script will evaluate all models in the tmp_models directory.')
    parser.add_argument('-m', '--model', type=str, help='source file with trained model')
    parser.add_argument('-f', '--fresh', action='store_true', help='Get fresh results; delete old results')
    parser.add_argument('-u', '--umlauts', action='store_true', help='if set, create additional testsets with transformed umlauts and use them instead')
    parser.add_argument('-n', '--topn', type=int, default=10, help='check the top n result (correct answer under top n answeres)')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel processes (default: 1)')
    parser.add_argument('--debug', action='store_true', help='Debug flag')

    args, unknown = parser.parse_known_args()

    if args.model:
        logging.basicConfig(filename=args.model.strip() + '.result', format='%(asctime)s : %(message)s', level=logging.INFO)
    else:  
        logging.basicConfig(filename='syntactic_evaluation_results.log', format='%(asctime)s : %(message)s', level=logging.INFO)

    if args.debug:
        consoleHandler = logging.StreamHandler()
        logging.getLogger().addHandler(consoleHandler)

    # If question files do not exist, create them automatically
    if not all([TARGET_SYN.exists(), TARGET_SEM_OP.exists(), TARGET_SEM_BM.exists(), TARGET_SEM_DF.exists()]):
        assert SRC_ADJECTIVES.exists()
        assert SRC_NOUNS.exists()
        assert SRC_ADJECTIVES.exists()
        assert SRC_VERBS.exists()
        assert SRC_BESTMATCH.exists()
        assert SRC_DOESNTFIT.exists()
        assert SRC_OPPOSITE.exists()
        logging.info('> CREATING SYNTACTIC TESTSET')
        create_syntactic_testset()
        logging.info('> CREATING SEMANTIC TESTSET')
        create_semantic_testset()
    
    if args.model:
        evaluate_model(args.model, args.topn, umlauts=args.umlauts, lowercase= 'lower' in args.model or 'wiki' in args.model)
    else:
        model_dir = get_data_dir()
        models_meta = [json.load(f.open()) for f in model_dir.glob('tmp_models/*/*.json')]
        # with Pool(args.threads) as pool:
        #     for model in models_meta:
        #         r = pool.apply(evaluate_model, (model["model_path"], args.topn, ), kwds={'meta': model})
        for model in models_meta:
            evaluate_model(model['model_path'], args.topn, meta=model, lowercase = 'lower' in model['parameter_string'])