# -*- coding: utf-8 -*-
import os, sys

import logging

import time
from nltk.stem.porter import *

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )

    fh = logging.FileHandler(logfile)
    # ch = logging.StreamHandler()
    ch = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging

class Config():
    # Display progress logs on stdout
    def __init__(self, param):
        self.param = param
        self.logger = init_logging(param['log_path'])

    def __getitem__(self, key):
        return self.param[key]

    def __setitem__(self, key, val):
        self.param[key] = val

    def __contains__(self, key):
        return key in self.param

def load_config():
    param = {}
    param['seed']            = 154316847
    '''
    metadata and path
    '''
    param['root_path']  = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))

    param['task_name']       = 'utterance_type'
    param['timemark']        = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    param['experiment_mode'] = ['single_run', 'single_run_context_feature', 'normal_cv', 'feature_selection', 'leave_one_out', 'keep_one_only', 'reformulation_detection', 'task_boundary_detection', 'bad_case', 'print_important_features'][1]

    # two key feature settings
    param['context_set']     = ['current', 'next', 'last', 'all'][1]

    selected_feature_set_id  = 1
    param['feature_set']     = ['0.all', '1.basic', '2.lexical', '3.syntactic', '4.lda', '5.w2v', '6.d2v'][selected_feature_set_id]
    param['feature_set_number']  = [['1','2','3','4','5','6','7','8','9','10'], ['1','2','3'], ['4'], ['5','6','7'], ['8'], ['9'], ['10']][selected_feature_set_id]
    param['similarity_feature']  = False

    # context window
    param['utterance_names'] = ['last_user_utterance', 'last_system_utterance','current_user_utterance', 'next_system_utterance', 'next_user_utterance']
    if param['context_set']     == 'current':
        param['utterance_range'] = ['current_user_utterance']
    elif param['context_set']   == 'next':
        param['utterance_range'] = ['current_user_utterance', 'next_system_utterance','next_user_utterance']
    elif param['context_set']   == 'last':
        param['utterance_range'] = ['current_user_utterance', 'last_system_utterance','current_user_utterance']
    elif param['context_set']   == 'all':
        param['utterance_range'] = param['utterance_names']

    # param['experiment_name'] = '.'.join([param['task_name'], param['experiment_mode'], param['timemark']])
    param['experiment_name'] = '.'.join([param['timemark'], 'context=%s' % param['context_set'], 'feature=%s' % param['feature_set']])

    param['experiment_path'] = os.path.join(param['root_path'], 'output', param['experiment_name'])

    if not os.path.exists(param['experiment_path']):
        os.makedirs(param['experiment_path'])

    param['log_path']   = os.path.join(param['experiment_path'], 'output.log')
    param['valid_type'] = set(['F', 'C', 'R', 'A'])  # 'F', 'C', 'R', 'N', 'CC', 'A', 'Chitchat', 'G', 'O'\

    '''
    dataset and experiment settings
    '''
    param['data_name'] = '' # current dataset that the experiment is running about, is set in entry.py line 22
    param['data_names']  = ['dstc2', 'dstc3', 'ghome', 'family'] # 'dstc2', 'dstc3', 'ghome', 'family'

    param['raw_feature_path']       = os.path.join(param['root_path'], 'dataset', 'feature', '%s.raw_feature.pkl')
    param['extracted_feature_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.extracted_feature.pkl')
    param['pipeline_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.pipeline.pkl')

    param['metrics'] = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time']

    '''
    cross-validation
    '''
    param['do_cross_validation'] = True
    param['#division'] = 1    # number of random divisions, set to 1 in preliminary experiments
    param['#cross_validation'] = 10 # number of folds
    param['cv_index_cache_path'] = '' # the path is set in experimenter.py line 109 because it depends on the dataset name

    '''
    feature setting
    '''
    # param['utterance_range'] = ['current_user_utterance', 'next_system_utterance', 'next_user_utterance']
    # action words dictionary
    action_list = set([# Family Assistant
                                    'add', 'remove', 'delete', 'clear', 'show', 'share', 'snooze', 'list', 'item', 'items', 'member', 'remind', 'reminds', 'reminder', 'reminders', 'help', 'discard', 'start', 'stop',
                                # Google Home
                                    'room', 'light', 'turn', 'song', 'music', 'video', 'shuffle', 'volume', 'skip', 'next', 'play', 'stop', 'watch', 'cast', 'weather', 'temperature', 'time', 'timer', 'alarm', 'tell', 'find'
                                # DSTC 2&3
                                    'care', 'matter', 'any', 'else', 'price', 'cheap', 'moderate', 'expensive', 'address', 'area', 'part', 'north', 'south', 'centre', 'post', 'phone', 'telephone', 'number', 'food'
                                 ])
    # put the stemmed action words into set as well
    param['action_words'] = set()
    stemmer = PorterStemmer()
    for word in action_list:
        param['action_words'].add(word)
        param['action_words'].add(stemmer.stem(word))

    # CoreNLP setting
    param['corenlp_jars'] = (
        "/Users/memray/Project/stanford/stanford-corenlp-full-3.8.0/*",
        "/Users/memray/Project/stanford/stanford-corenlp-full-3.8.0/stanford-english-kbp-corenlp-2017-06-09-models.jar",
    )

    # LDA setting
    if not os.path.exists(os.path.join(param['root_path'], 'dataset', 'feature', 'gensim')):
        os.makedirs(os.path.join(param['root_path'], 'dataset', 'feature', 'gensim'))
    param['lda_topic_number']           = 50
    param['lda_path']                   = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.topic=%d.lda.pkl' % ('%s', param['lda_topic_number']))
    param['gensim_corpus_path']         = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.corpus.pkl' % ('%s'))
    param['gensim_dict_path']           = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.dict' % ('%s'))

    # Word2Vec setting
    param['w2v_path']                   = os.path.join('/Users/memray/Data/glove', 'GoogleNews-vectors-negative300.bin')
    param['w2v_vector_length']          = 300

    # param['4.1-ngram']     = True
    # param['4.2-edit_distance']     = True

    # Doc2Vec setting
    param['d2v_vector_length']          = 300
    param['d2v_window_size']            = 5
    param['d2v_min_count']              = 2
    param['d2v_model_path']             = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%%s.doc2vec.dim=%d.window=%d.min_count=%d.model' % (param['d2v_vector_length'], param['d2v_window_size'], param['d2v_min_count']))
    param['d2v_vector_path']             = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%%s.doc2vec.dim=%d.window=%d.min_count=%d.vector' % (param['d2v_vector_length'], param['d2v_window_size'], param['d2v_min_count']))








    config = Config(param)
    for k,v in param.items():
        config.logger.info('%s  :   %s' % (k,v))

    return config