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

    param['experiment_mode'] = ['normal', 'feature_selection', 'leave_one_out', 'keep_one_only'][1]

    param['experiment_name'] = '.'.join([param['task_name'], param['experiment_mode'], param['timemark']])
    param['experiment_path'] = os.path.join(param['root_path'], 'output', param['experiment_name'])

    if not os.path.exists(param['experiment_path']):
        os.makedirs(param['experiment_path'])

    param['log_path']   = os.path.join(param['experiment_path'], 'output.log')
    param['valid_type'] = set(['F', 'C', 'R', 'A'])  # 'F', 'C', 'R', 'N', 'CC', 'A', 'Chitchat', 'G', 'O'\

    '''
    dataset and experiment settings
    '''
    param['data_name'] = '' # current dataset that the experiment is running about, is set in entry.py line 22
    param['data_names']  = ['dstc2', 'dstc3', 'ghome', 'family'] # 'dstc2', 'ghome', 'family'

    param['raw_feature_path']       = os.path.join(param['root_path'], 'dataset', 'feature', '%s.raw_feature.pkl')
    param['extracted_feature_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.extracted_feature.pkl')
    param['pipeline_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.pipeline.pkl')

    param['metrics'] = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time']

    '''
    cross-validation
    '''
    param['do_cross_validation'] = True
    param['#division'] = 1    # number of random divisions, set to 1 in preliminary experimentss
    param['#cross_validation'] = 10 # number of folds
    param['cv_index_cache_path'] = '' # the path is set in experimenter.py line 109 because it depends on the dataset name

    '''
    feature setting
    '''
    # context window
    param['utterance_range'] = ['current_user_utterance', 'next_system_utterance', 'next_user_utterance']
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
    # LDA setting
    param['lda_topic_number']           = 50
    param['lda_path']                   = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.topic=%d.lda.pkl' % ('%s', param['lda_topic_number']))
    param['gensim_corpus_path']         = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.corpus.pkl' % ('%s'))
    param['gensim_dict_path']           = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.dict' % ('%s'))

    # Word2Vec setting
    param['w2v_path']                   = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', 'GoogleNews-vectors-negative300.bin')
    param['w2v_vector_length']          = 300

    # param['4.1-ngram']     = True
    # param['4.2-edit_distance']     = True

    config = Config(param)
    for k,v in param.items():
        config.logger.info('%s  :   %s' % (k,v))

    return config