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
    logging.getLogger().handlers = [] # remove the previous handlers
    logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging

class Config():
    # Display progress logs on stdout
    def __init__(self, param):
        self.param = param
        if 'log_path' in param:
            self.logger = init_logging(param['log_path'])
        else:
            self.logger = logging.getLogger()

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

    # param['experiment_mode'] = ['single_run', 'single_run_context_feature', 'normal_cv', 'feature_selection', 'leave_one_out', 'keep_one_only', 'reformulation_detection', 'task_boundary_detection', 'bad_case', 'print_important_features'][1]

    # two key feature settings
    param['context_set']     = ['next', 'current', 'last', 'all'][0]

    selected_feature_set_id  = 12
    '''
    Feature Combination:
    20171104: insert 3.phrasal again
    '0-all', '1-basic', '2-lexical', '3-phrasal', '4-syntactic', '5-lda', '6-w2v', '7-d2v', '8-skipthought'
    '9-[1.2.3]'
    '10-[1.3.4]'
    '11-[1.3.5]'
    '12-[1.3.6]'
    '13-[1.3.7]'
    '''
    param['feature_set']     = ['0-all', '1-basic', '2-lexical', '3-phrasal', '4-syntactic', '5-lda', '6-w2v', '7-d2v', '8-skipthought', '9-[1.2.3]', '10-[1.3.4]', '11-[1.3.5]', '12-[1.3.6]', '13-[1.3.7]'][selected_feature_set_id]

    '''
    Feature Set
    1. basic - utterance length
    2. user action
    3. time feature
    4. n-gram
    5. noun phrase
    6. entity
    7. syntactic
    8. lda
    9. w2v
    10. d2v
    11. skip-thought
    '''
    param['feature_set_number']  = [['1','2','3','4','5','6','7','8','9','10','11'], ['1','2','3'], ['4'], ['5','6'], ['7'], ['8'], ['9'], ['10'], ['11'], ['1','2','3','5','6','7', '4'], ['1','2','3','5','6','7', '8'], ['1','2','3','5','6','7', '9'], ['1','2','3','5','6','7', '10'], ['1','2','3','5','6','7', '11']][selected_feature_set_id]
    param['similarity_feature']  = False

    # context window
    param['utterance_names'] = ['last_user_utterance', 'last_system_utterance','current_user_utterance', 'next_system_utterance', 'next_user_utterance']
    if param['context_set']     == 'current':
        param['utterance_range'] = ['current_user_utterance', 'next_system_utterance']
    elif param['context_set']   == 'next':
        param['utterance_range'] = ['current_user_utterance', 'next_system_utterance','next_user_utterance']
    elif param['context_set']   == 'last':
        param['utterance_range'] = ['current_user_utterance', 'last_system_utterance','current_user_utterance']
    elif param['context_set']   == 'all':
        param['utterance_range'] = param['utterance_names']

    # param['experiment_name'] = '.'.join([param['task_name'], param['experiment_mode'], param['timemark']])
    param['experiment_name'] = '.'.join([param['timemark'], 'context=%s' % param['context_set'], 'feature=%s' % param['feature_set'], 'similarity=true' if param['similarity_feature'] else 'similarity=false'])

    param['experiment_path'] = os.path.join(param['root_path'], 'output', param['experiment_name'])

    if not os.path.exists(param['experiment_path']):
        os.makedirs(param['experiment_path'])

    param['log_path']   = os.path.join(param['experiment_path'], 'output.log')
    param['valid_type'] = set(['F', 'C', 'R', 'A'])  # 'F', 'C', 'R', 'N', 'CC', 'A', 'Chitchat', 'G', 'O'\

    '''
    dataset and experiment settings
    '''
    param['data_name'] = '' # current dataset that the experiment is running about, is set in entry.py line 22
    param['data_names']  = ['dstc2', 'dstc3', 'family', 'ghome'] # 'dstc2', 'dstc3', 'family', 'ghome'

    param['raw_feature_path']       = os.path.join(param['root_path'], 'dataset', 'feature', '%s.raw_feature.pkl')
    param['extracted_feature_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.extracted_feature.pkl')
    param['pipeline_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.pipeline.pkl')

    param['metrics'] = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time']

    '''
    cross-validation
    '''
    param['do_cross_validation'] = True
    param['#division'] = 5    # number of random divisions, set to 1 in preliminary experiments
    param['#cross_validation'] = 10 # number of folds
    param['cv_index_cache_path'] = '' # the path is set in experimenter.py because it depends on the dataset name

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


    # Skip-thought setting
    param['skipthought_model_path']     = '/Users/memray/Data/skip-thought'
    # param['skipthought_model_path']     = '/home/memray/Data/skip-thought'
    param['skipthought_data_path']      = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.skip-thought.biskip.vector')





    config = Config(param)
    for k,v in param.items():
        config.logger.info('%s  :   %s' % (k,v))

    return config


def load_basic_config():
    param = {}
    param['seed']            = 154316847
    '''
    metadata and path
    '''
    # param['root_path']  = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))
    param['root_path']  = os.path.abspath(os.path.join(os.getcwd()))
    param['root_path']  = param['root_path'][:param['root_path'].find('y_classify') + 11]
    # print('root_path: %s' % param['root_path'])

    param['experiment_path'] = os.path.join(param['root_path'], 'output', 'temp')
    param['valid_type'] = set(['F', 'C', 'R', 'A'])  # 'F', 'C', 'R', 'N', 'CC', 'A', 'Chitchat', 'G', 'O'\
    '''
    dataset and experiment settings
    '''
    param['data_name'] = '' # current dataset that the experiment is running about, is set in entry.py line 22
    param['data_names']  = ['dstc2', 'dstc3', 'family', 'ghome'] # 'dstc2', 'dstc3', 'family', 'ghome'

    param['raw_feature_path']       = os.path.join(param['root_path'], 'dataset', 'feature', '%s.raw_feature.pkl')
    param['extracted_feature_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.extracted_feature.pkl')

    '''
    feature setting
    '''
    param['utterance_range'] = ['last_user_utterance', 'last_system_utterance', 'current_user_utterance',
                                'next_system_utterance', 'next_user_utterance']

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


    # Skip-thought setting
    param['skipthought_model_path']     = '/Users/memray/Data/skip-thought'
    # param['skipthought_model_path']     = '/home/memray/Data/skip-thought'
    param['skipthought_data_path']      = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.skip-thought.biskip.vector')


    config = Config(param)

    return config


def load_batch_config(key_params):
    """
    :param key_params: must include selected_context_id, selected_feature_set_id, and similarity_feature
    """
    param = key_params
    param['seed']            = 154316847
    '''
    metadata and path
    '''
    param['root_path']  = os.path.abspath(os.path.join(os.getcwd()))
    param['root_path']  = param['root_path'][:param['root_path'].find('y_classify') + 11]

    param['task_name']       = 'utterance_type'
    param['timemark']        = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))


    # selected_context_id: 0-3
    param['context_set']     = ['next', 'current', 'last', 'all'][param['selected_context_id']]

    # context window (setting for deep models is located in exp_deepmodel line 155)
    param['utterance_names'] = ['last_user_utterance', 'last_system_utterance','current_user_utterance', 'next_system_utterance', 'next_user_utterance']
    if param['context_set']     == 'current':
        param['utterance_range'] = ['current_user_utterance', 'next_system_utterance']
    elif param['context_set']   == 'next':
        param['utterance_range'] = ['current_user_utterance', 'next_system_utterance','next_user_utterance']
    elif param['context_set']   == 'last':
        param['utterance_range'] = ['current_user_utterance', 'last_system_utterance','current_user_utterance']
    elif param['context_set']   == 'all':
        param['utterance_range'] = param['utterance_names']


    if not param['deep_model']:
        # deprecated
        # param['experiment_mode'] = ['single_run', 'single_run_context_feature', 'normal_cv', 'feature_selection', 'leave_one_out', 'keep_one_only', 'reformulation_detection', 'task_boundary_detection', 'bad_case', 'print_important_features'][1]

        '''
        ID of merged feature sets [0-14]
        20180123: add 15-18: [1,2,3,4] + one of [5,6,7,8]
        '''
        param['feature_set'] = \
                ['0-all', '1-basic', '2-lexical', '3-phrasal', '4-syntactic', '5-lda', '6-w2v', '7-d2v', '8-skipthought',
                 '9-[2+1.3.4]', '10-[5+1.3.4]',
                 '11-[6+1.3.4]', '12-[7+1.3.4]',
                 '13-[8+1.3.4]', '14-[1.3.4]',
                 '15-[5+1.2.3.4]', '16-[6+1.2.3.4]',
                 '17-[7+1.2.3.4]', '18-[8+1.2.3.4]'][param['selected_feature_set_id']]

        '''
        Combo feature setting
        Each id corresponds to the following feature set:
            1. basic - utterance length
            2. user action
            3. time feature
            4. n-gram
            5. noun phrase
            6. entity
            7. syntactic
            8. lda
            9. w2v
            10. d2v
            11. skip-thought
        '''
        param['feature_set_number'] = \
                [['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], ['1', '2', '3'], ['4'], ['5', '6'], ['7'], ['8'], ['9'], ['10'], ['11'],
                 ['1', '2', '3', '5', '6', '7', '4'], ['1', '2', '3', '5', '6', '7', '8'],
                 ['1', '2', '3', '5', '6', '7', '9'], ['1', '2', '3', '5', '6', '7', '10'],
                 ['1', '2', '3', '5', '6', '7', '11'], ['1', '2', '3', '5', '6', '7'],
                 ['8', '1', '2', '3', '4', '5', '6', '7'], ['9', '1', '2', '3', '4', '5', '6', '7'], # '15-[5+1.2.3.4]', '16-[6+1.2.3.4]'
                 ['10', '1', '2', '3', '4', '5', '6', '7'], ['11', '1', '2', '3', '4', '5', '6', '7'] # '17-[7+1.2.3.4]', '18-[8+1.2.3.4]'
                 ][param['selected_feature_set_id']]

        param['similarity_feature']  = param['similarity_feature']

        # param['experiment_name'] = '.'.join([param['task_name'], param['experiment_mode'], param['timemark']])
        if key_params['experiment_mode'].endswith('discrete_feature_selection'):
            param['experiment_name'] = '.'.join([param['timemark'], key_params['experiment_mode'], 'feature_number=%s' % key_params['k_feature_to_keep'], 'context=%s' % param['context_set'], 'feature=%s' % param['feature_set'], 'similarity=true' if param['similarity_feature'] else 'similarity=false'])
        elif key_params['experiment_mode'].endswith('continuous_feature_selection'):
            param['experiment_name'] = '.'.join([param['timemark'], key_params['experiment_mode'], 'pca_component=%s' % key_params['k_component_for_pca'], 'feature_number=%s' % key_params['k_feature_to_keep'], 'context=%s' % param['context_set'], 'feature=%s' % param['feature_set'], 'similarity=true' if param['similarity_feature'] else 'similarity=false'])
        else:
            param['experiment_name'] = '.'.join([param['timemark'], key_params['experiment_mode'], 'context=%s' % param['context_set'], 'feature=%s' % param['feature_set'], 'similarity=true' if param['similarity_feature'] else 'similarity=false'])

    else:
        param['experiment_name'] = '.'.join([param['timemark'], 'context=%s' % param['context_set'], 'model=%s' % param['deep_model_name']])


    param['experiment_path'] = os.path.join(param['root_path'], 'output', param['experiment_name'])

    if not os.path.exists(param['experiment_path']):
        os.makedirs(param['experiment_path'])

    param['log_path']   = os.path.join(param['experiment_path'], 'output.log')
    param['valid_type'] = set(['F', 'C', 'R', 'A'])  # 'F', 'C', 'R', 'N', 'CC', 'A', 'Chitchat', 'G', 'O'\

    '''
    dataset and experiment settings
    '''
    param['data_name'] = '' # current dataset that the experiment is running about, is set in entry.py line 22
    param['data_names']  = ['dstc2', 'dstc3', 'family', 'ghome'] # 'dstc2', 'dstc3', 'family', 'ghome'

    param['raw_feature_path']       = os.path.join(param['root_path'], 'dataset', 'feature', '%s.raw_feature.pkl')
    param['extracted_feature_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.extracted_feature.pkl')
    param['pipeline_path'] = os.path.join(param['root_path'], 'dataset', 'feature', '%s.pipeline.pkl')

    param['metrics'] = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time']

    '''
    cross-validation
    '''
    param['do_cross_validation'] = True
    param['#division'] = 5    # number of random divisions, set to 1 in preliminary experiments
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
    # param['w2v_path']                   = os.path.join('/home/memray/Data/glove', 'GoogleNews-vectors-negative300.bin')
    param['w2v_vector_length']          = 300

    # param['4.1-ngram']     = True
    # param['4.2-edit_distance']     = True

    # Doc2Vec setting
    param['d2v_vector_length']          = 300
    param['d2v_window_size']            = 5
    param['d2v_min_count']              = 2
    param['d2v_model_path']             = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%%s.doc2vec.dim=%d.window=%d.min_count=%d.model' % (param['d2v_vector_length'], param['d2v_window_size'], param['d2v_min_count']))
    param['d2v_vector_path']             = os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%%s.doc2vec.dim=%d.window=%d.min_count=%d.vector' % (param['d2v_vector_length'], param['d2v_window_size'], param['d2v_min_count']))

    '''
    Deep models setting
    '''
    # words whose freq below the given thresholds will be discarded
    param['num_word_keep']                 = {'dstc2' : 300, # #(vocab)=668, #(freq>=10)=389
                                           'dstc3': 300,  # #(vocab)=891, #(freq>=10)=480
                                           'family': 1000, # #(vocab)=4068, #(freq>=15)=928
                                           'ghome': 1000   # #(vocab)=5856, #(freq>=10)=897
                                             }

    param['batch_size']                 = 128
    param['max_epoch']                  = 50
    param['early_stop_tolerance']       = 2
    param['concat_sents']               = False

    # CNN setting
    param['cnn_setting'] = {
        "model"             : 'rand', # available models: rand, static, non-static, multichannel
        "early_stopping"    : True,
        "word_dim"          : 300,
        "filters"           : [3, 4, 5],
        "filter_num"        : [100, 100, 100],
        'class_size'        : len(param['valid_type']),
        'batch_size'        : param['batch_size'],
        "learning_rate"     : 1.0e-3,
        "norm_limit"        : 10,
        "dropout_prob"      : 0.0,

        "sentence_num"      : len(param['utterance_range']),
    }

    # Skip-thought setting
    param['skipthought_setting'] = {
        # "skipthought_model_path"    : '/home/memray/Data/skip-thought',
        "skipthought_model_path"  : '/Users/memray/Data/skip-thought',
        "skipthought_data_path"     : os.path.join(param['root_path'], 'dataset', 'feature', 'gensim', '%s.skip-thought.biskip.vector'),
        "fixed_emb"                 : True,

        "sentence_num"              : len(param['utterance_range']),
        "hidden_size"               : 2400,
        "class_size"                : len(param['valid_type']),
        "learning_rate"             : 1.0e-4,
        "norm_limit"                : 3,
        "dropout_prob"              : 0.5,
    }

    # LSTM setting
    param['lstm_setting'] = {
        "model"             : 'non-static', # available models: rand, non-static, static
        'hidden_size'       : 32,
        'embedding_size'    : 300,
        'num_layers'        : 1,
        'bidirectional'     : False,
        "learning_rate"     : 1.0e-3,
        "class_size"        : len(param['valid_type']),
        "norm_limit"        : 2,
        'clip_grad_norm'    : 2,
        "dropout_prob"      : 0.1,
    }

    config = Config(param)
    for k,v in param.items():
        config.logger.info('%s  :   %s' % (k,v))

    return config
