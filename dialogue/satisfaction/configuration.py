# -*- coding: utf-8 -*-
import os, sys

import logging

import time

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
    param['data_names']  = ['dstc2', 'ghome', 'family'] # 'dstc2', 'ghome', 'family'

    '''
    metadata and path
    '''
    param['root_path']  = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))

    param['task_name']       = 'utterance_type'
    param['timemark']        = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    param['experiment_name'] = '.'.join([param['task_name'], param['timemark']])
    param['experiment_path'] = os.path.join(param['root_path'], 'output', param['experiment_name'])

    if not os.path.exists(param['experiment_path']):
        os.makedirs(param['experiment_path'])

    param['log_path']   = os.path.join(param['experiment_path'], 'output.log')
    param['valid_type'] = set(['F', 'C', 'R', 'A'])  # 'F', 'C', 'R', 'N', 'CC', 'A', 'Chitchat', 'G', 'O'\

    '''
    cross-validation
    '''
    param['do_cross_validation'] = True
    param['#cross_validation'] = 2
    param['cv_index_cache_path'] = '' # the path is set in experimenter.py because it depends on the dataset name

    '''
    feature setting
    '''
    param['utterance_range'] = ['current_user_utterance', 'next_user_utterance', 'next_system_utterance']

    param['4.1-ngram']     = True
    param['4.2-edit_distance']     = True

    config = Config(param)
    for k,v in param.items():
        config.logger.info('%s  :   %s' % (k,v))

    return config