# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn import preprocessing

from data.data_loader import data_loader, DataLoader, Utterance
from classify import configuration
from classify.feature_extractor import Feature_Extractor
from classify.exp_shallowmodel import Experimenter
import numpy as np
import os

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    # initialize
    config = configuration.load_config()
    extractor = Feature_Extractor(config)
    exp = Experimenter(config)

    best_results = {}
    # iterate each dataset
    for data_name in config['data_names']:
        config.param['data_name'] = data_name

        config.logger.info('*' * 50)
        config.logger.info('-' * 20 + data_name + '-' * 20)
        config.logger.info('*' * 50)
        # initialize data_loader
        loader = data_loader(data_name, {'config': config})
        config['data_loader'] = loader
        loader()
        # load annotated data
        session_ids, annotated_sessions = loader.load_annotated_data()
        loader.stats()

        # train and test
        X_raw, Y                = extractor.split_to_instances(annotated_sessions)
        X_raw_feature           = extractor.extract_raw_feature()
        result                  = exp.run_single_pass_doc2vec(X_raw_feature, Y)

    exp.export_summary(best_results.values(), os.path.join(config.param['experiment_path'], 'best_of_each_dataset.csv'))