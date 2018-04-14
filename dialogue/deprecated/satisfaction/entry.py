# -*- coding: utf-8 -*-
from __future__ import print_function
from data.data_loader import data_loader, DataLoader, Utterance
from satisfaction import configuration
from satisfaction.feature_extracter import Feature_Extracter
from satisfaction.experimenter import Experimenter
import numpy as np
import os

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    # initialize
    config = configuration.load_config()
    extractor = Feature_Extracter(config)
    exp = Experimenter(config)

    best_results = []
    # iterate each dataset
    for data_name in config['data_names']:
        config.param['data_name'] = data_name

        config.logger.info('*' * 50)
        config.logger.info('-' * 20 + data_name + '-' * 20)
        config.logger.info('*' * 50)
        # initialize data_loader
        loader = data_loader(data_name, {'config': config})

        # load raw and annotated data
        all_sessions = loader()
        session_ids, annotated_sessions = loader.load_annotated_data()
        loader.stats()

        # train and test
        X_raw, Y                = extractor.split_to_instances(annotated_sessions)
        X                       = extractor.extract()
        result                  = exp.run_cross_validation(X, Y)

        # find the best classifier (with best F1-score)
        result = result[np.asarray(result).T[4].argmax()]
        result[0] = data_name + ' - ' + result[0]
        best_results.append(result)

    exp.export_summary(best_results, os.path.join(config.param['experiment_path'], 'summary_of_each_dataset.csv'))