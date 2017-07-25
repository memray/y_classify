# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn import preprocessing

from data.data_loader import data_loader, DataLoader, Utterance
from classify import configuration
from classify.feature_extractor import Feature_Extractor
from classify.experimenter import Experimenter
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
        X                       = extractor.extract()

        '''
        the 1st version is scaled to zero-mean and uni-variance
        now it's scaled with MinMaxScaler to make feature values non-negative
        '''
        X                       = np.nan_to_num(X.todense())
        X_scaled                = preprocessing.scale(X)
        # min_max_scaler = preprocessing.MinMaxScaler()
        # X_scaled = min_max_scaler.fit_transform(X)
        X_clear                 = np.nan_to_num(X_scaled)
        # print("Checkinf for NaN and Inf")
        # print("np.inf=", np.where(np.isnan(X)))
        # print("is.inf=", np.where(np.isinf(X)))
        # print("np.max=", np.max(abs(X)))

        if config['experiment_mode'] == 'feature_selection':
            result                  = exp.run_cross_validation_with_feature_selection(X_clear, Y)
        elif config['experiment_mode'] == 'leave_one_out':
            result                  = exp.run_cross_validation_with_leave_one_out(X_clear, Y)
        else:
            result                  = exp.run_cross_validation(X_clear, Y)

            # find the best classifier (with best F1-score)
            result = result[np.asarray(result).T[4].argmax()]
            best_results[data_name] = result

    exp.export_summary(best_results.values(), os.path.join(config.param['experiment_path'], 'best_of_each_dataset.csv'))