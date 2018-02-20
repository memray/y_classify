# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import logging
import copy
import pickle

import gensim
import numpy as np
from optparse import OptionParser
import sys, os
from time import time
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.feature_selection import f_classif

plt.switch_backend('agg')

from collections import Counter

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics, preprocessing

from dialogue.classify.feature_extractor import ItemSelector, print_printable_features
from dialogue.data import data_loader

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report", default=True,
              help="Print a detailed classification report.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm", default=True,
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10", default=True,
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
(opts, args) = op.parse_args([])


class ShallowExperimenter():
    def __init__(self, config):
        self.config = config
        self.logger = self.config.logger

    def feature_statistics(self, feature_names):
        counter = dict()

        for f_id, f_name in enumerate(feature_names):
            f_start_number = f_name[0:f_name.find('-')]
            counter[f_start_number] = counter.get(f_start_number, 0) + 1

            if f_start_number.find('.') > 0:
                f_start_number = f_name[0:f_start_number.find('.')]
                counter[f_start_number] = counter.get(f_start_number, 0) + 1

        count_tuples = sorted(counter.items(), key=lambda x: int(x[0][:x[0].find('.')]) if x[0].find('.') > 0 else int(x[0]))

        for f_id, f_count in count_tuples:
            if f_id.find('.') > 0:
                self.logger.info('\t\t%s : %d' % (f_id, f_count))
            else:
                self.logger.info('%s : %d' % (f_id, f_count))

        return count_tuples

    def load_single_run_index(self, X, Y):
        '''
        Split data into 8:1:1 randomly
        :param X:
        :param Y:
        :return:
        '''
        # load the data index for cross-validation
        if self.config['experiment_mode'] == 'reformulation_detection' or self.config['experiment_mode'] == 'task_boundary_detection':
            sr_index_cache_path = os.path.join(self.config.param['root_path'], 'dataset', 'single_run',
                                               self.config.param['data_name'] + '.%s.index_cache.evenly.pkl' % (
                                                   self.config['experiment_mode']))
        else:
            sr_index_cache_path = os.path.join(self.config.param['root_path'], 'dataset', 'single_run',
                                               self.config.param['data_name'] + '.index_cache.evenly.pkl')

        if os.path.exists(sr_index_cache_path):
            with open(sr_index_cache_path, 'rb') as idx_cache:
                train_ids, valid_ids, test_ids = pickle.load(idx_cache)
        else:
            if not os.path.exists(os.path.join(self.config.param['root_path'], 'dataset', 'single_run')):
                os.makedirs(os.path.join(self.config.param['root_path'], 'dataset', 'single_run'))

            with open(sr_index_cache_path, 'wb') as idx_cache:
                # get ids and sort out by Y
                y_dict = {}
                for y_id, y in enumerate(Y):
                    y_list = y_dict.get(y, [])
                    y_list.append(y_id)
                    y_dict[y] = y_list

                # get a new division - shuffle the data of each class
                data_dict_copy = copy.deepcopy(y_dict)
                for y, y_list in data_dict_copy.items():
                    np.random.shuffle(y_list)
                    data_dict_copy[y] = np.asarray(y_list)
                # print('*' * 20 + ' div=%d ' % div)

                # segment to folds evenly to avoid data skewness
                # initialize lists for the train/test id of this div-fold
                train_ids = np.array([], dtype=int)
                valid_ids = np.array([], dtype=int)
                test_ids = np.array([], dtype=int)
                # pick up the specific fold of data from each class
                for y, y_list in data_dict_copy.items():
                    fold_size = int(len(y_list) / self.config['#cross_validation'])

                    test_idx = np.asarray(range(fold_size))
                    valid_idx = np.asarray(range(fold_size, 2*fold_size))
                    train_idx = np.asarray([i for i in range(len(y_list)) if i not in test_idx and i not in valid_idx])

                    # print('fold=%d, #(%s)=%d, #(train)=%d, #(test)=%d' % (
                    #     fold, y, len(y_list), len(train_idx), len(test_idx)))

                    train_ids = np.append(train_ids, y_list[train_idx])
                    valid_ids = np.append(valid_ids, y_list[valid_idx])
                    test_ids  = np.append(test_ids, y_list[test_idx])

                pickle.dump([train_ids, valid_ids, test_ids], idx_cache)

        return train_ids, valid_ids, test_ids


    def load_cv_index_train8_valid1_test1(self, Y):
        '''
        load the data index for cross-validation, split into 3 parts
        '''
        cv_index_cache_path = os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation',
                                           self.config.param['data_name'] + '.index_cache.evenly.#div=%d.#cv=%d.pkl' % (
                                           self.config['#division'], self.config['#cross_validation']))

        if os.path.exists(cv_index_cache_path):
            with open(cv_index_cache_path, 'rb') as idx_cache:
                train_ids, valid_ids, test_ids = pickle.load(idx_cache)
        else:
            if not os.path.exists(os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation')):
                os.makedirs(os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation'))

            num_data = len(Y)
            train_ids = []
            valid_ids = []
            test_ids  = []

            # get ids and sort out by Y
            data_dict = {}
            for y_id, y in enumerate(Y):
                data_list = data_dict.get(y, [])
                data_list.append(y_id)
                data_dict[y] = data_list

            # get a new division - shuffle the data of each class
            for div in range(self.config['#division']):
                data_dict_copy = copy.deepcopy(data_dict)
                for y, y_list in data_dict_copy.items():
                    np.random.shuffle(y_list)
                    data_dict_copy[y] = np.asarray(y_list)
                # print('*' * 20 + ' div=%d ' % div)

                # segment to folds evenly to avoid data skewness
                for fold in range(self.config['#cross_validation']):
                    # initialize lists for the train/test id of this div-fold
                    train_id = np.array([], dtype=int)
                    valid_id = np.array([], dtype=int)
                    test_id = np.array([], dtype=int)

                    # pick up the specific fold of data from each class
                    for y, y_list in data_dict_copy.items():
                        fold_size = int(len(y_list) / self.config['#cross_validation'])
                        if fold == self.config['#cross_validation'] - 1:
                            test_idx = np.asarray(range(fold * fold_size, len(y_list)))
                            valid_idx = np.asarray(range(0, fold_size))
                            print('fold=%d, test=[%d, %d], valid=[%d, %d]' % (fold, fold * fold_size, len(y_list), 0, fold * fold_size))
                        else:
                            test_idx = np.asarray(range(fold * fold_size, (fold + 1) * fold_size))
                            valid_idx = np.asarray(range((fold + 1) * fold_size, (fold + 2) * fold_size))
                            print('fold=%d, test=[%d, %d], valid=[%d, %d]' % (fold, fold * fold_size, (fold + 1) * fold_size, (fold + 1) * fold_size, (fold + 2) * fold_size))

                        train_idx = np.asarray([i for i in range(len(y_list)) if i not in test_idx and i not in valid_idx])

                        train_id = np.append(train_id, y_list[train_idx])
                        valid_id = np.append(valid_id, y_list[valid_idx])
                        test_id = np.append(test_id, y_list[test_idx])

                        print('fold=%d, #(%s)=%d, #(train)=%d, #(valid)=%d, #(test)=%d' % (
                            fold, y, len(y_list), len(train_idx), len(valid_idx), len(test_idx)))

                        print('*' * 50)

                    # finish generation of this round, append to *_ids
                    train_ids.append(train_id)
                    test_ids.append(test_id)
                    valid_ids.append(valid_id)

            with open(cv_index_cache_path, 'wb') as idx_cache:
                pickle.dump([train_ids, valid_ids, test_ids], idx_cache)

        return train_ids, valid_ids, test_ids

    def load_cv_index(self, X, Y):
        # load the data index for cross-validation
        if self.config['experiment_mode'] == 'reformulation_detection' or self.config['experiment_mode'] == 'task_boundary_detection':
            cv_index_cache_path = os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation',
                                               self.config.param['data_name'] + '.%s.index_cache.evenly.#div=%d.#cv=%d.pkl' % (
                                                   self.config['experiment_mode'], self.config['#division'], self.config['#cross_validation']))
        else:
            cv_index_cache_path = os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation',
                                               self.config.param['data_name'] + '.index_cache.evenly.#div=%d.#cv=%d.pkl' % (
                                                   self.config['#division'], self.config['#cross_validation']))
        if os.path.exists(cv_index_cache_path):
            with open(cv_index_cache_path, 'rb') as idx_cache:
                train_ids, test_ids = pickle.load(idx_cache)
        else:
            if not os.path.exists(os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation')):
                os.makedirs(os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation'))

            with open(cv_index_cache_path, 'wb') as idx_cache:
                num_data = len(Y)
                train_ids = []
                test_ids = []

                # get ids and sort out by Y
                data_dict = {}
                for y_id, y in enumerate(Y):
                    data_list = data_dict.get(y, [])
                    data_list.append(y_id)
                    data_dict[y] = data_list

                # get a new division - shuffle the data of each class
                for div in range(self.config['#division']):
                    data_dict_copy = copy.deepcopy(data_dict)
                    for y, y_list in data_dict_copy.items():
                        np.random.shuffle(y_list)
                        data_dict_copy[y] = np.asarray(y_list)
                    # print('*' * 20 + ' div=%d ' % div)

                    # segment to folds evenly to avoid data skewness
                    for fold in range(self.config['#cross_validation']):
                        # initialize lists for the train/test id of this div-fold
                        train_id = np.array([], dtype=int)
                        test_id = np.array([], dtype=int)
                        # pick up the specific fold of data from each class
                        for y, y_list in data_dict_copy.items():
                            fold_size = int(len(y_list) / self.config['#cross_validation'])
                            if fold == self.config['#cross_validation'] - 1:
                                test_idx = np.asarray(range(fold * fold_size, len(y_list)))
                            else:
                                test_idx = np.asarray(range(fold * fold_size, (fold + 1) * fold_size))
                            train_idx = np.asarray([i for i in range(len(y_list)) if i not in test_idx])

                            train_id = np.append(train_id, y_list[train_idx])
                            test_id = np.append(test_id, y_list[test_idx])
                            # print('fold=%d, #(%s)=%d, #(train)=%d, #(test)=%d' % (
                            #     fold, y, len(y_list), len(train_idx), len(test_idx)))

                        train_ids.append(train_id)
                        test_ids.append(test_id)

                pickle.dump([train_ids, test_ids], idx_cache)

        return train_ids, test_ids

    def benchmark(self, model_name, clf):
        global X_train, Y_train, X_test, Y_test
        results = []

        if 'vectorizer' in self.config:
            feature_names = np.asarray(self.config['vectorizer'].get_feature_names())
        else:
            feature_names = None

        self.logger.info('_' * 80)
        self.logger.info("Training: ")
        self.logger.info(clf)
        t0 = time()
        clf.fit(X_train, Y_train)
        train_time = time() - t0
        self.logger.info("train time: %0.3fs" % train_time)

        '''
        have to handle validation results
        '''
        if 'X_valid' in globals() and 'Y_valid' in globals():
            global X_valid, Y_valid
            data_for_output = [('valid', X_valid, Y_valid), ('test', X_test, Y_test)]
        else:
            data_for_output = [('test', X_test, Y_test)]

        # clf = None
        for valid_or_test, X, Y in data_for_output:
            t0 = time()
            pred = clf.predict(X)

            '''
            baseline, predict the majority label
            '''
            # majorarity_label = sorted(Counter(X).items(), key=lambda x:x[1])[::-1][0][0]
            # pred = [majorarity_label] * len(Y)

            '''
            baseline, predict a uniform random number
            '''
            # pred = np.random.random_integers(low=0, high=3, size=(len(Y)))

            test_time = time() - t0
            self.logger.info("test time:  %0.3fs" % test_time)
            result = self.classification_report(Y, pred, model_name, valid_or_test, clf)

            results.append(result)

        # if return_y_pred:
        #     return [model_name, acc_score, precision_score, recall_score, f1_score, train_time, test_time], pred
        # else:
        #     return model_name, acc_score, precision_score, recall_score, f1_score, train_time, test_time

        # return a list, the 1st item is validation result, the 2nd is testing result
        return results

    def benchmark_train_test(self, model_name, clf, return_y_pred = False):
        '''
        old benchmark method, only do train and then test, have to train twice if needs validation
        :param model_name:
        :param clf:
        :param return_y_pred:
        :return:
        '''
        global X_train, Y_train, X_test, Y_test, X_valid, Y_valid
        results = {}

        if 'vectorizer' in self.config:
            feature_names = np.asarray(self.config['vectorizer'].get_feature_names())
        else:
            feature_names = None
        # feature_names = np.asarray(self.config['vectorizer'].get_feature_names())
        target_names  = np.asarray(self.config['label_encoder'].classes_)

        self.logger.info('_' * 80)
        self.logger.info("Training: ")
        self.logger.info(clf)
        t0 = time()
        clf.fit(X_train, Y_train)
        train_time = time() - t0
        self.logger.info("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        self.logger.info("test time:  %0.3fs" % test_time)

        acc_score = metrics.accuracy_score(Y_test, pred)
        precision_score = metrics.precision_score(Y_test, pred, average='macro')
        recall_score = metrics.recall_score(Y_test, pred, average='macro')
        f1_score = metrics.f1_score(Y_test, pred, average='macro')

        self.logger.info("accuracy:   %0.3f" % acc_score)
        self.logger.info("f1_score:   %0.3f" % f1_score)

        if (not hasattr(clf, 'kernel')) or (hasattr(clf, 'kernel') and clf.kernel != 'rbf'):
            if hasattr(clf, 'coef_'):
                self.logger.info("dimensionality: %d" % clf.coef_.shape[1])
                self.logger.info("density: %f" % density(clf.coef_))

                if opts.print_top10 and feature_names is not None:
                    self.logger.info("top 10 keywords per class:")
                    for i, label in enumerate(target_names):
                        top10 = np.argsort(clf.coef_[i])[-10:]
                        self.logger.info("%s: %s" % (label, " ".join(feature_names[top10])))
                self.logger.info('')

        if opts.print_report:
            self.logger.info("classification report:")
            report = metrics.classification_report(Y_test, pred,
                                                target_names=target_names)
            self.logger.info(report)

        if opts.print_cm:
            self.logger.info("confusion matrix:")
            confusion_mat = str(metrics.confusion_matrix(Y_test, pred))
            self.logger.info('\n'+confusion_mat)

        clf_descr = str(clf) # str(clf).split('(')[0]

        results['dataset']          = self.config.param['data_name']
        results['model']            = model_name
        if 'valid_test' in self.config.param:
            results['valid_test']   = self.config.param['valid_test']

        results['accuracy']         = acc_score
        results['precision']        = precision_score
        results['recall']           = recall_score
        results['f1_score']         = f1_score
        results['training_time']    = train_time
        results['test_time']        = test_time
        results['report']           = report
        results['confusion_matrix'] = confusion_mat
        results['y_test']           = Y_test.tolist()
        results['y_pred']           = pred.tolist()

        # if return_y_pred:
        #     return [model_name, acc_score, precision_score, recall_score, f1_score, train_time, test_time], pred
        # else:
        #     return model_name, acc_score, precision_score, recall_score, f1_score, train_time, test_time
        return results

    def run_cross_validation(self, X, Y):
        train_ids, valid_ids, test_ids = self.load_cv_index_train8_valid1_test1(Y)
        cv_results = []

        global X_train, Y_train, X_test, Y_test
        for r_id, (train_id, test_id) in enumerate(zip(train_ids, test_ids)):
            # if r_id > 0:
            #     break

            self.config['test_round'] = r_id

            X_train = np.nan_to_num(preprocessing.scale(X[train_id]))
            Y_train = Y[train_id]
            X_test = np.nan_to_num(preprocessing.scale(X[test_id]))
            Y_test = Y[test_id]

            self.logger.info('*' * 20 + ' %s - Round %d ' % (self.config['data_name'], r_id))
            self.logger.info('#(data) = %s' % str(len(X_train)))
            self.logger.info('#(feature) = %s' % str(X_train.shape[1]))

            cv_results.extend(self.run_experiment())

        self.export_cv_results(cv_results, test_ids, Y)

        return cv_results

    def run_experiment(self):
        '''

        :return: 'model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time'
        '''
        # global X_train, Y_train, X_test, Y_test

        # num_data = len(Y)
        # X_train = X[: int(0.8 * num_data)]
        # Y_train = Y[: int(0.8 * num_data)]
        # X_test  = X[int(0.8 * num_data):]
        # Y_test  = Y[int(0.8 * num_data):]

        results = []

        '''
        for C in [2**x for x in [0]]: # [-4, -3, -2, -1, 0, 1, 2, 3]
            self.logger.info('=' * 80)
            self.logger.info("LR.pen=l1.C=%f" % C)
            results.append(self.benchmark('LR.pen=l1.C=%f' % C, LogisticRegression(solver="liblinear", penalty='l1', C=C)))

            self.logger.info('=' * 80)
            self.logger.info("LinearSVC.pen=l1, C=%f" % C)
            results.append(self.benchmark('LinearSVC.pen=l1.C=%f' % C, LinearSVC(penalty='l1', tol=1e-3, dual=False, C=C)))

            self.logger.info('=' * 80)
            self.logger.info("LinearSVC.pen=l2, C=%f" % C)
            results.append(self.benchmark('LinearSVC.pen=l2.C=%f' % C, LinearSVC(penalty='l2', tol=1e-3, C=C)))

            self.logger.info('=' * 80)
            self.logger.info("LR.pen=l1.C=%f" % C)
            results.append(self.benchmark('LR.pen=l1.C=%f' % C, LogisticRegression(solver="liblinear", penalty='l1', C=C)))

            self.logger.info('=' * 80)
            self.logger.info("LR.pen=l2.C=%f" % C)
            results.append(self.benchmark('LR.pen=l2.C=%f' % C, LogisticRegression(solver="lbfgs", multi_class='multinomial', penalty='l2', C=C, dual=False)))

            self.logger.info('=' * 80)
            self.logger.info("RBF SVC with C=%f" % C)
            results.append(self.benchmark('RBF SVC with C=%f' % C, SVC(C=C, cache_size=200, class_weight=None,
                                          degree=3, gamma='auto', kernel='rbf',
                                          max_iter=-1, probability=False, random_state=None, shrinking=True,
                                          tol=0.001, verbose=False)))
        '''

        '''
        # a special one with large C to disable the regularization
        self.logger.info('=' * 80)
        C = 1e20
        self.logger.info("LR.pen=l2.C=%f" % C)
        results.append(self.benchmark('LR.pen=l2.C=%f' % C, LogisticRegression(solver="liblinear", penalty='l2', C=C)))

        '''

        """
        if self.config.param['data_name'] in ['dstc2', 'dstc3']:
            C = 2**(-4)
        else:
            C = 2**(1)
        self.logger.info('=' * 80)
        self.logger.info("LinearSVC.pen=l1, C=%f" % 2**(-4))
        # results.append(self.benchmark('LinearSVC.pen=l1.C=%f' % C, OneVsRestClassifier(LinearSVC(penalty='l1', tol=1e-3, dual=False, C=C), n_jobs=-1)))
        results.append(self.benchmark('LinearSVC.pen=l1.C=%f' % C, LinearSVC(penalty='l1', tol=1e-3, dual=False, C=C)))
        """

        for clf, name in [
                # (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                # (Perceptron(n_iter=50), "Perceptron"),
                # (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                # (KNeighborsClassifier(n_neighbors=10), "kNN"),
                # (RandomForestClassifier(n_estimators=50), "Random forest.#tree=50"),
                # (RandomForestClassifier(n_estimators=100, n_jobs=-1), "Random forest.#tree=100"),
                # (RandomForestClassifier(n_estimators=300, n_jobs=-1), "Random forest.#tree=300"),
                # (RandomForestClassifier(n_estimators=500, n_jobs=-1), "Random forest.#tree=500")
                # (RandomForestClassifier(n_estimators=64, n_jobs=-1), "Random forest.#tree=64"),
                # (RandomForestClassifier(n_estimators=128, n_jobs=-1), "Random forest.#tree=128"),
                (RandomForestClassifier(n_estimators=256, n_jobs=-1), "Random forest.#tree=256"),
                # (RandomForestClassifier(n_estimators=512, n_jobs=-1), "Random forest.#tree=512"),
                # (RandomForestClassifier(n_estimators=1024, n_jobs=-1), "Random forest.#tree=1024")
        ]:
            self.logger.info('=' * 80)
            self.logger.info(name)
            results.append(self.benchmark(name, clf))
        '''

        for C in [2**x for x in [-4, 0, 2, 3]]: # [0]+[2**x for x in [-3, -2, -1, 0]]
            self.logger.info('=' * 80)
            self.logger.info("LinearSVC.pen=l1, C=%f" % C)
            # results.append(self.benchmark('LinearSVC.pen=l1.C=%f' % C, OneVsRestClassifier(LinearSVC(penalty='l1', tol=1e-3, dual=False, C=C, max_iter=1000), n_jobs=-1)))
            results.append(self.benchmark('LinearSVC.pen=l1.C=%f' % C, LinearSVC(penalty='l1', tol=1e-3, dual=False, C=C, max_iter=1000)))

            self.logger.info('=' * 80)
            self.logger.info("LinearSVC.pen=l2, C=%f" % C)
            results.append(self.benchmark('LinearSVC.pen=l2.C=%f' % C, LinearSVC(penalty='l2', tol=1e-3, C=C, max_iter=1000)))
            # results.append(self.benchmark('LinearSVC.pen=l2.C=%f' % C, OneVsRestClassifier(LinearSVC(penalty='l2', tol=1e-3, C=C, max_iter=1000), n_jobs=-1)))

            self.logger.info('=' * 80)
            self.logger.info("RBF SVC with gamma=auto, C=%f" % C)
            results.append(self.benchmark('RBF SVC with C=%f' % C, SVC(C=C, cache_size=2000, class_weight=None, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)))
            # results.append(self.benchmark('RBF SVC with C=%f' % C, OneVsRestClassifier(SVC(C=C, cache_size=2000, class_weight=None, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False), n_jobs=-1)))

            self.logger.info('=' * 80)
            self.logger.info("LR.pen=l1.C=%f" % C)
            results.append(self.benchmark('LR.pen=l1.C=%f' % C, LogisticRegression(solver="liblinear", multi_class='ovr', penalty='l1', C=C)))
            # results.append(self.benchmark('LR.pen=l1.C=%f' % C, OneVsRestClassifier(LogisticRegression(solver="liblinear", multi_class='ovr', penalty='l1', C=C), n_jobs=-1)))

            self.logger.info('=' * 80)
            self.logger.info("LR.pen=l2.C=%f" % C)
            results.append(self.benchmark('LR.pen=l2.C=%f' % C, LogisticRegression(solver="lbfgs", multi_class='multinomial', penalty='l2', C=C, dual=False)))

        self.logger.info('=' * 80)
        self.logger.info("LinearSVC with L1-based feature selection, C=%f" % 1)
        results.append(self.benchmark('LinearSVC+L1-FeatSel, C=%f' % C, Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", tol=1e-3, dual=False, C=1, max_iter=1000))),
            ('classification', LinearSVC(penalty="l2"))
        ])))

        for penalty in ["l2", "l1"]:
            self.logger.info('=' * 80)
            self.logger.info("%s penalty" % penalty.upper())

            # Train Liblinear model
            results.append(self.benchmark('LinearSVC.loss=l2.penalty=%s' % penalty, LinearSVC(loss='l2', penalty=penalty,
                                               dual=False, tol=1e-3)))

            # Train SGD model
            results.append(self.benchmark('SGDClassifier.penalty=%s' % penalty, SGDClassifier(alpha=.0001, n_iter=50,
                                                   penalty=penalty)))

        # Train SGD with Elastic Net penalty
        self.logger.info('=' * 80)
        self.logger.info("Elastic-Net penalty")
        results.append(self.benchmark('SGDClassifier.penalty=%s' % 'elasticnet', SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty="elasticnet")))

        # Train NearestCentroid without threshold
        self.logger.info('=' * 80)
        self.logger.info("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark('NearestCentroid', NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        self.logger.info('=' * 80)
        self.logger.info("Naive Bayes")
        results.append(self.benchmark('MultinomialNB', MultinomialNB(alpha=.01)))
        results.append(self.benchmark('BernoulliNB', BernoulliNB(alpha=.01)))

        self.logger.info('=' * 80)
        self.logger.info("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        # The larger C, the more complex the model is
        results.append(self.benchmark('LinearSVC+L1-FeatSel', Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
            ('classification', LinearSVC(penalty="l2"))
        ])))

        '''

        '''
        for C in [0] + [2 ** x for x in [0, 2, 4]]:  # [0]+[2**x for x in [-3, -2, -1, 0]]
            self.logger.info('=' * 80)
            self.logger.info("LR.pen=l1.C=%f" % C)
            results.append(self.benchmark('LR.pen=l1.C=%d' % C,
                      LogisticRegression(solver="liblinear", multi_class='multinomial', penalty='l1', C=C)))

            self.logger.info('=' * 80)
            self.logger.info("LinearSVC.pen=l1, C=%d" % C)
            results.append(self.benchmark('LinearSVC.pen=l1.C=%d' % C,
                                          LinearSVC(loss='squared_hinge', penalty='l1', dual=True, tol=1e-3, C=C)))

            self.logger.info('=' * 80)
            self.logger.info("LinearSVC.pen=l2, C=%d" % C)
            results.append(self.benchmark('LinearSVC.pen=l2.C=%d' % C,
                                          LinearSVC(loss='hinge', penalty='l2', dual=True, tol=1e-3, C=C)))

            self.logger.info('=' * 80)
            self.logger.info("RBF SVC with C=%f" % C)
            results.append(self.benchmark('RBF SVC with C=%f' % C, SVC(C=C, cache_size=200, class_weight=None,
                                          degree=3, gamma='auto', kernel='rbf',
                                          max_iter=-1, probability=False, random_state=None, shrinking=True,
                                          tol=0.001, verbose=False)))

        '''

        return results

    def run_single_pass(self, X, Y):
        X = np.nan_to_num(X.todense())

        train_id, valid_id, test_id = self.load_single_run_index(X, Y)

        global X_train, Y_train, X_test, Y_test, X_valid, Y_valid
        X_train = np.nan_to_num(preprocessing.scale(X[train_id]))
        Y_train = Y[train_id]

        # validation set
        X_valid = np.nan_to_num(preprocessing.scale(X[valid_id]))
        Y_valid = Y[valid_id]
        # test set
        X_test = np.nan_to_num(preprocessing.scale(X[test_id]))
        Y_test = Y[test_id]
        results = self.run_experiment()

        self.export_single_pass_results(results)
        return

    def run_single_pass_context_feature(self, X, Y):
        X = np.nan_to_num(X.todense())

        train_id, valid_id, test_id = self.load_single_run_index(X, Y)
        global X_train, Y_train, X_test, Y_test, X_valid, Y_valid

        if self.config.param['context_set'] == 'current':
            excluded_context_keywords = ['next', 'last']
        if self.config.param['context_set'] == 'next':
            excluded_context_keywords = ['last']
        if self.config.param['context_set'] == 'last':
            excluded_context_keywords = ['next']
        if self.config.param['context_set'] == 'all':
            excluded_context_keywords = []

        retained_feature_indices = []
        retained_feature_names = []

        # get the features
        for f_id,f_name in enumerate(self.config['feature_names']):
            f_start_number = f_name[0:f_name.find('-')]
            if f_start_number.find('.') > 0:
                f_start_number = f_name[0:f_start_number.find('.')]

            if f_start_number not in self.config.param['feature_set_number']:
                continue

            if not self.config.param['similarity_feature'] and (f_name.find('similarity') > 0 or f_name.find('overlap') > 0 or f_name.find('distance') > 0):
                continue

            within_context = True
            for context_keyword in excluded_context_keywords:
                if f_name.find(context_keyword) > 0:
                    within_context = False
            if not within_context:
                continue

            retained_feature_indices.append(f_id)
            retained_feature_names.append(f_name)


        X_new = copy.deepcopy(X)[:, retained_feature_indices]
        self.logger.info('%' * 50)
        self.logger.info('retained features: [%s]' % (','.join(retained_feature_names)))
        self.logger.info('context=%s, feature=%s, #=%d' % (self.config.param['context_set'], self.config.param['feature_set'], len(retained_feature_indices)))
        self.logger.info('retained feature numbers=[%s]' % ', '.join(list(set([f[0:f.find('-')] for f in retained_feature_names]))))
        self.logger.info('X_new.shape=%s' % str(X_new.shape))
        self.logger.info('%' * 50)

        X_train = np.nan_to_num(preprocessing.scale(X_new[train_id]))
        Y_train = Y[train_id]
        # validation set
        X_valid = np.nan_to_num(preprocessing.scale(X_new[valid_id]))
        Y_valid = Y[valid_id]
        # test set
        X_test = np.nan_to_num(preprocessing.scale(X_new[test_id]))
        Y_test = Y[test_id]
        result = self.run_experiment()

        self.export_single_pass_results(result)
        return

    def run_single_pass_doc2vec(self, X_raw_feature, Y):
        train_id, valid_id, test_id = self.load_single_run_index(X_raw_feature, Y)

        documents = []
        selector = ItemSelector(keys=self.config['utterance_range'])
        for item_no, sents in enumerate(zip(*selector.transform(X_raw_feature))):
            sents = '. '.join(sents) # can be multiple sentences
            doc = TaggedDocument(words=gensim.utils.to_unicode(sents).split(), tags=[item_no])
            documents.append(doc)

        X_train_docs = [documents[i] for i in train_id]
        X_valid_docs = [documents[i] for i in valid_id]
        X_test_docs  = [documents[i] for i in test_id]

        '''
        Load or train Doc2Vec
        '''
        if os.path.exists(self.config['d2v_model_path'] % self.config['data_name']) and os.path.exists(self.config['d2v_vector_path'] % self.config['data_name']):
            d2v_model  = Doc2Vec.load(self.config['d2v_model_path'] % self.config['data_name'])
            d2v_vector = data_loader.deserialize_from_file(self.config['d2v_vector_path'] % self.config['data_name'])
        else:
            # d2v_model = Doc2Vec(size=self.config['d2v_vector_length'], window=self.config['d2v_window_size'], min_count=self.config['d2v_min_count'], workers=4, alpha=0.025, min_alpha=0.025) # use fixed documents rate
            d2v_model = Doc2Vec(size=self.config['d2v_vector_length'], window=self.config['d2v_window_size'], min_count=self.config['d2v_min_count'], workers=4)
            d2v_model.build_vocab(X_train_docs)
            d2v_model.intersect_word2vec_format(self.config['w2v_path'], binary=True)
            for epoch in range(10):
                d2v_model.train(X_train_docs, total_examples=len(X_train_docs), epochs=1)
                # d2v_model.alpha -= 0.002  # decrease the learning rate
                # d2v_model.min_alpha = d2v_model.alpha  # fix the learning rate, no decay

            d2v_vector = [[] for i in range(len(documents))]
            for id in train_id:
                d2v_vector[id] = d2v_model.docvecs[id]
            for id, doc in zip(valid_id, X_valid_docs):
                d2v_vector[id] = d2v_model.infer_vector(doc.words)
            for id, doc in zip(test_id, X_test_docs):
                d2v_vector[id] = d2v_model.infer_vector(doc.words)

            # store the model to mmap-able files
            d2v_model.save(self.config['d2v_model_path'] % self.config['data_name'])
            data_loader.serialize_to_file(d2v_vector, self.config['d2v_vector_path'] % self.config['data_name'])


        '''
        Training and Testing
        '''
        X = np.asarray(d2v_vector)
        cv_results = []
        global X_train, Y_train, X_test, Y_test
        X_train = X[train_id]
        Y_train = Y[train_id]

        # run experiment on validation set
        X_test = X[valid_id]
        Y_test = Y[valid_id]
        valid_result = self.run_experiment()
        cv_results.append(valid_result)

        print('*' * 50 + '\nValidation Performance \n ' + str(valid_result) + '\n' + '*' * 50)

        # run experiment on test set
        X_test = X[test_id]
        Y_test = Y[test_id]
        test_result = (self.run_experiment())
        cv_results.append(test_result)
        print('*' * 50 + '\nTest Performance \n ' + str(test_result) + '\n' + '*' * 50)

        avg_results = self.average_results(cv_results)
        return avg_results


    def run_cross_validation_bad_case(self):
        '''
        Compare the prediction and ground-truth, find the bad cases and print their features as well
        :return:
        '''
        feature_names   = self.config['feature_names']
        label_encoder   = self.config['label_encoder']
        X_raw           = self.config['X_raw']
        X_raw_feature   = self.config['X_raw_feature']
        X               = self.config['X']
        Y               = self.config['Y']

        exp_path = 'output/feature_selection/continuous/15-[5+1.2.3.4]/20180123-220214.continuous_feature_selection.pca_component=5.feature_number=8.context=next.feature=15-[5+1.2.3.4].similarity=true/'
        self.logger.info(os.path.abspath(exp_path))
        pred_file = open(exp_path + self.config['data_name'] + '.test.pred.txt', 'r')
        truth_file = open(exp_path + self.config['data_name'] + '.test.truth.txt', 'r')

        print('*' * 50)
        print(self.config['data_name'])
        print('*' * 50)
        labels   = self.config['label_encoder'].classes_

        total_counter = 0
        detail_counter = {}

        # features = extractor.load_printable_features()
        csv_file = open(self.config['experiment_path']+ '/' + self.config['data_name'] + '.bad_case.csv', 'w')
        csv_file.write('data_id,session_id,user_id,time,direction,msg_text,ground-truth,prediction,annotion,note\n')

        for did, (y_preds, y_test) in enumerate(zip(pred_file.readlines(), truth_file.readlines())):
            # print('%s #%d' % (self.config['data_name'], did))
            x               = X[did]
            x_raw           = X_raw[did]
            x_raw_feature   = X_raw_feature[did]
            y               = Y[did]
            assert y == int(y_test)
            y_preds         = [labels[int(p.strip())] for p in y_preds.split(',')]
            y_test          = labels[int(y_test.strip())]


            num_wrong       = sum([pred != y_test for pred in y_preds])
            if num_wrong != 5:
                continue

            total_counter += 1
            detail_counter['%s -> %s' % (y_preds, y_test)] = detail_counter.get('%s -> %s' % (y_preds, y_test), 0) + 1

            print('pred: %s' % y_preds)
            print('truth: %s' % y_test)
            print('num_wrong: %d' % num_wrong)

            dialogue    = x_raw['dialogue']
            utterance   = x_raw['utterance']

            csv_file.write('%d, %s, , , , , %s, "%s", , \n' % (did, dialogue.session_id, y_test, y_preds))

            self.logger.info('*' * 50)
            self.logger.info(' ' * 10 + 'index=%d, true label=%s, predicted=%s' % (did, y_test, y_preds))
            self.logger.info('*' * 50)
            for u_id, u in enumerate(dialogue):
                if x_raw['utterance'] != u:
                    self.logger.info('\t [%s][%s] %s' % (u.time, u.direction, u.msg_text))
                    csv_file.write('%d, %s, %s, %s, %s, "%s", , , , \n' % (did, dialogue.session_id, u.userid, u.time, u.direction, u.msg_text))
                else:
                    self.logger.info('')
                    self.logger.info('\t [%s][%s] %s' % (u.time, u.direction, u.msg_text))
                    self.logger.info('')
                    csv_file.write('%d, %s, %s, %s, %s, "%s", %s, "%s", , \n' % (did, dialogue.session_id, u.userid, u.time, u.direction, u.msg_text, y_test, y_preds))

            csv_file.write('\n')

            print_printable_features(self.config['utterance_range'], x_raw_feature)

            self.logger.info('-' * 50)

            self.logger.info('Find %d data points in %s that classifier fails to predict' % (total_counter, self.config['data_name']))
        sorted_counter = sorted(detail_counter.items(), key=lambda x:x[1], reverse=True)
        for k,v in sorted_counter:
            self.logger.info('\t%s : %d' % (k, v))

        csv_file.close()

        return []

    def run_cross_validation_bad_case_deprecated(self, X, Y):
        '''
        Old version, actually it only outputs the predicted labels
        :param X:
        :param Y:
        :return:
        '''
        train_ids, test_ids = self.load_cv_index(X, Y)
        cv_results  = []
        y_preds     = []

        diff_idx_cache_path = os.path.join(self.config.param['root_path'], 'dataset', 'bad_case_index',
                                           self.config.param[
                                               'data_name'] + '.%s.index_cache.evenly.#div=%d.#cv=%d.pkl' % (
                                               self.config['experiment_mode'], self.config['#division'],
                                               self.config['#cross_validation']))
        if os.path.exists(diff_idx_cache_path):
            with open(diff_idx_cache_path, 'rb') as idx_cache:
                cv_results, diff_idx, test_idx, y_tests, y_preds = pickle.load(idx_cache)
        else:
            global X_train, Y_train, X_test, Y_test
            for r_id, (train_id, test_id) in enumerate(zip(train_ids, test_ids)):
                self.logger.info('*' * 20 + ' %s - Round %d ' % (self.config['data_name'], r_id))
                X_train = np.nan_to_num(preprocessing.scale(X[train_id]))
                Y_train = Y[train_id]
                X_test = np.nan_to_num(preprocessing.scale(X[test_id]))
                Y_test = Y[test_id]

                score, y_pred = self.run_experiment_bad_case()
                cv_results.append([score])
                y_preds.extend(y_pred)

            test_idx = np.concatenate(test_ids)
            y_tests  = Y[test_idx]
            diff_idx = np.where([y_p != y_t for y_p,y_t in zip(y_preds, y_tests)])

            with open(diff_idx_cache_path, 'wb') as idx_cache:
                pickle.dump([cv_results, diff_idx, test_idx, y_tests, y_preds], idx_cache)

        labels   = self.config['label_encoder'].classes_

        for x_raw, y_pred, y_test in zip(np.asarray(self.config['X_raw'])[diff_idx], np.asarray(y_preds)[diff_idx], np.asarray(y_tests)[diff_idx]):
            x_index     = 0 #x_raw['index'], wrong!!!
            dialogue    = x_raw['dialogue']
            utterance   = x_raw['utterance']

            self.logger.info('*' * 50)
            self.logger.info(' ' * 10 + 'index=%d, true label=%s, predicted=%s' % (x_index, labels[y_test], labels[y_pred]))
            self.logger.info('*' * 50)
            for u_id, u in enumerate(dialogue):
                if x_raw['utterance'] != u:
                    self.logger.info('\t [%s][%s] %s' % (u.time, u.direction, u.msg_text))
                else:
                    self.logger.info('\t [%s][%s][True=%s,Pred=%s] %s' % (u.time, u.direction, labels[y_test], labels[y_pred], u.msg_text))
            self.logger.info('*' * 50)

        # get the average score of cross-validation
        avg_results = self.average_results(cv_results)

        return avg_results

    def run_cross_validation_binary_task(self, X, Y):
        X = np.nan_to_num(X.todense())
        label_encoder = self.config['label_encoder']

        if self.config['experiment_mode'] == 'reformulation_detection':
            positive_labels = ['R']
            negative_labels = ['A', 'C', 'F']
            classes_ = np.asarray(['reform', 'non-reform'])
        elif self.config['experiment_mode'] == 'task_boundary_detection':
            positive_labels = ['F', 'A']
            negative_labels = ['C', 'R']
            classes_ = np.asarray(['end', 'ongoing'])
        label_to_y = dict([(l, y) for y, l in enumerate(label_encoder.classes_)])
        Y_new = copy.deepcopy(Y)

        for l in positive_labels:
            Y_new[np.where(Y == label_to_y[l])] = 0
        for l in negative_labels:
            Y_new[np.where(Y == label_to_y[l])] = 1

        label_encoder.classes_ = classes_
        Y = Y_new
        self.config['label_encoder'] = label_encoder

        train_ids, test_ids = self.load_cv_index(X, Y)
        cv_results = []

        global X_train, Y_train, X_test, Y_test
        for r_id, (train_id, test_id) in enumerate(zip(train_ids, test_ids)):
            self.logger.info('*' * 20 + ' %s - Round %d ' % (self.config['data_name'], r_id))
            X_train = np.nan_to_num(preprocessing.scale(X[train_id]))
            Y_train = Y[train_id]
            X_test = np.nan_to_num(preprocessing.scale(X[test_id]))
            Y_test = Y[test_id]

            cv_results.append(self.run_experiment())

        # get the average score of cross-validation
        avg_results = self.average_results(cv_results)

        return avg_results

    def report_feature_importance(self, X, Y, feature_names):
        '''
        Run chi-square for discrete features and ANOVA f-test for similarity features
        :param X:
        :param Y:
        :param retained_feature_indices:
        :param retained_feature_names:
        :param k_feature_to_keep:
        :return:
        '''
        '''
        keep discrete features for selection only (1-7), note that LDA is discrete as well but we don't select it
        '''
        selectable_feature_indices = []
        selectable_feature_names = []
        not_selectable_feature_names = []
        for f_id, f_name in enumerate(feature_names):
            f_series = f_name[: f_name.find('-')]
            if f_series.find('.') > 0:
                f_series = f_series[: f_series.find('.')]
            if f_series in ['1', '2', '3', '4', '5', '6', '7'] and not (f_name.find('similarity') > 0 or f_name.find('overlap') > 0 or f_name.find('distance') > 0):
                selectable_feature_indices.append(f_id)
                selectable_feature_names.append(f_name)
            else:
                not_selectable_feature_names.append(f_name)

        X_selectable        =   copy.deepcopy(X)[:,selectable_feature_indices]
        X_not_selectable    =   np.delete(copy.deepcopy(X), selectable_feature_indices, axis=1)

        '''
        print_important_features
        '''
        chi2_stats, pvals   = chi2(X_selectable, Y)
        chi2_stats[np.where(np.isnan(chi2_stats))] = 0.0

        sorted_idx = np.argsort(chi2_stats)[::-1]

        with open(os.path.join(self.config['experiment_path'], '%s.chi2_top_features.csv' % self.config['data_name']), 'w') as csv_writer:
            csv_writer.write('id,prefix,name,chi2,pval\n')
            for f_id, f_name, chi2_stat, pval in zip(np.asarray(selectable_feature_indices)[sorted_idx], np.asarray(selectable_feature_names)[sorted_idx], chi2_stats[sorted_idx], pvals[sorted_idx]):
                # self.logger.info('%d\t%s\t%.4f\t%.4f\n' % (f_id, f_name, chi2_stat, pval))
                csv_writer.write('%d,%s,%s,%.4f,%.4f\n' % (f_id, str(f_name.encode('utf-8')), str(f_name[:f_name.find('-')]), chi2_stat, pval))

        feature_prefixes  = sorted(list(set([f[:f.find('-')] for f in feature_names])))
        feature_set_names = {'1'   :'1-utterance length',
                             '2.1' :'2.1-user action words', '2.2': '2.2-number of user action words', '2.3.1':'2.3.1-action_jaccard_similarity.next_user_utterance_pairs', '2.3.2':'2.3.2-action_jaccard_similarity.last_user_utterance_pairs',
                             '3'   :'3-time features',
                             '4.1' :'4.1-n_gram', '4.2.1':'4.2.1-edit_distance.next_user_utterance_pairs', '4.2.2':'4.2.2-edit_distance.last_user_utterance_pairs',
                             '4.3.1' :'4.3.1-jaccard_similarity.next_user_utterance_pairs', '4.3.2' :'4.3.2-jaccard_similarity.last_user_utterance_pairs',
                             '5'   :'5-noun phrase', '6':'6-entity', '7':'7-syntactic features',
                             '8.1' :'8.1-LDA_features', '8.2.1':'8.2.1-lda_similarity.next_user_utterance_pairs', '8.2.2':'8.2.2-lda_similarity.last_user_utterance_pairs',
                             '9.1' :'9.1-w2v_features', '9.2.1':'9.2.1-w2v_similarity.next_user_utterance_pairs','9.2.2':'9.2.2-w2v_similarity.last_user_utterance_pairs', '9.3.1':'9.3.1-wmd_similarity.next_user_utterance_pairs','9.3.2':'9.3.2-wmd_similarity.last_user_utterance_pairs',
                             '10.1': '10.1-d2v_features', '10.2.1': '10.2.1-d2v_similarity.next_user_utterance_pairs','10.2.2': '10.2.2 d2v_similarity.last_user_utterance_pairs',
                             '11.1': '11.1-skipthought_features', '11.2.1': '11.2.1-skipthought_similarity.next_user_utterance_pairs', '11.2.2': '11.2.2-skipthought_similarity.last_user_utterance_pairs'
                             }

        if os.path.exists(os.path.join(self.config['experiment_path'], 'feature_stats.csv')):
            print_header = False
        else:
            print_header = True

        with open(os.path.join(self.config['experiment_path'], 'feature_stats.csv'), 'a') as csv_writer:
            if print_header:
                csv_writer.write(',%s\n' % (','.join(feature_prefixes)))
                csv_writer.write(',%s\n' % (','.join([feature_set_names[p_] for p_ in feature_prefixes])))
            num_feature = []
            for prefix, feature_set_name in zip(feature_prefixes, feature_set_names):
                self.logger.info('%s\t%d\n' % (prefix, len([f for f in feature_names if f[:f.find('-')] == prefix])))
                # [print(f) for f in feature_names if f[:f.find('-')] == prefix]
                # csv_writer.write('%s\t%d\n' % (prefix, len([f for f in feature_names if f.startswith(prefix)])))
                num_feature.append(len([f for f in feature_names if f[:f.find('-')] == prefix]))
            csv_writer.write('%s,%s\n' % (self.config['data_name'], ','.join([str(n) for n in num_feature])))

        '''
        do ANAVA f-test for similarity features
        '''
        selectable_feature_indices = []
        selectable_feature_names = []
        not_selectable_feature_names = []
        for f_id, f_name in enumerate(feature_names):
            f_series = f_name[: f_name.find('-')]
            if f_series.find('.') > 0:
                f_series = f_series[: f_series.find('.')]
            if f_name.find('similarity') > 0 or f_name.find('overlap') > 0 or f_name.find('distance') > 0:
                selectable_feature_indices.append(f_id)
                selectable_feature_names.append(f_name)
            else:
                not_selectable_feature_names.append(f_name)

        X_selectable     = copy.deepcopy(X)[:, selectable_feature_indices]
        X_selectable     = np.asarray(X_selectable, dtype='float32')
        X_not_selectable = np.delete(copy.deepcopy(X), selectable_feature_indices, axis=1)

        X_selectable[np.where(np.isnan(X_selectable))] = 0.0
        X_selectable[np.where(np.isinf(X_selectable))] = 0.0

        # ANOVA F-value for the four classes
        f_stats, pvals   = f_classif(X_selectable, Y)
        sorted_idx = np.argsort(f_stats)[::-1]

        with open(os.path.join(self.config['experiment_path'], '%s.ftest_top_features.csv' % self.config['data_name']), 'w') as csv_writer:
            csv_writer.write('id,name,prefix,f-test,pval\n')
            for f_id, f_name, f_stat, pval in zip(np.asarray(selectable_feature_indices)[sorted_idx], np.asarray(selectable_feature_names)[sorted_idx], f_stats[sorted_idx], pvals[sorted_idx]):
                csv_writer.write('%d,%s,%s,%.4f,%.4f\n' % (f_id, str(f_name), str(f_name[:f_name.find('-')]), f_stat, pval))

        # f_oneway on each class
        for k in np.unique(Y):
            label = self.config['label_encoder'].classes_[k]
            arg_x = [X_selectable[Y == k], X_selectable[Y != k]]
            f_stats, pvals = f_oneway(*arg_x)
            sorted_idx = np.argsort(f_stats)[::-1]

            with open(os.path.join(self.config['experiment_path'],
                                   '%s.class=%s.ftest_top_features.csv' % (self.config['data_name'], label)), 'w') as csv_writer:
                csv_writer.write('id,name,prefix,f-test,pval\n')
                for f_id, f_name, f_stat, pval in zip(np.asarray(selectable_feature_indices)[sorted_idx],
                                                      np.asarray(selectable_feature_names)[sorted_idx],
                                                      f_stats[sorted_idx], pvals[sorted_idx]):
                    csv_writer.write(
                        '%d,%s,%s,%.4f,%.4f\n' % (f_id, str(f_name), str(f_name[:f_name.find('-')]), f_stat, pval))

        return []

    def run_cross_validation_with_discrete_feature_selection(self, X_original, Y, retained_feature_indices, retained_feature_names, k_feature_to_keep):
        ''''''
        '''
        keep discrete features for selection only (1-7), note that LDA is discrete as well but we don't select it
        '''
        selectable_feature_indices = []
        selectable_feature_names = []
        not_selectable_feature_indices = []
        not_selectable_feature_names = []
        for f_id, f_name in enumerate(retained_feature_names):
            f_series = f_name[: f_name.find('-')]
            if f_series.find('.') > 0:
                f_series = f_series[: f_series.find('.')]
            if f_series in ['1', '2', '3', '4', '5', '6', '7'] and not (f_name.find('similarity') > 0 or f_name.find('overlap') > 0 or f_name.find('distance') > 0):
                selectable_feature_indices.append(f_id)
                selectable_feature_names.append(f_name)
            else:
                not_selectable_feature_indices.append(f_id)
                not_selectable_feature_names.append(f_name)

        X_selectable        =   copy.deepcopy(X_original)[:,selectable_feature_indices]
        X_not_selectable    =   np.delete(copy.deepcopy(X_original), selectable_feature_indices, axis=1)

        '''
        run experiment with selected features
        '''
        X_to_select     = copy.deepcopy(X_selectable)

        # if num_feature_to_keep is -1, we don't select anything and keep all the features
        if k_feature_to_keep != -1:
            num_feature     = 2 ** k_feature_to_keep
            X_selected      = SelectKBest(chi2, k=num_feature).fit_transform(X_to_select, Y)
        else:
            num_feature     = X_to_select.shape[1]
            X_selected      = X_to_select

        X_selected      = np.nan_to_num(X_selected)
        if X_not_selectable.shape[1] == 0:
            X           = X_selected
        else:
            X           = np.concatenate([X_selected, X_not_selectable], axis=1)

        self.logger.info('%' * 50)
        self.logger.info(' ' * 10 + 'dataset=%s' % str(self.config['data_name']))
        self.logger.info(' ' * 10 + 'k_feature_to_keep=%s' % str(k_feature_to_keep))
        self.logger.info(' ' * 10 + 'X_new.shape=%s' % str(X.shape))
        self.logger.info(' ' * 10 + '#(X_selected)=%d' % num_feature)
        self.logger.info(' ' * 10 + '#(X_not_selectable)=%d' % X_not_selectable.shape[1])
        self.logger.info('%' * 50)

        cv_results = self.run_cross_validation(X, Y)

        '''
        print_important_features
        '''
        chi2_stats, pvals        = chi2(X_selectable, Y)
        chi2_stats[np.where(np.isnan(chi2_stats))] = 0.0
        sorted_selectable_idx    = np.argsort(chi2_stats)[::-1]
        selected_idx             = sorted(sorted_selectable_idx[:num_feature])
        selected_chi2_stats      = chi2_stats[selected_idx]
        selected_pvals           = pvals[selected_idx]

        selectable_feature_indices      = np.asarray(selectable_feature_indices)
        selectable_feature_names        = np.asarray(selectable_feature_names)
        not_selectable_feature_indices  = np.asarray(not_selectable_feature_indices)
        not_selectable_feature_names    = np.asarray(not_selectable_feature_names)

        # these indices correspond to the original feature order (all features)
        selected_feature_indices = selectable_feature_indices[selected_idx]
        selected_feature_names   = selectable_feature_names[selected_idx]

        self.print_feature_importance_report(cv_results, selected_chi2_stats, selected_pvals, selected_feature_indices, selected_feature_names, not_selectable_feature_indices, not_selectable_feature_names)

        return cv_results

    def print_feature_importance_report(self, results, chi2_stats, pvals, selected_feature_indices, selected_feature_names, not_selectable_feature_indices, not_selectable_feature_names):

        clf_weights = np.asarray([result[0]['coef'] for result in results])
        clf_weights = np.concatenate(clf_weights, axis=0) # concatenate them
        clf_weights = (clf_weights ** 2).sum(axis=0)
        clf_weights /= clf_weights.max()
        clf_weights_ranks = np.argsort(np.argsort(clf_weights)[::-1])
        clf_weights_argsort = np.argsort(clf_weights)[::-1]

        feature_indices = np.concatenate([selected_feature_indices, not_selectable_feature_indices])
        feature_names   = np.concatenate([selected_feature_names, not_selectable_feature_names])
        chi2_stats      = np.concatenate([chi2_stats, np.zeros(not_selectable_feature_names.shape)])
        chi2_stats_ranks = np.argsort(np.argsort(chi2_stats)[::-1])

        with open(os.path.join(self.config['experiment_path'], '%s.feature_importance_after_classification.csv' % self.config['data_name']), 'w') as csv_writer:
            csv_writer.write('id,name,prefix,chi2,chi2_rank,clf_weight,clf_weight_rank\n')
            for f_id, f_name, chi2_stat, chi2_stats_rank, clf_weight, clf_weights_rank in zip(feature_indices[clf_weights_argsort], feature_names[clf_weights_argsort], chi2_stats[clf_weights_argsort], chi2_stats_ranks[clf_weights_argsort], clf_weights[clf_weights_argsort], clf_weights_ranks[clf_weights_argsort]):
                # self.logger.info('%d\t%s\t%.4f\t%.4f\n' % (f_id, f_name, chi2_stat, pval))
                prefix = f_name[: f_name.find('-')]
                csv_writer.write('%d,%s,%s,%.4f,%d,%.4f,%d\n' % (f_id, prefix, f_name.encode('utf-8'), chi2_stat, chi2_stats_rank, clf_weight, clf_weights_rank))

        top_K = 100
        if top_K > len(chi2_stats):
            top_K = len(chi2_stats)

        X_indices = np.arange(top_K)

        chi2_stats       = chi2_stats / chi2_stats.max()
        chi2_stats_topk  = chi2_stats[clf_weights_argsort[:top_K]]
        clf_weights_topk = clf_weights[clf_weights_argsort[:top_K]]
        num_chi2_stats_topk = sum(chi2_stats_topk > 0)

        plt.figure(num=1, dpi=200)
        plt.clf()

        plt.bar(X_indices - .45, chi2_stats_topk, width=.2, label='chi2', color='darkorange')
        plt.bar(X_indices - .25, clf_weights_topk, width=.2, label='LR weight', color='navy')

        self.logger.info("Feature selection of top %d, selectable:non-selectable=%d:%d" % (top_K, num_chi2_stats_topk, top_K-num_chi2_stats_topk))

        plt.title("Feature selection of top %d, selectable:non-selectable=%d:%d" % (top_K, num_chi2_stats_topk, top_K-num_chi2_stats_topk))
        plt.xlabel('Features')
        plt.yticks(())
        plt.axis('tight')
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(os.path.join(self.config['experiment_path'], '%s.feature_importance_after_classification.png' % self.config['data_name']))

        with open(os.path.join(self.config['experiment_path'], 'top_important_feature_stats.csv'), 'a') as csv_writer:
            k_range = [100, 200, 300, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000]
            if len(chi2_stats) < k_range[0]:
                k_range = [len(chi2_stats)] + k_range
            for top_K in k_range:
                if top_K > len(chi2_stats):
                    break
                chi2_stats_topk  = chi2_stats[np.argsort(clf_weights)[::-1][:top_K]]
                num_chi2_stats_topk = sum(chi2_stats_topk > 0)
                csv_writer.write('%s,%s,%d,%d,%d,%.4f,%.4f\n' % (self.config['data_name'] + str(top_K), self.config['data_name'], top_K, num_chi2_stats_topk, top_K-num_chi2_stats_topk, num_chi2_stats_topk/float(top_K) * 100.0, (top_K-num_chi2_stats_topk)/float(top_K) * 100.0))


    def run_cross_validation_with_continuous_feature_selection(self, X_original, Y, retained_feature_indices, retained_feature_names, k_feature_to_keep, k_component_for_pca):
        ''''''
        '''
        keep continuous features for selection only including: 8.LDA 9. w2v 10. d2v 11. skip-thought
        '''
        discrete_feature_indices = []
        discrete_feature_names = []
        continuous_feature_indices = []
        continuous_feature_names = []
        similarity_feature_indices = []
        similarity_feature_names = []
        for f_id, f_name in enumerate(retained_feature_names):
            f_series = f_name[: f_name.find('-')]
            if f_series.find('.') > 0:
                f_series = f_series[: f_series.find('.')]

            if f_name.find('similarity') > 0 or f_name.find('overlap') > 0 or f_name.find('distance') > 0:
                similarity_feature_indices.append(f_id)
                similarity_feature_names.append(f_name)
            elif f_series in ['1', '2', '3', '4', '5', '6', '7']:
                discrete_feature_indices.append(f_id)
                discrete_feature_names.append(f_name)
            elif f_series in ['8', '9', '10', '11']:
                continuous_feature_indices.append(f_id)
                continuous_feature_names.append(f_name)
            else:
                assert False, "feature bug: %s - %s" % (f_id, f_name)

        X_discrete        =   copy.deepcopy(X_original)[:,discrete_feature_indices]
        X_continuous      =   copy.deepcopy(X_original)[:,continuous_feature_indices]
        X_similarity      =   copy.deepcopy(X_original)[:,similarity_feature_indices]

        '''
        run experiment with selected features
        '''
        # if num_feature_to_keep is -1, we don't select anything and keep all the features
        if k_feature_to_keep != -1:
            num_discrete_feature     = 2 ** k_feature_to_keep
            X_discrete      = SelectKBest(chi2, k=num_discrete_feature).fit_transform(X_discrete, Y)
        else:
            num_discrete_feature     = X_discrete.shape[1]
            X_discrete      = X_discrete
        X_discrete      = np.nan_to_num(X_discrete)

        '''
        run experiment with PCA feature reduction
        '''
        # if k_component_for_pca is -1, we don't select anything and keep all the features
        if k_component_for_pca != -1:
            num_continuous_feature = 2 ** k_component_for_pca
            pca           = PCA(n_components=num_continuous_feature, svd_solver='full')
            X_continuous  = pca.fit_transform(X_continuous)
            continuous_feature_names = continuous_feature_names[: num_continuous_feature]
        else:
            num_continuous_feature  = X_continuous.shape[1]
            X_continuous            = X_continuous
        X_continuous      = np.nan_to_num(X_continuous)
        X           = np.concatenate([X_discrete, X_continuous, X_similarity], axis=1)

        self.logger.info('%' * 50)
        self.logger.info(' ' * 10 + 'dataset=%s' % str(self.config['data_name']))
        self.logger.info(' ' * 10 + 'X_new.shape=%s' % str(X.shape))
        self.logger.info(' ' * 10 + '#(discrete_feature)=%d' % num_discrete_feature)
        self.logger.info(' ' * 10 + '#(continuous_feature)=%d' % num_continuous_feature)
        self.logger.info(' ' * 10 + '#(similarity_feature)=%d' % X_similarity.shape[1])
        self.logger.info('%' * 50)

        cv_results = self.run_cross_validation(X, Y)

        '''
        print_important_features
        '''
        if len(discrete_feature_indices) > 0:
            X_discrete               = copy.deepcopy(X_original)[:,discrete_feature_indices]
            chi2_stats, pvals        = chi2(X_discrete, Y)
            chi2_stats[np.where(np.isnan(chi2_stats))] = 0.0
            sorted_selectable_idx    = np.argsort(chi2_stats)[::-1]
            selected_idx             = sorted(sorted_selectable_idx[:num_discrete_feature])
            selected_chi2_stats      = chi2_stats[selected_idx]
            selected_pvals           = pvals[selected_idx]
        else:
            selected_idx = []
            selected_chi2_stats = []
            selected_pvals = []

        selectable_feature_indices      = np.asarray(discrete_feature_indices)
        selectable_feature_names        = np.asarray(discrete_feature_names)

        not_selectable_feature_indices  = np.concatenate([continuous_feature_indices, similarity_feature_indices])
        not_selectable_feature_names    = np.concatenate([continuous_feature_names, similarity_feature_names])

        # these indices correspond to the original feature order (all features)
        selected_feature_indices = selectable_feature_indices[selected_idx]
        selected_feature_names   = selectable_feature_names[selected_idx]

        self.print_feature_importance_report(cv_results, selected_chi2_stats, selected_pvals, selected_feature_indices, selected_feature_names, not_selectable_feature_indices, not_selectable_feature_names)

        return cv_results

    def run_cross_validation_with_leave_one_out(self, X, Y):
        X = np.nan_to_num(X.todense())

        train_ids, test_ids = self.load_cv_index(X, Y)
        cv_results = []
        global X_train, Y_train, X_test, Y_test

        feature_names = self.config['feature_names']
        if self.config['experiment_mode'] == 'keep_one_only':
            # skip the feature sets with too few features
            feature_series = ['0', '2', '4', '5', '7', '8']
            feature_series_names = ['0.all', '2.user_act', '4.n-gram', '5.phrasal', '7.syntactic', '8.semantic']
        elif self.config['experiment_mode'] == 'leave_one_out':
            feature_series = ['0', '1', '2', '3', '4', '5', '7', '8']
            feature_series_names = ['0.all', '1.utt_len', '2.user_act', '3.time', '4.n-gram', '5.phrasal', '7.syntactic', '8.semantic']

        avg_result_dict = {}

        # iterate the features to leave out
        for f_series, f_series_name in zip(feature_series, feature_series_names):
            if self.config['experiment_mode'] == 'keep_one_only':
                task_label_to_print = 'feature_to_keep'
                feature_indices = [f_id for f_id,f_name in enumerate(feature_names) if not f_name.startswith(f_series)]
                if f_series == '0': # merge the entity and phrasal features
                    feature_indices = []
                elif f_series == '5': # merge the entity and phrasal features
                    feature_indices = [f_id for f_id,f_name in enumerate(feature_names) if not f_name.startswith('5') and not f_name.startswith('6')]
            elif self.config['experiment_mode'] == 'leave_one_out':
                task_label_to_print = 'leave_out_features'
                feature_indices = [f_id for f_id,f_name in enumerate(feature_names) if f_name.startswith(f_series)]
                if f_series == '5': # merge the entity and phrasal features
                    feature_indices = [f_id for f_id,f_name in enumerate(feature_names) if f_name.startswith('5') or f_name.startswith('6')]

            X_new = np.delete(copy.deepcopy(X), feature_indices, axis=1)
            self.logger.info('%' * 50)
            self.logger.info(' '*10 + ' %s=%s.#=%d' % (task_label_to_print, f_series_name, len(feature_indices)))
            self.logger.info(' '*10 + ' X_new.shape=%s' % str(X_new.shape))
            self.logger.info('%' * 50)

            self.config['f_series_name'] = f_series_name
            self.config['#features'] = X_new.shape[1]

            # iterate folds for cross validation and get the average score
            for r_id, (train_id, test_id) in enumerate(zip(train_ids, test_ids)):
                self.logger.info('*' * 20 + ' %s - %s=%s, #(feature)=%d, Round %d ' % (self.config['data_name'], task_label_to_print, f_series_name, X_new.shape[1], r_id) + '*' * 20)
                X_train = np.nan_to_num(preprocessing.scale(X_new[train_id]))
                Y_train = Y[train_id]
                X_test = np.nan_to_num(preprocessing.scale(X_new[test_id]))
                Y_test = Y[test_id]
                cv_results.append(self.run_experiment())

            # get the average score of cross-validation
            avg_result_models = self.average_results(cv_results)
            for avg_result_model in avg_result_models:
                avg_results_of_model = avg_result_dict.get(avg_result_model['model'], [])
                avg_results_of_model.append(avg_result_model)
                avg_result_dict[avg_result_model['model']] = avg_results_of_model

        for model_name, model_result in avg_result_dict.items():
            # plot the result of each model
            fig, ax = plt.subplots()

            index = np.arange(len(feature_series))
            bar_width = 0.35
            error_config = {'ecolor': '0.3'}

            rects1 = ax.bar(index, [r['accuracy'] for r in model_result], bar_width,
                             color='r',
                             yerr=[r['accuracy_std'] for r in model_result],
                             error_kw=error_config,
                             label='Accuracy')
            rects2 = ax.bar(index + bar_width, [r['f1_score'] for r in model_result], bar_width,
                             color='b',
                             yerr=[r['f1_score_std'] for r in model_result],
                             error_kw=error_config,
                             label='F1-score')

            ax.set_title(
                'Performance of the %s by %s' % (model_name, task_label_to_print))
            ax.set_xlabel('Leave-out-feature')
            ax.set_ylabel('Prediction rate')

            plt.xticks(index + bar_width / 2, (feature_series_names))
            plt.legend()

            plt.tight_layout()
            # plt.show()

            def autolabel(rects):
                """
                Attach a text label above each bar displaying its height
                """
                for rect in rects:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
                            '%.3f' % float(height),
                            ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)

            ax.axis('tight')
            # plt.show()
            fig.savefig(os.path.join(self.config.param['experiment_path'], '%s.model=%s.%s.plot.jpg' % (self.config.param['data_name'], model_name, self.config.param['experiment_mode'])))

    def run_experiment_bad_case(self):
        '''

        :return: 'model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time'
        '''
        C = 1.0
        self.logger.info('=' * 80)
        self.logger.info("LR.pen=l1.C=%f" % C)
        # Train Logistic Regression model
        return self.k('LR.pen=l1.C=%d' % C,
                                      LogisticRegression(solver="liblinear", penalty='l1', C=C), return_y_pred = True)

    def export_single_pass_results(self, results):
        # field_names in results of benchmarks = ['dataset', 'model', 'valid_test', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time', 'report', 'confusion_mat', 'y_test', 'y_pred']
        field_names = ['dataset', 'context_set','feature_set', 'model', 'valid-accuracy', 'valid-precision', 'valid-recall', 'valid-f1_score', 'test-accuracy', 'test-precision', 'test-recall', 'test-f1_score']
        with open(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.valid_test.csv'), 'w') as csv_file:
            csv_file.write(','.join(field_names) + '\n')

            for valid, test in results:
                field_values = [self.config.param['data_name'], self.config.param['context_set'], self.config.param['feature_set'] + ' w/ similarity' if self.config.param['similarity_feature'] else self.config.param['feature_set'] + 'w/o similarity', valid['model']]
                [field_values.append(str(valid[k])) for k in ['accuracy', 'precision', 'recall', 'f1_score']]
                [field_values.append(str(test[k])) for k in ['accuracy', 'precision', 'recall', 'f1_score']]
                csv_file.write(','.join(field_values) + '\n')

        with open(os.path.join(self.config.param['experiment_path'], 'all.test.csv'), 'a') as csv_file:
            csv_file.write(','.join(field_names) + '\n')

            for valid, test in results:
                field_values = [self.config.param['data_name'], self.config.param['context_set'], self.config.param['feature_set'] + ' w/ similarity' if self.config.param['similarity_feature'] else self.config.param['feature_set'] + 'w/o similarity', valid['model']]
                [field_values.append(str(valid[k])) for k in ['accuracy', 'precision', 'recall', 'f1_score']]
                [field_values.append(str(test[k])) for k in ['accuracy', 'precision', 'recall', 'f1_score']]
                csv_file.write(','.join(field_values) + '\n')

        with open(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.valid_test.json'), 'w') as json_file:
            json.dump(results, json_file)

    def export_cv_results(self, results, test_ids, Y):
        # field_names in results of benchmarks = ['dataset', 'model', 'valid_test', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time', 'report', 'confusion_mat', 'y_test', 'y_pred']
        field_names = ['dataset', 'context_set','feature_set', 'model', 'test_round','accuracy', 'precision', 'recall', 'f1_score']

        '''
        export to individual files
        '''
        with open(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.test.csv'), 'w') as csv_file:
            csv_file.write(','.join(field_names) + '\n')

            for result in results:
                if self.config['deep_model']:
                    field_values = [self.config.param['data_name'], self.config.param['context_set'], ' ', result[0]['model'], result[0]['test_round']]
                else:
                    field_values = [self.config.param['data_name'], self.config.param['context_set'], self.config.param['feature_set'] + ' w/ similarity' if self.config.param['similarity_feature'] else self.config.param['feature_set'] + ' w/o similarity', result[0]['model'], result[0]['test_round']]

                [field_values.append(str(result[0][k])) for k in ['accuracy', 'precision', 'recall', 'f1_score']]
                field_values = [str(v).replace(',', '.') for v in field_values]
                csv_file.write(','.join(field_values) + '\n')

        '''
        Append all results to one file
        '''
        if os.path.exists(os.path.join(self.config.param['experiment_path'], 'all.test.csv')):
            print_header = False
        else:
            print_header = True

        with open(os.path.join(self.config.param['experiment_path'], 'all.test.csv'), 'a') as csv_file:
            if print_header:
                csv_file.write(','.join(field_names) + '\n')

            for result in results:
                if self.config['deep_model']:
                    field_values = [self.config.param['data_name'], self.config.param['context_set'], ' ', result[0]['model'], result[0]['test_round']]
                else:
                    field_values = [self.config.param['data_name'], self.config.param['context_set'], self.config.param['feature_set'] + ' w/ similarity' if self.config.param['similarity_feature'] else self.config.param['feature_set'] + ' w/o similarity', result[0]['model'], result[0]['test_round']]

                [field_values.append(str(result[0][k])) for k in ['accuracy', 'precision', 'recall', 'f1_score']]
                field_values = [str(v).replace(',', '.') for v in field_values]
                csv_file.write(','.join(field_values) + '\n')

        '''
        Export all the preds vs groundtruth for further error analysis
        '''
        x_raws   = [np.asarray(self.config['X_raw'])[t_id] for t_id in test_ids]
        y_tests  = [Y[t_id].tolist() for t_id in test_ids]
        y_preds  = [result[0]['y_pred'] for result in results]

        sorted_preds            = {}
        sorted_correctness      = {}
        # iterate each round of CV
        for x_raw, test_id, y_test, y_pred in zip(x_raws, test_ids, y_tests, y_preds):
            # iterate each prediction
            for id, test, pred in zip(test_id, y_test, y_pred):
                pred_list = sorted_preds.get(id, [])
                tf_list   = sorted_correctness.get(id, [])

                pred_list.append(pred)
                tf_list.append(1 if pred==test else 0)

                sorted_preds[id] = pred_list
                sorted_correctness[id] = tf_list

        with open(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.test.truth.txt'), 'w') as txt_file:
            txt_file.write('\n'.join([str(i) for i in Y.tolist()]))

        sorted_preds = sorted(sorted_preds.items(), key=lambda x:x[0])
        with open(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.test.pred.txt'), 'w') as txt_file:
            txt_file.write('\n'.join([','.join([str(t_) for t_ in t[1]]) for t in sorted_preds]))

        sorted_correctness = sorted(sorted_correctness.items(), key=lambda x:x[0])
        with open(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.test.corr.txt'), 'w') as txt_file:
            txt_file.write('\n'.join([','.join([str(t_) for t_ in t[1]]) for t in sorted_correctness]))

        '''
        pickle all these results
        '''
        with open(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.test.json'), 'w') as json_file:
            json.dump(results, json_file)
        # with open(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.test.pkl'), 'wb') as pkl_file:
        #     pickle.dump(results, pkl_file)

    def export_averaged_summary(self, results, csv_path):
        '''
        Average the test performances of all cv rounds for each model
        :param results:
        :param csv_path:
        :return:
        '''
        result_dict = {}
        field_names = ['dataset', 'context_set','feature_set', 'model', 'accuracy', 'precision', 'recall', 'f1_score']

        for result in results:
            result_key = result[0]['dataset']+'_'+result[0]['model']
            result_list = result_dict.get(result_key, [])
            result_list.append(result[0])
            result_dict[result_key] = result_list

        with open(csv_path, 'a') as csv_file:
            csv_file.write(','.join(field_names) + '\n')

            for exp_name, result in result_dict.items():
                field_values    = [result[0]['dataset'], result[0]['context_set'], result[0]['feature_set'], result[0]['model']]
                [field_values.append(str(np.average([r[k] for r in result]))) for k in ['accuracy', 'precision', 'recall', 'f1_score']]
                field_values    = [str(v).replace(',', '.') for v in field_values]
                csv_file.write(','.join(field_values) + '\n')

    @staticmethod
    def export_summary(results, csv_path):
        if not os.path.exists(csv_path):
            print_header = True
        else:
            print_header = False

        field_names = ['dataset', 'model', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time']
        with open(csv_path, 'a') as csv_file:
            if print_header:
                csv_file.write(','.join(field_names)+'\n')
            for result in results:
                csv_file.write(','.join([str(result[fn]) for fn in field_names])+'\n')

    def average_results(self, cv_results):
        # average the results of cross validation
        score_dict = {}
        num_metrics = 0

        for cv_result in cv_results:
            for clf_result in cv_result:
                clf_names, accuracy, precision_score, recall_score, f1_score, training_time, test_time= clf_result
                num_metrics = len(clf_result) - 1
                clf_scores = score_dict.get(clf_names, [])
                clf_scores.extend([accuracy, precision_score, recall_score, f1_score, training_time, test_time])
                score_dict[clf_names] = clf_scores

        '''
        Returns a list of dicts like:
            [{'dataset': 'dstc2', 'model': 'Random forest', ... 'test_time': 0.04263}, {...},...]
        Before, it just returns a list of lists containing all the data without labels like
            [['dstc2', 'Random forest', 0.47469487219062767, 0.13496877853565034, 0.25207010891040588, 0.16629532360650451, 1.0303024768829345, 0.042631435394287112], [], ... []]
        '''
        avg_results = []
        field_names = self.config['metrics'] #['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time']
        for clf_name, clf_scores in score_dict.items():
            score_rows = np.asarray(clf_scores).reshape((num_metrics, int(len(clf_scores) / num_metrics)), order='F')
            avg_score = np.average(score_rows, axis=1)
            std_score = np.std(score_rows, axis=1)
            dict_ = {}
            if self.config['experiment_mode'] == 'feature_selection':
                dict_['dataset'] = '%s.#percent=%d.#feature=%d' % (self.config['data_name'], self.config['percentile'], self.config['#features'])
            elif self.config['experiment_mode'] == 'leave_one_out':
                dict_['dataset'] = '%s.leave-out-features=%s.#=%d' % (self.config['data_name'], self.config['f_series_name'], self.config['#features'])
            elif self.config['experiment_mode'] == 'keep_one_only':
                dict_['dataset'] = '%s.feature-to-keep=%s.#=%d' % (self.config['data_name'], self.config['f_series_name'], self.config['#features'])
            else:
                dict_['dataset'] = self.config['data_name']

            dict_['model'] = clf_name
            for k,mean,std in zip(field_names, avg_score, std_score):
                dict_[k] = mean
                dict_[k+'_std'] = std

            avg_results.append(dict_)
        '''
        avg_results = []
        for clf_name, clf_scores in score_dict.items():
            avg_score = np.average(np.asarray(clf_scores).reshape((6, int(len(clf_scores)/6)), order='F'), axis=1)
            r = [self.config.param['data_name'], clf_name]
            r.extend(avg_score)
            avg_results.append(r)
        '''

        field_names = ['dataset', 'model']
        field_names.extend(self.config['metrics'])
        # make some plots
        indices = np.arange(len(avg_results))
        # transpose the result matrix
        t_results = [[result[name] for result in avg_results] for name in field_names]

        dataset_names, clf_names, accuracy, precision_score, recall_score, f1_score, training_time, test_time = t_results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title('Performance of different models on %s' % self.config['data_name'])
        plt.barh(indices, accuracy, .2, label="accuracy", color='navy')
        plt.barh(indices + .2, f1_score, .2, label="f1_score", color='red')
        plt.barh(indices + .4, precision_score, .2, label="precision",
                 color='c')
        plt.barh(indices + .6, recall_score, .2, label="recall", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.35)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c, a, f in zip(indices, clf_names, accuracy, f1_score):
            plt.text(-.2, i, '%s\nacc=%.2f\nf1=%.4f' % (c,a,f))

        # plt.show()

        if self.config['experiment_mode'] == 'feature_selection':
            plt.savefig(os.path.join(self.config.param['experiment_path'], '%s.#percent=%d.#feature=%d.jpg' % (self.config.param['data_name'], self.config['percentile'], self.config['#features'])))
            self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], '%s.#percent=%d.#feature=%d.csv' % (self.config.param['data_name'], self.config['percentile'], self.config['#features'])))
            self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], 'all.test.csv'))
        elif self.config['experiment_mode'] == 'leave_one_out':
            plt.savefig(os.path.join(self.config.param['experiment_path'], '%s.leave-out-features=%s.#=%d.jpg' % (self.config['data_name'], self.config['f_series_name'], self.config['#features'])))
            self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], '%s.leave-out-features=%s.#=%d.csv' % (self.config['data_name'], self.config['f_series_name'], self.config['#features'])))
            self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], 'all.test.csv'))
        elif self.config['experiment_mode'] == 'keep_one_only':
            plt.savefig(os.path.join(self.config.param['experiment_path'], '%s.keeo-one-only=%s.#=%d.jpg' % (self.config['data_name'], self.config['f_series_name'], self.config['#features'])))
            self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], '%s.keeo-one-only=%s.#=%d.csv' % (self.config['data_name'], self.config['f_series_name'], self.config['#features'])))
            self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], 'all.test.csv'))
        else:
            plt.savefig(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.jpg'))
            self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.test.csv'))
            self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], 'all.test.csv'))

        return avg_results

    def classification_report(self, Y, pred, model_name, valid_or_test, classifier=None):
        acc_score = metrics.accuracy_score(Y, pred)
        precision_score = metrics.precision_score(Y, pred, average='macro')
        recall_score = metrics.recall_score(Y, pred, average='macro')
        f1_score = metrics.f1_score(Y, pred, average='macro')

        self.logger.info("accuracy:   %0.3f" % acc_score)
        self.logger.info("f1_score:   %0.3f" % f1_score)

        if opts.print_report:
            self.logger.info("classification report:")
            report = metrics.classification_report(Y, pred,
                                                   target_names=np.asarray(self.config['label_encoder'].classes_))
            self.logger.info(report)

        if opts.print_cm:
            self.logger.info("confusion matrix:")
            confusion_mat = str(metrics.confusion_matrix(Y, pred))
            self.logger.info('\n' + confusion_mat)

        result = {}
        result['dataset'] = self.config.param['data_name']
        result['model'] = model_name
        result['valid_test'] = valid_or_test
        result['test_round'] = self.config.param['test_round']
        result['context_set'] = self.config.param['context_set']

        if 'feature_set' in self.config.param:
            result['feature_set'] = self.config.param['feature_set']
        else:
            result['feature_set'] = ''

        result['accuracy'] = acc_score
        result['precision'] = precision_score
        result['recall'] = recall_score
        result['f1_score'] = f1_score
        # result['training_time'] = train_time
        # result['test_time'] = test_time
        result['report'] = report
        result['confusion_matrix'] = confusion_mat
        result['y_test'] = list([int(y) for y in Y])
        result['y_pred'] = list([int(y) for y in pred])

        if classifier and hasattr(classifier, 'coef_'):
            result['coef'] = classifier.coef_.tolist()

        return result
