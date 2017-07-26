# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
import copy
import pickle
import numpy as np
from optparse import OptionParser
import sys, os
from time import time
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
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

(opts, args) = op.parse_args()


class Experimenter():
    def __init__(self, config):
        self.config = config
        self.logger = self.config.logger

    def load_cv_index(self, X, Y):
        # load the data index for cross-validation
        cv_index_cache_path = os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation',
                                           self.config.param['data_name'] + '.index_cache.evenly.#div=%d.#cv=%d.pkl' % (
                                               self.config['#division'], self.config['#cross_validation']))
        if os.path.exists(cv_index_cache_path):
            with open(cv_index_cache_path, 'rb') as idx_cache:
                train_ids, test_ids = pickle.load(idx_cache)
        else:
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
            self.logger.info(metrics.classification_report(Y_test, pred,
                                                target_names=target_names))

        if opts.print_cm:
            self.logger.info("confusion matrix:")
            self.logger.info('\n'+str(metrics.confusion_matrix(Y_test, pred)))

        clf_descr = str(clf) # str(clf).split('(')[0]
        return model_name, acc_score, precision_score, recall_score, f1_score, train_time, test_time

    def run_cross_validation(self, X, Y):
        X = np.nan_to_num(X.todense())

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

    def run_cross_validation_with_feature_selection(self, X, Y):
        X = np.nan_to_num(X.todense())

        train_ids, test_ids = self.load_cv_index(X, Y)
        cv_results = []
        global X_train, Y_train, X_test, Y_test

        percentiles = (1, 5, 10, 20, 40, 60, 80, 100)

        avg_result_dict = {}

        w2v_feature_indices = [fid for fid, fname in enumerate(self.config['feature_names']) if fname.startswith('8.3')]
        X_w2v           =   copy.deepcopy(X)[:,w2v_feature_indices]

        # iterate the percentile of features to retain
        for percentile in percentiles:
            X_to_select     =   np.delete(copy.deepcopy(X), w2v_feature_indices, axis=1)
            X_new = SelectPercentile(chi2, percentile=percentile).fit_transform(X_to_select, Y)
            X_new = np.concatenate((X_new, X_w2v), axis=1)
            X_new = np.nan_to_num(X_new)

            self.logger.info('%' * 50)
            self.logger.info(' '*10 + 'Percentile=%d' % percentile)
            self.logger.info(' '*10 + 'X_new.shape=%s' % str(X_new.shape))
            self.logger.info('%' * 50)

            self.config['percentile'] = percentile
            self.config['#features'] = X_new.shape[1]

            # iterate folds for cross validation and get the average score
            for r_id, (train_id, test_id) in enumerate(zip(train_ids, test_ids)):
                self.logger.info('*' * 20 + ' %s - Percentile=%d, #(Feature)=%d, Round %d ' % (self.config['data_name'], percentile, X_new.shape[1], r_id) + '*' * 20)
                X_train = np.nan_to_num(preprocessing.scale(X[train_id]))
                Y_train = Y[train_id]
                X_test = np.nan_to_num(preprocessing.scale(X[test_id]))
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
            ax.errorbar(percentiles, [r['f1_score'] for r in model_result], [r['f1_score_std'] for r in model_result])

            ax.set_title('Performance of the %s - feature selected by Chi2' % model_name)
            ax.set_xlabel('Percentile')
            ax.set_ylabel('Prediction rate')

            ax.axis('tight')
            # plt.show()
            fig.savefig(os.path.join(self.config.param['experiment_path'], '%s.model=%s.feature_selection.plot.jpg' % (self.config.param['data_name'], model_name)))


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
        for clf, name in [
                # (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                # (Perceptron(n_iter=50), "Perceptron"),
                # (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                # (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(n_estimators=100), "Random forest")]:
            self.logger.info('=' * 80)
            self.logger.info(name)
            results.append(self.benchmark(name, clf))

        '''
        for penalty in ["l2", "l1"]:
            self.logger.info('=' * 80)
            self.logger.info("%s penalty" % penalty.upper())

            # Train Liblinear model
            results.append(self.benchmark('LinearSVC.loss=l2.penalty=%s' % penalty, LinearSVC(loss='l2', penalty=penalty,
                                               dual=False, tol=1e-3)))

            # Train SGD model
            # results.append(self.benchmark('SGDClassifier.penalty=%s' % penalty, SGDClassifier(alpha=.0001, n_iter=50,
            #                                        penalty=penalty)))
        '''
        # Train SGD with Elastic Net penalty
        # self.logger.info('=' * 80)
        # self.logger.info("Elastic-Net penalty")
        # results.append(self.benchmark('SGDClassifier.penalty=%s' % 'elasticnet', SGDClassifier(alpha=.0001, n_iter=50,
        #                                        penalty="elasticnet")))

        # Train NearestCentroid without threshold
        # self.logger.info('=' * 80)
        # self.logger.info("NearestCentroid (aka Rocchio classifier)")
        # results.append(self.benchmark('NearestCentroid', NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        # self.logger.info('=' * 80)
        # self.logger.info("Naive Bayes")
        # results.append(self.benchmark('MultinomialNB', MultinomialNB(alpha=.01)))
        # results.append(self.benchmark('BernoulliNB', BernoulliNB(alpha=.01)))

        '''
        self.logger.info('=' * 80)
        self.logger.info("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark('LinearSVC+L1-FeatSel', Pipeline([
            ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
            ('classification', LinearSVC())
        ])))
        '''

        for C in [1]:
        # for C in [0.1, 1, 10]:
            self.logger.info('=' * 80)
            self.logger.info("LR.pen=l1.C=%f" % C)
            # Train Logistic Regression model
            results.append(self.benchmark('LR.pen=l1.C=%d' % C,
                                          LogisticRegression(solver="liblinear", penalty='l1', C=C)))

            self.logger.info('=' * 80)
            self.logger.info("LinearSVC.pen=l1, C=%d" % C)
            results.append(self.benchmark('LinearSVC.pen=l1.C=%d' % C,
                                          LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3)))
            '''
            self.logger.info('=' * 80)
            self.logger.info("Logistic Regression with penalty=l2, C=%f" % C)
            # Train Logistic Regression model
            results.append(self.benchmark('Logistic Regression Classifier.penalty=l2.C=%s' % C, LogisticRegression(solver="liblinear", penalty='l2', C=C)))

            self.logger.info('=' * 80)
            self.logger.info("RBF SVC with C=%f" % C)
            results.append(self.benchmark('RBF SVC with C=%f' % C, SVC(C=C, cache_size=200, class_weight=None,
                                          degree=3, gamma='auto', kernel='rbf',
                                          max_iter=-1, probability=False, random_state=None, shrinking=True,
                                          tol=0.001, verbose=False))
            )
            '''

        return results

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