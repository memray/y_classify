# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
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
from sklearn import metrics

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
        # load the data index for cross-validation
        cv_index_cache_path = os.path.join(self.config.param['root_path'], 'dataset', 'cross_validation', self.config.param['data_name'] + '.index_cache.#cv=%d.pkl' % self.config['#cross_validation'])
        if os.path.exists(cv_index_cache_path):
            with open(cv_index_cache_path, 'rb') as idx_cache:
                train_idx, test_idx = pickle.load(idx_cache)
        else:
            with open(cv_index_cache_path, 'wb') as idx_cache:
                num_data = len(Y)
                train_idx = []
                test_idx = []
                for i in range(self.config['#cross_validation']):
                    train_ = np.random.choice(num_data, int(0.9 * num_data))
                    train_set = set(train_)
                    test_  = np.asarray([i for i in range(num_data) if i not in train_set])
                    train_idx.append(train_)
                    test_idx.append(test_)

                pickle.dump([train_idx, test_idx], idx_cache)

        cv_results = []
        global X_train, Y_train, X_test, Y_test
        for r_id, (train_id, test_id) in enumerate(zip(train_idx, test_idx)):
            self.logger.info('*' * 20 + ' %s - Round %d ' % (self.config['data_name'], r_id))
            X_train = X[train_id]
            Y_train = Y[train_id]
            X_test  = X[test_id]
            Y_test  = Y[test_id]
            cv_results.append(self.run_experiment())

        # get the average score of cross-validation
        avg_results = self.average_results(cv_results)

        return avg_results


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
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                (Perceptron(n_iter=50), "Perceptron"),
                (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(n_estimators=100), "Random forest")):
            self.logger.info('=' * 80)
            self.logger.info(name)
            results.append(self.benchmark(name, clf))

        for penalty in ["l2", "l1"]:
            self.logger.info('=' * 80)
            self.logger.info("%s penalty" % penalty.upper())

            # Train Liblinear model
            results.append(self.benchmark('LinearSVC.loss=l2.penalty=%s' % penalty, LinearSVC(loss='l2', penalty=penalty,
                                               dual=False, tol=1e-3)))

            # Train SGD model
            # results.append(self.benchmark('SGDClassifier.penalty=%s' % penalty, SGDClassifier(alpha=.0001, n_iter=50,
            #                                        penalty=penalty)))

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

        self.logger.info('=' * 80)
        self.logger.info("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark('LinearSVC+L1-FeatSel', Pipeline([
            ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
            ('classification', LinearSVC())
        ])))

        for C in [0.1, 1, 10]:
            self.logger.info('=' * 80)
            self.logger.info("Logistic Regression with penalty=l1, C=%f" % C)
            # Train Logistic Regression model
            results.append(self.benchmark('Logistic Regression Classifier.penalty=l1.C=%s' % C, LogisticRegression(solver="liblinear", penalty='l1', C=C)))

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

        return results

    @staticmethod
    def export_summary(results, csv_path):
        if not os.path.exists(csv_path):
            print_header = True
        else:
            print_header = False

        with open(csv_path, 'a') as csv_file:
            if print_header:
                csv_file.write(','.join(['dataset', 'model', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'test_time'])+'\n')
            for result in results:
                csv_file.write(','.join([str(r) for r in result])+'\n')

    def average_results(self, cv_results):
        # average the results of cross validation
        score_dict = {}
        for cv_result in cv_results:
            for clf_result in cv_result:
                clf_names, accuracy, precision_score, recall_score, f1_score, training_time, test_time = clf_result
                clf_scores = score_dict.get(clf_names, [])
                clf_scores.extend(clf_result[1:7])
                score_dict[clf_names] = clf_scores

        avg_results = []
        for clf_name, clf_scores in score_dict.items():
            avg_score = np.average(np.asarray(clf_scores).reshape((6, int(len(clf_scores)/6)), order='F'), axis=1)
            r = [self.config.param['data_name'], clf_name]
            r.extend(avg_score)
            avg_results.append(r)

        # make some plots
        indices = np.arange(len(avg_results))
        # transpose the result matrix
        t_results = [[x[i] for x in avg_results] for i in range(len(avg_results[0]))]

        dataset_names, clf_names, accuracy, precision_score, recall_score, f1_score, training_time, test_time = t_results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, accuracy, .2, label="accuracy", color='navy')
        plt.barh(indices + .2, f1_score, .2, label="f1_score", color='red')
        plt.barh(indices + .4, precision_score, .2, label="precision",
                 color='c')
        plt.barh(indices + .6, recall_score, .2, label="recall", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        # plt.show()
        plt.savefig(os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.jpg'))
        self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], self.config.param['data_name']+'.test.csv'))
        self.export_summary(avg_results, os.path.join(self.config.param['experiment_path'], 'all.test.csv'))

        return avg_results