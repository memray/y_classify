# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

import leven

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        print('#%s = %d' % (self.key, len(data_dict[self.key])))
        return data_dict[self.key]

class RawFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract the utterance text & context from a dialogue in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, X_raw):
        global logger
        features = np.recarray(shape=(len(X_raw),),
                               dtype=[('current_user_utterance', object),
                                      ('next_system_utterance', object),
                                      ('next_user_utterance', object),
                                      ('user_utterance_pairs', object)])

        next_system_utterance_count = 0
        next_user_utterance_count = 0
        for i, x_raw in enumerate(X_raw):
            features['current_user_utterance'][i] = x_raw['utterance'].msg_text

            x_index = x_raw['index']
            if x_index + 1 < len(x_raw['dialogue']) and x_raw['dialogue'][x_index+1].direction == 'bot_to_sb':
                features['next_system_utterance'][i] = x_raw['dialogue'][x_index+1].msg_text
                if features['next_system_utterance'][i] != '':
                    next_system_utterance_count += 1
                    # self.logger.info('%s - %s' % ('next_system_utterance', features['next_system_utterance'][i]))
            else:
                features['next_system_utterance'][i] = ''

            if x_index + 2 < len(x_raw['dialogue']) and x_raw['dialogue'][x_index+2].direction == 'user_to_sb':
                features['next_user_utterance'][i] = x_raw['dialogue'][x_index+2].msg_text
                if features['next_user_utterance'][i] != '':
                    next_user_utterance_count += 1
                    # self.logger.info('%s - %s' % ('next_user_utterance', features['next_user_utterance'][i]))
                features['user_utterance_pairs'][i] = (x_raw['utterance'].msg_text, x_raw['dialogue'][x_index+2].msg_text)
            else:
                features['next_user_utterance'][i] = ''
                features['user_utterance_pairs'][i] = (x_raw['utterance'].msg_text, '')

        logger.info('has_next_system_utterance_count = %d/%d' % (next_system_utterance_count, len(X_raw)))
        logger.info('has_next_user_utterance_count = %d/%d' % (next_user_utterance_count, len(X_raw)))

        return features

class EditDistance(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self

    def transform(self, pairs):
        return [{'edit_distance': leven.levenshtein(pairs[0].lower(), pairs[1].lower())} for pairs in pairs]


class Feature_Extracter():
    def __init__(self, config):
        self.config = config
        self.logger = self.config.logger
        global logger
        logger = self.logger

    def extract(self):
        '''
        Extract features from utterances
        '''
        '''
        Define feature extractor
        '''
        '''
        Define transformer_list
        '''
        transformer_list = []
        # extract features from each utterance in range
        for utterance in self.config['utterance_range']:
            # 1. Basic Feature
            # 2. User Action Feature
            # 3. Time Interval Feature
            # 4. Lexical Feature
            #   4.1 n_gram
            # Perform an IDF normalization on the output of HashingVectorizer
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                               ngram_range=(1,3), lowercase=True,
                                               use_idf=True, smooth_idf = True,
                                               max_features = None,
                                               stop_words='english')
            n_gram_pipe = Pipeline([
                ('selector', ItemSelector(key=utterance)),
                ('tfidf', tfidf_vectorizer)
            ])
            transformer_list.append(
                ('4.1-ngram.%s' % utterance, n_gram_pipe)
            )
            # 5. Phrasal Feature
            # 6. Syntactic Feature
            # 7. Semantic/Topic Feature

        #   4.2 edit_distance
        edit_distance_vectorizer = DictVectorizer()
        edit_distance_pipe = Pipeline([
            ('selector', ItemSelector(key='user_utterance_pairs')),
            ('edit_distance', EditDistance()),
            ('vectorize', edit_distance_vectorizer),
        ])
        transformer_list.append(
            ('4.2-edit_distance', edit_distance_pipe)
        )
        #   4.3 jaccard_similarity



        union_features = FeatureUnion(transformer_list=transformer_list)

        pipeline = Pipeline([
            ('raw_feature_extractor', RawFeatureExtractor()),
            # Use FeatureUnion to combine the features
            ('union', union_features)
        ])

        X = pipeline.fit_transform(self.config['X_raw'])
        for transformer in transformer_list:
            if transformer[1].__class__.__name__ == 'Pipeline':
                transformer[1].vocabulary_ = transformer[1].steps[-1][1].vocabulary_

        self.config['X']               = X
        self.config['union_features']  = union_features
        union_features.get_feature_names()

        return X

    def split_to_instances(self, annotated_sessions):
        '''
        for now just return the text of corresponding utterances (ignore any context information)
        :param annotated_sessions:
        :return:
        X_raw = [], a list of dicts, contains all the usable information of current utterance (utterance, index, and dialogue)
        Y_raw = [], true labels, after mapping to [0,..., n-1] by LabelEncoder
        '''
        X_raw = []
        Y_raw = []

        valid_type = self.config.param['valid_type']

        for session in annotated_sessions:
            for u_idx, u_ in enumerate(session):
                if u_.direction != 'user_to_sb':
                    continue
                if u_.type == None or u_.type == '' or u_.type not in valid_type:
                    continue

                raw_features = {}
                raw_features['index'] = u_idx # position of current utterance
                raw_features['utterance'] = u_ # current utterance
                raw_features['dialogue'] = session # context
                X_raw.append(raw_features)
                Y_raw.append(u_.type)

        le = preprocessing.LabelEncoder()
        le.fit(Y_raw)
        # list(le.classes_)
        Y = le.transform(Y_raw)

        self.config['label_encoder']   = le
        self.config['X_raw']           = X_raw
        self.config['Y']               = Y
        return X_raw, Y