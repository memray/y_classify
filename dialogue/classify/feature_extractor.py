# -*- coding: utf-8 -*-
import os

import nltk
import numpy as np
import torch
from gensim import corpora
from gensim.models import Doc2Vec
from gensim.models import LdaModel
from gensim.models.doc2vec import TaggedDocument
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity

from dialogue.skipthought.skipthoughts import BiSkip

import leven
from nltk.stem.porter import *
from nltk.corpus import stopwords

from dialogue.data import data_loader
import numpy.core.numeric as _nx
from numpy.core import getlimits, umath

# initialize the Stanford wrapper
from dialogue.stanford_corenlp.pycorenlp.corenlp import StanfordCoreNLP
import gensim
from sklearn.externals import joblib

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

stemmer = PorterStemmer()

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
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        for key in self.keys:
            print('#%s = %d' % (key, len(data_dict[key])))

        if len(self.keys) == 1:
            return data_dict[self.keys[0]]
        else:
            return [data_dict[key] for key in self.keys]

def extract_noun_phrases(source_text, source_postag, max_len=3):
    np_regex = r'^(JJ|JJR|JJS|VBG|VBN)*(NN|NNS|NNP|NNPS|VBG)+$'
    np_list = []

    for i in range(len(source_text)):
        for j in range(i+1, len(source_text)+1):
            if j-i > max_len:
                continue
            tagseq = ''.join(source_postag[i:j])
            if re.match(np_regex, tagseq):
                np_list.append((source_text[i:j], source_postag[i:j]))

    # print('Text: \t\t %s' % str(source_text))
    # print('PosTag: \t\t %s' % str(source_postag))
    # print('None Phrases:[%d] \n\t\t\t%s' % (len(np_list), str('\n\t\t\t'.join([str(p[0])+'['+str(p[1])+']' for p in np_list]))))

    return np_list

class RawFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract the utterance text & context from a dialogue in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.

    Each x_raw contains:
        x_raw['index'] = u_idx # position of current utterance, start from 0
        x_raw['utterance'] = u_ # current utterance
        x_raw['dialogue'] = session # context
    """
    def __init__(self, config):
        self.config = config

    def fit(self, x, y=None):
        return self

    def transform(self, X_raw):
        global logger
        global config

        feature_cache_path = config['raw_feature_path'] % config['data_name']

        if os.path.exists(feature_cache_path):
            logger.info('Loading saved raw feature of %s from %s' % (config['data_name'], feature_cache_path))
            features = data_loader.deserialize_from_file(feature_cache_path)
        else:
            logger.info('Not found saved raw feature of %s, extracting...' % (config['data_name']))

            features = np.recarray(shape=(len(X_raw),),
                                   dtype=[('current_user_utterance', object),
                                          ('next_system_utterance', object),
                                          ('next_user_utterance', object),
                                          ('next_user_utterance_pairs', object),
                                          ('last_system_utterance', object),
                                          ('last_user_utterance', object),
                                          ('last_user_utterance_pairs', object),
                                          ('parsed_results__current_user_utterance', object),
                                          ('parsed_results__next_system_utterance', object),
                                          ('parsed_results__next_user_utterance', object),
                                          ('parsed_results__last_system_utterance', object),
                                          ('parsed_results__last_user_utterance', object),
                                          ('noun_phrases__current_user_utterance', object),
                                          ('noun_phrases__next_system_utterance', object),
                                          ('noun_phrases__next_user_utterance', object),
                                          ('noun_phrases__last_system_utterance', object),
                                          ('noun_phrases__last_user_utterance', object),
                                          ('x_raw', object)])

            next_system_utterance_count = 0
            next_user_utterance_count = 0

            # corenlp = CoreNLP('nerparse', corenlp_jars=config['corenlp_jars'], comm_mode='PIPE')

            nlp = StanfordCoreNLP('http://localhost:9000')

            logger.info('Generating raw feature of %s' % (config['data_name']))
            for i, x_raw in enumerate(X_raw):
                if i % 1000 == 0:
                    print(i)
                '''
                1. Raw features of current_user_utterance
                '''
                features['current_user_utterance'][i] = x_raw['utterance'].msg_text
                features['x_raw'][i] = x_raw

                '''
                2. Raw features of next_system_utterance
                '''
                x_index = x_raw['index']
                if x_index + 1 < len(x_raw['dialogue']) and x_raw['dialogue'][x_index+1].direction == 'bot_to_sb':
                    features['next_system_utterance'][i] = x_raw['dialogue'][x_index+1].msg_text
                    if features['next_system_utterance'][i] != '':
                        next_system_utterance_count += 1
                        # self.logger.info('%s - %s' % ('next_system_utterance', features['next_system_utterance'][i]))
                else:
                    features['next_system_utterance'][i] = ''

                '''
                3. Raw features of next_user_utterance, and pairs of (current_user_utterance, next_user_utterance)
                '''
                if x_index + 2 < len(x_raw['dialogue']) and x_raw['dialogue'][x_index+2].direction == 'user_to_sb':
                    features['next_user_utterance'][i] = x_raw['dialogue'][x_index+2].msg_text
                    if features['next_user_utterance'][i] != '':
                        next_user_utterance_count += 1
                        # self.logger.info('%s - %s' % ('next_user_utterance', features['next_user_utterance'][i]))
                    features['next_user_utterance_pairs'][i] = (x_raw['utterance'].msg_text, x_raw['dialogue'][x_index+2].msg_text)
                else:
                    features['next_user_utterance'][i] = ''
                    features['next_user_utterance_pairs'][i] = (x_raw['utterance'].msg_text, '')

                '''
                4. Raw features of last_system_utterance
                '''
                x_index = x_raw['index']
                if x_index - 1 >= 0 and x_raw['dialogue'][x_index-1].direction == 'bot_to_sb':
                    features['last_system_utterance'][i] = x_raw['dialogue'][x_index-1].msg_text
                    if features['last_system_utterance'][i] != '':
                        next_system_utterance_count += 1
                        # self.logger.info('%s - %s' % ('next_system_utterance', features['next_system_utterance'][i]))
                else:
                    features['last_system_utterance'][i] = ''

                '''
                5. Raw features of last_user_utterance
                '''
                if x_index - 2 >= 0 and x_raw['dialogue'][x_index-2].direction == 'user_to_sb':
                    features['last_user_utterance'][i] = x_raw['dialogue'][x_index-2].msg_text
                    if features['last_user_utterance'][i] != '':
                        next_user_utterance_count += 1
                    features['last_user_utterance_pairs'][i] = (x_raw['utterance'].msg_text, x_raw['dialogue'][x_index-2].msg_text)
                else:
                    features['last_user_utterance'][i] = ''
                    features['last_user_utterance_pairs'][i] = (x_raw['utterance'].msg_text, '')

            logger.info('Extracting basic features by CoreNLP')
            # Extract basic features by CoreNLP
            for i, x_raw in enumerate(X_raw):
                print("*" * 50)
                print(i)
                #
                # if i < 1108:
                #     continue

                for utterance_type in config['utterance_range']:
                    print(features[utterance_type][i])
                    '''
                    # Changed from CoreNLP to PyCoreNLP
                    consists of ['deps_cc', 'pos', 'lemmas', 'tokens', 'char_offsets', 'ner', 'entitymentions', 'parse', 'deps_basic', 'normner']
                    parsed_results = corenlp.parse_doc(features[utterance_type][i])
                    '''

                    parsed_results = nlp.annotate(features[utterance_type][i], properties={
                        'annotators': 'tokenize, ssplit, pos, lemma, ner, entitymentions, parse',
                        'outputFormat': 'json', 'timeout': 60000
                    })

                    # if len(parsed_results['sentences']) > 1:
                    #     print('Find %d sentences' % len(parsed_results['sentences']))
                    #     print(features[utterance_type][i])

                    if len(parsed_results['sentences']) == 0:
                        features['parsed_results__%s' % utterance_type][i]  = []
                        features['noun_phrases__%s' % utterance_type][i]    = []
                    else:
                        # print(parsed_results['sentences'][0])
                        # print('parsed_results__%s' % utterance_type)
                        # print(features['parsed_results__%s' % utterance_type])

                        # should be a list of dicts
                        features['parsed_results__%s' % utterance_type][i] = parsed_results['sentences']

                        '''
                        # Changed from CoreNLP to PyCoreNLP
                        tokens = np.concatenate([[t.lower() for t in s['tokens']] for s in parsed_results['sentences']])
                        pos    = np.concatenate([s['pos'] for s in parsed_results['sentences']])
                        '''
                        # could be multiple sentences in one utterance, so merge them into one array
                        words = np.concatenate(
                            [[t['originalText'].lower() for t in s['tokens']] for s in parsed_results['sentences']])
                        pos = np.concatenate([[t['pos'] for t in s['tokens']] for s in parsed_results['sentences']])

                        features['noun_phrases__%s' % utterance_type][i] = extract_noun_phrases(words, pos)

            # logger.info('has_next_system_utterance_count = %d/%d' % (next_system_utterance_count, len(X_raw)))
            # logger.info('has_next_user_utterance_count = %d/%d' % (next_user_utterance_count, len(X_raw)))

            logger.info('Saving raw feature of %s to %s' % (config['data_name'], feature_cache_path))
            data_loader.serialize_to_file(features, feature_cache_path)

        config.param['raw_feature'] = features

        return features

class UtteranceLength(BaseEstimator, TransformerMixin):
    '''
    1. Basic Feature
    '''
    def fit(self, x, y=None):
        return self

    def transform(self, utts):
        return [{'utterance_length': len(utt.split())} for utt in utts]

class UserActionWords(BaseEstimator, TransformerMixin):
    '''
    2.1 user action words
    '''
    def __init__(self, user_action):
        self.user_action = user_action

    def fit(self, x, y=None):
        return self

    def transform(self, utts):
        action_words_all = []
        for utt in utts:
            action_words = []
            for w in utt.lower().split():
                stemmed_w = stemmer.stem(w)
                if stemmed_w in self.user_action:
                    action_words.append(stemmed_w)
            action_words_all.append(action_words)
        # return [{'utterance_length': action_words} for action_words in action_words_all]
        return [dict(zip([w for w in action_words], [True]*len(action_words))) for action_words in action_words_all]

class NumberOfUserActionWords(BaseEstimator, TransformerMixin):
    '''
    2.2 number of user action words
    '''
    def __init__(self, user_action):
        self.user_action = user_action

    def fit(self, x, y=None):
        return self

    def transform(self, utts):
        action_words_all = []
        for utt in utts:
            action_words = []
            for w in utt.lower().split():
                stemmed_w = stemmer.stem(w)
                if stemmed_w in self.user_action:
                    action_words.append(stemmed_w)
            action_words_all.append(action_words)
        return [{'utterance_length': len(action_words)} for action_words in action_words_all]

def jaccard_similarity(set1, set2):
    intersec_ = set.intersection(set1, set2)
    union_ = set.union(set1, set2)
    if len(union_) > 0:
        return float(len(intersec_))/ float(len(union_))
    else:
        return 0.0

class UserActionJaccardSimilarity(BaseEstimator, TransformerMixin):
    '''
    2.3 jaccard_similarity of user action words, between the current user's utterances and the next
    in union_features, the name of this feature is gonna be "2.3-action_jaccard_similarity__jaccard_similarity"
    '''
    def __init__(self, user_action):
        self.user_action = user_action

    def fit(self, x, y=None):
        return self

    def transform(self, utt_pairs):
        action_pairs = []
        for utt_pair in utt_pairs:
            utt1_action = []
            utt2_action = []
            for w in utt_pair[0].lower().split():
                stemmed_w = stemmer.stem(w)
                if stemmed_w in self.user_action:
                    utt1_action.append(stemmed_w)

            for w in utt_pair[1].lower().split():
                stemmed_w = stemmer.stem(w)
                if stemmed_w in self.user_action:
                    utt2_action.append(stemmed_w)

            action_pairs.append((utt1_action, utt2_action))
        return [{'jaccard_similarity': jaccard_similarity(set(pair[0]), set(pair[1]))} for pair in action_pairs]


class TimeFeature(BaseEstimator, TransformerMixin):
    '''
    3 - time features
        x_raw['index'] = u_idx # position of current utterance, starting from 0
        x_raw['utterance'] = u_ # current utterance
        x_raw['dialogue'] = session # context
    '''
    def __init__(self):
        print('3 - time features')

    def fit(self, x, y=None):
        return self

    def transform(self, x_raws):
        f_list = []
        for x_raw in x_raws:
            f_dict = {}
            if x_raw['index'] == 0:
                f_dict['start_of_session'] = True
            else:
                f_dict['start_of_session'] = False

            if (x_raw['dialogue'][-1].direction == 'bot_to_sb' and x_raw['index'] == len(x_raw['dialogue']) - 2) or (x_raw['dialogue'][-1].direction == 'user_to_sb' and x_raw['index'] == len(x_raw['dialogue']) - 1):
                f_dict['end_of_session'] = True
            else:
                f_dict['end_of_session'] = False

            f_dict['turns_from_start']  = x_raw['index']
            f_dict['turns_to_end']      = len(x_raw['dialogue']) - x_raw['index']
            f_dict['time_ratio']        = float(x_raw['index']) / float(len(x_raw['dialogue']))

            f_list.append(f_dict)

        return f_list

class EditDistance(BaseEstimator, TransformerMixin):
    '''
    4.2 edit_distance, between the current and next user's utterances
    '''
    def fit(self, x, y=None):
        return self

    def transform(self, pairs):
        return [{'edit_distance': leven.levenshtein(pairs[0].lower(), pairs[1].lower())} for pairs in pairs]

class JaccardSimilarity(BaseEstimator, TransformerMixin):
    '''
    4.3 jaccard_similarity, between the current and next user's utterances
    '''
    def fit(self, x, y=None):
        return self

    def transform(self, pairs):
        return [{'jaccard_similarity': jaccard_similarity(set(pair[0].lower().split()), set(pair[1].lower().split()))} for pair in pairs]

class PhraseFeature(BaseEstimator, TransformerMixin):
    '''
    5. phrase_features
    '''
    def __init__(self, config):
        print('5. phrase_features')
        self.config = config

    def fit(self, x, y=None):
        return self

    def transform(self, np_records):
        return_list = []

        # iterate each data sample, contains three parts (currrent_user, current_system, next_user)
        for k, np_record in enumerate(zip(*np_records)):
            # iterate each type of utterance
            np_dict = {}

            # 5.1 noun_phrase: one-hot representation of all noun phrases (extracted by POS-tagging patterns)
            for i in range(len(self.config['utterance_range'])):
                # iterate each phrase in the sentence, np[0] is words, np[1] is postags
                for np_ in np_record[i]:
                    np_dict['noun_phrases__%s__%s' % (self.config['utterance_range'][i], '_'.join(np_[0]))] = True
                    
            last_index_in_dict  = self.config['utterance_range'].index("last_user_utterance")
            current_index_in_dict  = self.config['utterance_range'].index("current_user_utterance")
            next_index_in_dict  = self.config['utterance_range'].index("next_user_utterance")

            last_np_set  = set(['_'.join(np_[0]) for np_ in np_record[last_index_in_dict]])
            current_np_set  = set(['_'.join(np_[0]) for np_ in np_record[current_index_in_dict]])
            next_np_set     = set(['_'.join(np_[0]) for np_ in np_record[next_index_in_dict]])

            last_inter_np        = set.intersection(current_np_set, last_np_set)
            next_inter_np        = set.intersection(current_np_set, next_np_set)

            # 5.2 noun_phrase_overlap:  True, if there is any NP overlap between two user utterances
            if len(last_inter_np) > 0:
                np_dict['last_noun_phrases__have_overlap']    = True
                np_dict['last_noun_phrases__#overlap']        = len(last_inter_np)
            else:
                np_dict['last_noun_phrases__have_overlap']    = False
                np_dict['last_noun_phrases__#overlap']        = 0

            if len(next_inter_np) > 0:
                np_dict['next_noun_phrases__have_overlap']    = True
                np_dict['next_noun_phrases__#overlap']        = len(next_inter_np)
            else:
                np_dict['next_noun_phrases__have_overlap']    = False
                np_dict['next_noun_phrases__#overlap']        = 0

            # 5.3 np_jaccard_similarity: Jaccard similarity of NPs.
            np_dict['last_noun_phrases__jaccard_similarity']  = jaccard_similarity(current_np_set, last_np_set)
            np_dict['next_noun_phrases__jaccard_similarity']  = jaccard_similarity(current_np_set, next_np_set)

            return_list.append(np_dict)

        return return_list

class EntityFeature(BaseEstimator, TransformerMixin):
    '''
    6. entity_features
    '''
    def __init__(self, config):
        print('6. entity_features')
        self.config = config

    def fit(self, x, y=None):
        return self

    def transform(self, parse_records):
        return_list = []

        # iterate each data sample, contains five parts (last_user, last_system, currrent_user, current_system, next_user)
        for k, parse_record in enumerate(zip(*parse_records)):
            # iterate each type of utterance
            entity_dict = {}

            entity_lists = []
            # 6.1 entity: one-hot representation of all entities
            for i in range(len(self.config['utterance_range'])):
                entity_list = []
                cache = []
                # iterate each word and NER tag
                if len(parse_record[i]) > 0:
                    for ner in np.concatenate([r['entitymentions'] for r in parse_record[i]]):
                        entity_dict[
                            'entity__%s__%s_%s' % (self.config['utterance_range'][i], ner['text'].lower(), ner['ner'])] = True

                    '''
                    # Changed to pycorenlp
                    for token, ner in zip(np.concatenate([r['tokens'] for r in parse_record[i]]), np.concatenate([r['ner'] for r in parse_record[i]])):
                        if len(cache) > 0 and ner != cache[-1][1]:
                            entity_name = '_'.join([c[0] for c in cache])
                            entity_type = cache[0][1]
                            entity_list.append((entity_name, entity_type))
                            cache = []
                            entity_dict['entity__%s__%s_%s' % (self.config['utterance_range'][i], entity_name, entity_type)] = True
                        if ner == 'O':
                            continue
                        else:
                            cache.append((token, ner))
                    '''
                entity_lists.append(entity_list)

            last_index_in_dict  = self.config['utterance_range'].index("last_user_utterance")
            current_index_in_dict  = self.config['utterance_range'].index("current_user_utterance")
            next_index_in_dict  = self.config['utterance_range'].index("next_user_utterance")
            last_entity_set  = set([e for e in entity_lists[last_index_in_dict]])
            current_entity_set  = set([e for e in entity_lists[current_index_in_dict]])
            next_entity_set     = set([e for e in entity_lists[next_index_in_dict]])

            last_inter_entity        = set.intersection(current_entity_set, last_entity_set)
            next_inter_entity        = set.intersection(current_entity_set, next_entity_set)

            # 6.2 entity_overlap:  True, if there is any entity overlap between two user utterances
            if len(last_inter_entity) > 0:
                entity_dict['last_entity__have_overlap']    = True
                entity_dict['last_entity__#overlap']        = len(last_inter_entity)
            else:
                entity_dict['last_entity__have_overlap']    = False
                entity_dict['last_entity__#overlap']        = 0

            if len(next_inter_entity) > 0:
                entity_dict['next_entity__have_overlap']    = True
                entity_dict['next_entity__#overlap']        = len(next_inter_entity)
            else:
                entity_dict['next_entity__have_overlap']    = False
                entity_dict['next_entity__#overlap']        = 0

            # 6.3 entity_jaccard_similarity: Jaccard similarity of entities.
            entity_dict['last_entity__jaccard_similarity']  = jaccard_similarity(current_entity_set, last_entity_set)
            entity_dict['next_entity__jaccard_similarity'] = jaccard_similarity(current_entity_set, next_entity_set)

            return_list.append(entity_dict)
        return return_list

class SyntacticFeature(BaseEstimator, TransformerMixin):
    '''
    7. syntactic_features
    '''
    def __init__(self, config):
        print('7. syntactic_features')
        self.config = config

    def fit(self, x, y=None):
        return self

    def transform(self, parse_records):
        return_list = []

        # iterate each data sample, contains three parts (currrent_user, current_system, next_user)
        for k, parse_record in enumerate(zip(*parse_records)):
            # iterate each type of utterance
            dependency_dict = {}

            # 7.1-7.3 key syntactic components
            last_root       = set()
            current_root    = set()
            next_root       = set()

            last_subj       = set()
            current_subj    = set()
            next_subj       = set()

            last_obj        = set()
            current_obj     = set()
            next_obj        = set()
            
            last_index_in_dict  = self.config['utterance_range'].index("last_user_utterance")
            current_index_in_dict  = self.config['utterance_range'].index("current_user_utterance")
            next_index_in_dict  = self.config['utterance_range'].index("next_user_utterance")

            for i in range(len(self.config['utterance_range'])):
                if len(parse_record[i]) > 0:
                    # iterate each parsed sentence
                    for sent in parse_record[i]:
                        # 7.1 root_word: the word that at the root of parse tree (shot).
                        # for dep in sent['deps_basic']:
                        for dep in sent['basicDependencies']:
                            if dep['dep'].lower() == 'root':
                                dep_word = stemmer.stem(dep['dependentGloss'].lower())
                                dependency_dict['root_word__%s__%s' % (self.config['utterance_range'][i], dep_word)] = True
                                root_index = dep['dependent']
                                if i == current_index_in_dict:
                                    current_root.add(dep_word)
                                elif i == next_index_in_dict:
                                    next_root.add(dep_word)
                                elif i == last_index_in_dict:
                                    last_root.add(dep_word)
                                break

                        # 7.2 subj_word: the topmost subjects
                        for dep in sent['basicDependencies']:
                            if dep['dep'].lower().endswith('subj') and dep['governor'] == root_index:
                                dep_word = stemmer.stem(dep['dependentGloss'].lower())
                                dependency_dict['subj_word__%s__%s' % (self.config['utterance_range'][i], dep_word)] = True
                                if i == current_index_in_dict:
                                    current_subj.add(dep_word)
                                elif i == next_index_in_dict:
                                    next_subj.add(dep_word)
                                elif i == last_index_in_dict:
                                    last_subj.add(dep_word)

                        # 7.3 obj_word: the topmost object
                        for dep in sent['basicDependencies']:
                            if dep['dep'].lower().endswith('obj') and dep['governor'] == root_index:
                                dep_word = stemmer.stem(dep['dependentGloss'].lower())
                                dependency_dict['obj_word__%s__%s' % (self.config['utterance_range'][i], dep_word)] = True
                                if i == current_index_in_dict:
                                    current_obj.add(dep_word)
                                elif i == next_index_in_dict:
                                    next_obj.add(dep_word)
                                elif i == last_index_in_dict:
                                    last_obj.add(dep_word)

            # 7.4-7.6 root_words_overlap: True if any of the root words of two user utterances are same.
            if len(set.intersection(current_root, next_root)) > 0:
                dependency_dict['next_root_words_overlap'] = True
            else:
                dependency_dict['next_root_words_overlap'] = False

            if len(set.intersection(current_subj, next_subj)) > 0:
                dependency_dict['next_subj_words_overlap'] = True
            else:
                dependency_dict['next_subj_words_overlap'] = False

            if len(set.intersection(current_obj, next_obj)) > 0:
                dependency_dict['next_obj_words_overlap'] = True
            else:
                dependency_dict['next_obj_words_overlap'] = False

            if len(set.intersection(current_root, last_root)) > 0:
                dependency_dict['last_root_words_overlap'] = True
            else:
                dependency_dict['last_root_words_overlap'] = False

            if len(set.intersection(current_subj, last_subj)) > 0:
                dependency_dict['last_subj_words_overlap'] = True
            else:
                dependency_dict['last_subj_words_overlap'] = False

            if len(set.intersection(current_obj, last_obj)) > 0:
                dependency_dict['last_obj_words_overlap'] = True
            else:
                dependency_dict['last_obj_words_overlap'] = False

            return_list.append(dependency_dict)
        return return_list

class LDAFeature(BaseEstimator, TransformerMixin):
    '''
    8.1 LDA_features
    '''
    def load_dict_corpus(self, dict_path, corpus_path):
        if not os.path.exists(dict_path) or not os.path.exists(corpus_path):

            all_sessions = self.config['data_loader']()
            documents = []
            for session in all_sessions:
                for utt in session:
                    documents.append(utt.msg_text)
            # remove common words and tokenize
            stoplist = set(stopwords.words('english'))
            texts = [[word for word in document.lower().split() if word not in stoplist]
                     for document in documents]
            # remove words that appear only once
            from collections import defaultdict
            frequency = defaultdict(int)
            for text in texts:
                for token in text:
                    frequency[token] += 1
            texts = [[token for token in text if frequency[token] > 1]
                     for text in texts]

            dictionary = corpora.Dictionary(texts)
            dictionary.save(dict_path)
            corpus = [dictionary.doc2bow(text) for text in texts]
            corpora.MmCorpus.serialize(corpus_path, corpus)  # store to disk, for later use

        else:
            # load id->word mapping (the dictionary), one of the results of step 2 above
            dictionary = gensim.corpora.Dictionary.load(dict_path)
            # load corpus iterator
            corpus = gensim.corpora.MmCorpus(corpus_path)
            # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

        return dictionary, corpus

    def load_or_train_LDA(self, model_path, topic_number):
        if not os.path.exists(config['gensim_dict_path']):
            os.makedirs(config['gensim_dict_path'])

        dict_path           = config['gensim_dict_path'] % config['data_name']
        corpus_path         = config['gensim_corpus_path'] % config['data_name']
        id2word, mmcorpus   = self.load_dict_corpus(dict_path, corpus_path)

        if os.path.exists(model_path):
            lda_model = LdaModel.load(model_path)
        else:
            lda_model = LdaModel(mmcorpus, num_topics=topic_number)
            lda_model.save(model_path)

        return lda_model, id2word, mmcorpus

    def __init__(self, config):
        print('8.1-lda_feature')
        self.config = config
        model_path = config['lda_path'] % config['data_name']
        self.vector_length = config['lda_topic_number']
        self.lda_model, self.dict, self.corpus = self.load_or_train_LDA(model_path, config['lda_topic_number'])

    def vectorize(self, x):
        '''
        for a sentence x
        :param x:
        :return:
        '''
        vec         = self.dict.doc2bow(x.lower().split())
        topic_vec   = self.lda_model[vec]

        vec = np.zeros((self.vector_length)).reshape(-1,1)
        for i,v in topic_vec:
            vec[i] += v
        return vec

    def fit(self, x, y=None):
        '''
        for an array of sentence
        :param x:
        :return:
        '''
        return self

    def transform(self, utterances):
        return_list = []

        for k, utt in enumerate(utterances):
            vec_dict = {}
            vec      = self.vectorize(utt)
            for i,v in enumerate(vec):
                vec_dict['topic_%d' % i] = v

            return_list.append(vec_dict)

        return return_list

def normalize(word_vec):
    norm = np.sqrt(sum([i**2 for i in word_vec]))
    # norm=np.linalg.norm(word_vec) # norm gives an irregular large value on word2vec
    if norm == 0:
        return word_vec

    if np.isnan(norm):
        vec = np.zeros_like(word_vec)
        return vec
    else:
        return word_vec/norm

class CosineSimilarity(BaseEstimator, TransformerMixin):
    '''
    8.2 & 8.4 Cosine Similarity after LDA/W2V vectorization
    '''
    def __init__(self, config, vectorizer):
        self.config     = config
        self.vectorizer = vectorizer

    def fit(self, x, y=None):
        return self

    def transform(self, utt_lists):
        return_list = []
        for s1, s2 in zip(utt_lists[0], utt_lists[1]):
            v1 = normalize(self.vectorizer.vectorize(s1)).reshape(1,-1)
            v2 = normalize(self.vectorizer.vectorize(s2)).reshape(1,-1)

            return_list.append({'cosine': cosine_similarity(v1, v2)[0][0]})

        return return_list

class Word2VecFeature(BaseEstimator, TransformerMixin):
    '''
    9.1 Word2Vec_features
    '''
    def __init__(self, config):
        self.config = config
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format(config['w2v_path'], binary=True).wv
        self.vector_length = config['w2v_vector_length']

    def vectorize(self, x):
        '''
        for a sentence x
        :param x:
        :return:
        '''
        vec         = np.average([self.model.wv[w] for w in x.lower().split() if w in self.model.wv], axis=0)
        if vec.any() == None or vec.any() == 'nan' or vec.shape != (self.vector_length, ):
            vec = np.zeros((self.vector_length, )).reshape(-1,1)
        return vec

    def fit(self, x, y=None):
        return self

    def transform(self, utterances):
        return_list = []

        for k, utt in enumerate(utterances):
            vec_dict = {}
            vec         = self.vectorize(utt)
            for i,v in enumerate(vec):
                vec_dict['w2v_dim=%d' % i] = v

            return_list.append(vec_dict)

        return return_list

class WmdDistance(BaseEstimator, TransformerMixin):
    '''
    9.3 WMD_features
    '''
    def __init__(self, config, vectorizer):
        self.config = config
        self.model  = vectorizer.model

    def fit(self, x, y=None):
        return self

    def transform(self, utt_lists):
        return_list = []

        for s1, s2 in zip(utt_lists[0], utt_lists[1]):
            # Some sentences to test.
            s1 = s1.lower().split()
            s2 = s2.lower().split()
            # Remove their stopwords.
            stopwords = nltk.corpus.stopwords.words('english')
            s1 = [w for w in s1 if w not in stopwords]
            s2 = [w for w in s2 if w not in stopwords]
            # Compute WMD.
            distance = self.model.wmdistance(s1, s2)

            return_list.append({'wmd_distance': distance})

        return return_list

class Doc2VecFeature(BaseEstimator, TransformerMixin):
    '''
    10.1 Doc2Vec_features
    '''
    def get_id(self, doc_str):
        '''
        Given a str, return its ID
        :param doc_str:
        :return:
        '''
        return self.doc2idx_dict[doc_str]

    def __init__(self, config):
        self.config = config
        self.doc2idx_dict = {}
        self.d2v_model, self.d2v_vector, self.doc2idx_dict = self.load_or_train_D2V()

    def load_or_train_D2V(self):
        '''
        Load or train Doc2Vec
        '''
        if os.path.exists(self.config['d2v_model_path'] % self.config['data_name']) and os.path.exists(self.config['d2v_vector_path'] % self.config['data_name']):
            d2v_model  = Doc2Vec.load(self.config['d2v_model_path'] % self.config['data_name'])
            d2v_vector, doc2idx_dict = data_loader.deserialize_from_file(self.config['d2v_vector_path'] % self.config['data_name'])
        else:
            all_sessions = self.config['data_loader']()
            document_dict = {}
            doc2idx_dict  = {}
            documents = []
            for session in all_sessions:
                for utt in session:
                    if utt.msg_text not in document_dict:
                        doc_id = len(document_dict)
                        words  = gensim.utils.to_unicode(utt.msg_text.strip().lower()).split()
                        doc = TaggedDocument(words=words, tags=[doc_id])
                        document_dict[utt.msg_text] = (doc_id, doc)
                        doc2idx_dict[utt.msg_text] = doc_id

                        documents.append(doc)
                        if doc_id != documents.index(doc):
                            assert 'docid Error'

            # d2v_model = Doc2Vec(size=self.config['d2v_vector_length'], window=self.config['d2v_window_size'], min_count=self.config['d2v_min_count'], workers=4, alpha=0.025, min_alpha=0.025) # use fixed documents rate
            d2v_model = Doc2Vec(size=self.config['d2v_vector_length'], window=self.config['d2v_window_size'], min_count=self.config['d2v_min_count'], workers=4)
            d2v_model.build_vocab(documents)
            d2v_model.intersect_word2vec_format(self.config['w2v_path'], binary=True)
            for epoch in range(10):
                d2v_model.train(documents, total_examples=len(documents), epochs=1)
                # d2v_model.alpha -= 0.002  # decrease the learning rate
                # d2v_model.min_alpha = d2v_model.alpha  # fix the learning rate, no decay

            d2v_vector = d2v_model.docvecs

            # store the model to mmap-able files
            d2v_model.save(self.config['d2v_model_path'] % self.config['data_name'])
            data_loader.serialize_to_file([d2v_vector, doc2idx_dict], self.config['d2v_vector_path'] % self.config['data_name'])

        return d2v_model, d2v_vector, doc2idx_dict

    def vectorize(self, x):
        '''
        for a sentence x
        :param x:
        :return:
        '''
        if x in self.doc2idx_dict:
            return self.d2v_model.docvecs[self.get_id(x)]
        else:
            return self.d2v_model.infer_vector(x.strip().lower().split())

    def fit(self, x, y=None):
        '''
        for an array of sentence
        :param x:
        :return:
        '''
        return self

    def transform(self, utterances):
        return_list = []

        for k, utt in enumerate(utterances):
            vec_dict    = {}
            vec         = self.vectorize(utt)
            for i,v in enumerate(vec):
                vec_dict['d2v_dim=%d' % i] = v

            return_list.append(vec_dict)

        return return_list

class SkipThoughtFeature(BaseEstimator, TransformerMixin):
    '''
    11.1 SkipThought_features
    '''
    def words_to_one_hot(self, str):
        '''
        Given a str, return its ID
        :param doc_str:
        :return:
        '''
        str = str.strip().lower()
        word_list = str.split()
        one_hot = [self.word2idx[w] for w in word_list if w in self.word2idx]

        if len(str) == 0 or len(one_hot) == 0:
            word_list = ['UNK']
            one_hot = [self.word2idx[w] for w in word_list if w in self.word2idx]

        print(word_list)
        print(one_hot)

        return torch.autograd.Variable(torch.LongTensor([one_hot]))

    def __init__(self, config):
        self .config = config
        self.word2idx = {}
        self.model, self.st_vector, self.word2idx, self.vocab = self.load_or_train_SkipThought()

    def load_or_train_SkipThought(self):
        '''
        Load or train Doc2Vec
        '''
        if os.path.exists(self.config['skipthought_data_path'] % self.config['data_name']):
            model, st_vector, self.word2idx, vocab = data_loader.deserialize_from_file(self.config['skipthought_data_path'] % self.config['data_name'])
        else:
            dir_st = self.config['skipthought_model_path']

            # vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']
            vocab = set()

            all_sessions = self.config['data_loader']()

            document_dict = {}
            for session in all_sessions:
                for utt in session:
                    if utt.msg_text not in document_dict:
                        words = utt.msg_text.strip().lower().split()
                        [vocab.add(w) for w in words]
                        document_dict[utt.msg_text] = (utt.msg_text, words)

            vocab       = ['<eos>', 'UNK'] + list(vocab)
            for id, word in enumerate(vocab):
                self.word2idx[word] = id

            st_vector =  {}
            model = BiSkip(dir_st, vocab)
            tally = 0
            for text, words in document_dict.values():
                tally += 1
                if tally % 100 == 0:
                    print('%d/%d' % (tally, len(document_dict)))

                one_hot = self.words_to_one_hot(text)
                st_vector[text] = model(one_hot).data.numpy()[0]

            # store the model to mmap-able files
            data_loader.serialize_to_file([model, st_vector, self.word2idx, vocab], self.config['skipthought_data_path'] % self.config['data_name'])

        return model, st_vector, self.word2idx, vocab

    def vectorize(self, x):
        '''
        for a sentence x
        :param x:
        :return:
        '''
        print(x)
        if x in self.st_vector:
            return self.st_vector[x]
        else:
            return self.model(self.words_to_one_hot(x)).data.numpy()[0]

    def fit(self, x, y=None):
        '''
        for an array of sentence
        :param x:
        :return:
        '''
        return self

    def transform(self, utterances):
        return_list = []

        for k, utt in enumerate(utterances):
            vec_dict    = {}
            vec         = self.vectorize(utt)
            for i,v in enumerate(vec):
                vec_dict['skipthought_dim=%d' % i] = v

            return_list.append(vec_dict)

        return return_list


class Feature_Extractor():
    def __init__(self, config_arg):
        self.config = config_arg
        self.logger = self.config.logger
        global logger
        global config
        logger = self.logger
        config = self.config

    def nan_to_num(x):
        x = _nx.array(x, subok=True, copy=True)
        xtype = x.dtype.type
        if not issubclass(xtype, _nx.inexact):
            return x

        iscomplex = issubclass(xtype, _nx.complexfloating)
        isscalar = (x.ndim == 0)

        x = x[None] if isscalar else x
        dest = (x.real, x.imag) if iscomplex else (x,)

        for d in dest:
            _nx.copyto(d, 0.0, where=umath.isnan(d))
            _nx.copyto(d, 0.0, where=umath.isposinf(d))
            _nx.copyto(d, 0.0, where=umath.isneginf(d))
        return x[0] if isscalar else x

    def do_extract(self):
        '''
        Extract features from utterances
        '''
        '''
        Define feature extractors
        '''
        '''
        Define transformer_list
        '''
        transformer_list = []

        # extract features from each utterance in range
        for utterance_type in self.config['utterance_range']:
            # 1. Basic Feature
            transformer_list.append(
                ('1-utterance_length.%s' % utterance_type, Pipeline([
                    ('selector', ItemSelector(keys=[utterance_type])),
                    ('utterance_length', UtteranceLength()),
                    ('vectorize', DictVectorizer()),
                ]))
            )
            # 2. User Action Feature (only applicable to user utterances)
            # 2.1 user action words
            if utterance_type.endswith('user_utterance'):
                transformer_list.append(
                    ('2.1-user_action_words.%s' % utterance_type, Pipeline([
                        ('selector', ItemSelector(keys=[utterance_type])),
                        ('user_action_words', UserActionWords(self.config['action_words'])),
                        ('vectorize', DictVectorizer()),
                    ]))
                )

            # 2.2 number of user action words
            if utterance_type.endswith('user_utterance'):
                transformer_list.append(
                    ('2.2-#user_action_words.%s' % utterance_type, Pipeline([
                        ('selector', ItemSelector(keys=[utterance_type])),
                        ('#user_action_words', NumberOfUserActionWords(self.config['action_words'])),
                        ('vectorize', DictVectorizer()),
                    ]))
                )

            # 2.3 jaccard_similarity of user action words
            if utterance_type == 'current_user_utterance':
                transformer_list.append(
                    ('2.3.1-action_jaccard_similarity.next_user_utterance_pairs', Pipeline([
                        ('selector', ItemSelector(keys=['next_user_utterance_pairs'])),
                        ('action_jaccard_similarity', UserActionJaccardSimilarity(self.config['action_words'])),
                        ('vectorize', DictVectorizer()),
                    ]))
                )
                transformer_list.append(
                    ('2.3.2-action_jaccard_similarity.last_user_utterance_pairs', Pipeline([
                        ('selector', ItemSelector(keys=['last_user_utterance_pairs'])),
                        ('action_jaccard_similarity', UserActionJaccardSimilarity(self.config['action_words'])),
                        ('vectorize', DictVectorizer()),
                    ]))
                )

            # 3. Time Feature
            if utterance_type == 'current_user_utterance':
                transformer_list.append(
                    ('3-time_feature', Pipeline([
                        ('selector', ItemSelector(keys=['x_raw'])),
                        ('time_feature', TimeFeature()),
                        ('vectorize', DictVectorizer()),
                    ]))
                )

            # 4. Lexical Feature
            #   4.1 n_gram
            # Perform an IDF normalization on the output of HashingVectorizer
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                               ngram_range=(1,3), lowercase=True,
                                               use_idf=True, smooth_idf = True,
                                               max_features = None,
                                               stop_words='english')
            n_gram_pipe = Pipeline([
                ('selector', ItemSelector(keys=[utterance_type])),
                ('tfidf', tfidf_vectorizer)
            ])
            transformer_list.append(
                ('4.1-ngram.%s' % utterance_type, n_gram_pipe)
            )

            if utterance_type == 'current_user_utterance':
                #   4.2 edit_distance
                transformer_list.append(
                    ('4.2.1-edit_distance.next_user_utterance_pairs', Pipeline([
                        ('selector', ItemSelector(keys=['next_user_utterance_pairs'])),
                        ('edit_distance', EditDistance()),
                        ('vectorize', DictVectorizer()),
                    ]))
                )
                transformer_list.append(
                    ('4.2.2-edit_distance.last_user_utterance_pairs', Pipeline([
                        ('selector', ItemSelector(keys=['last_user_utterance_pairs'])),
                        ('edit_distance', EditDistance()),
                        ('vectorize', DictVectorizer()),
                    ]))
                )
                #   4.3 jaccard_similarity
                transformer_list.append(
                    ('4.3.1-jaccard_similarity.next_user_utterance_pairs', Pipeline([
                        ('selector', ItemSelector(keys=['next_user_utterance_pairs'])),
                        ('jaccard_similarity', JaccardSimilarity()),
                        ('vectorize', DictVectorizer()),
                    ]))
                )
                transformer_list.append(
                    ('4.3.2-jaccard_similarity.last_user_utterance_pairs', Pipeline([
                        ('selector', ItemSelector(keys=['last_user_utterance_pairs'])),
                        ('jaccard_similarity', JaccardSimilarity()),
                        ('vectorize', DictVectorizer()),
                    ]))
                )

        # 5. Phrasal Feature
        transformer_list.append(
            ('5-phrase_feature', Pipeline([
                ('selector', ItemSelector(keys=['noun_phrases__%s' % type for type in self.config['utterance_range']])),
                ('phrase_feature', PhraseFeature(self.config)),
                ('vectorize', DictVectorizer()),
            ]))
        )

        # 6. Entity Feature
        transformer_list.append(
            ('6-entity_feature', Pipeline([
                ('selector', ItemSelector(keys=['parsed_results__%s' % type for type in self.config['utterance_range']])),
                ('entity_feature', EntityFeature(self.config)),
                ('vectorize', DictVectorizer()),
            ]))
        )

        # 7. Syntactic Feature
        transformer_list.append(
            ('7-syntactic_feature', Pipeline([
                ('selector', ItemSelector(keys=['parsed_results__%s' % type for type in self.config['utterance_range']])),
                ('syntactic_feature', SyntacticFeature(self.config)),
                ('vectorize', DictVectorizer()),
            ]))
        )
        # 8. Semantic/Topic Feature
        # 8.1 LDA Feature
        lda_feature_extractor = LDAFeature(self.config)
        for utterance_type in self.config['utterance_range']:
            transformer_list.append(
                ('8.1-lda_feature.%s' % utterance_type, Pipeline([
                    ('selector', ItemSelector(keys=[utterance_type])),
                    ('lda_feature', lda_feature_extractor),
                    ('vectorize', DictVectorizer()),
                ]))
            )

        # 8.2 LDA similarity
        transformer_list.append(
            ('8.2.1-lda_similarity.next_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'next_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, lda_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )
        transformer_list.append(
            ('8.2.2-lda_similarity.last_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'last_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, lda_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )


        # 9.1 W2V Feature
        w2v_feature_extractor = Word2VecFeature(self.config)
        for utterance_type in self.config['utterance_range']:
            transformer_list.append(
                ('9.1-w2v_feature.%s' % utterance_type, Pipeline([
                    ('selector', ItemSelector(keys=[utterance_type])),
                    ('w2v_feature', w2v_feature_extractor),
                    ('vectorize', DictVectorizer()),
                ]))
            )

        # 9.2 W2V similarity
        transformer_list.append(
            ('9.2.1-w2v_similarity.next_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'next_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, w2v_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )
        transformer_list.append(
            ('9.2.2-w2v_similarity.last_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'last_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, w2v_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )

        # 9.3 WMD Distance
        transformer_list.append(
            ('9.3.1-wmd_similarity.next_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'next_user_utterance'])),
                ('wmd_similarity', WmdDistance(self.config, w2v_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )
        transformer_list.append(
            ('9.3.2-wmd_similarity.last_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'last_user_utterance'])),
                ('wmd_similarity', WmdDistance(self.config, w2v_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )

        # 10.1 Doc2Vec
        d2v_feature_extractor = Doc2VecFeature(self.config)
        for utterance_type in self.config['utterance_range']:
            transformer_list.append(
                ('10.1-d2v_feature.%s' % utterance_type, Pipeline([
                    ('selector', ItemSelector(keys=[utterance_type])),
                    ('d2v_feature', d2v_feature_extractor),
                    ('vectorize', DictVectorizer()),
                ]))
            )

        # 10.2 D2V similarity
        transformer_list.append(
            ('10.2.1-d2v_similarity.next_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'next_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, d2v_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )
        transformer_list.append(
            ('10.2.2-d2v_similarity.last_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'last_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, d2v_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )

        # 11.1 Skip-thought
        skipthought_feature_extractor = SkipThoughtFeature(self.config)
        for utterance_type in self.config['utterance_range']:
            transformer_list.append(
                ('11.1-skipthought_feature.%s' % utterance_type, Pipeline([
                    ('selector', ItemSelector(keys=[utterance_type])),
                    ('skipthought_feature', skipthought_feature_extractor),
                    ('vectorize', DictVectorizer()),
                ]))
            )

        # 11.2 Skip-thought similarity
        transformer_list.append(
            ('11.2.1-skipthought_similarity.next_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'next_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, skipthought_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )
        transformer_list.append(
            ('11.2.2-skipthought_similarity.last_user_utterance_pairs', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'last_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, skipthought_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )
        '''
        Run the extracting pipeline
        '''
        union_features = FeatureUnion(transformer_list=transformer_list)

        pipeline = Pipeline([
            # 1. Set the context utterances up
            ('raw_feature_extractor', RawFeatureExtractor(self.config)),
            # 2. Use FeatureUnion to extract features
            ('union', union_features)
        ])

        X = pipeline.fit_transform(self.config['X_raw'])

        # set vocabulary of union_features by concatenating all its sub-vocabularies
        # [IMPORTANT!!!]add the following function to Pipeline
        '''
        from operator import itemgetter
        
        def get_feature_names(self):
            self.feature_names_ = [f for f, i in sorted(six.iteritems(self.vocabulary_),
                                                        key=itemgetter(1))]
            return self.feature_names_
        '''
        for transformer in transformer_list:
            if transformer[1].__class__.__name__ == 'Pipeline':
                transformer[1].vocabulary_ = transformer[1].steps[-1][1].vocabulary_

        feature_names = union_features.get_feature_names() # used for masking
        data_loader.serialize_to_file([X, feature_names], self.config['extracted_feature_path'] % self.config['data_name'])
        # joblib.dump(transformer_list, self.config['pipeline_path'] % self.config['data_name'])
        # data_loader.serialize_to_file_by_dill([pipeline, union_features, transformer_list], self.config['pipeline_path'] % self.config['data_name'])


        return X, feature_names

    def extract(self):
        if os.path.exists(self.config['extracted_feature_path'] % self.config['data_name']):
            X, feature_names = data_loader.deserialize_from_file(self.config['extracted_feature_path'] % self.config['data_name'])
            # transformer_list = joblib.load(self.config['pipeline_path'] % self.config['data_name'])
            # pipeline, union_features, transformer_list = data_loader.deserialize_from_file_by_dill(self.config['pipeline_path'] % self.config['data_name'])
        else:
            X, feature_names   = self.do_extract()

        self.config['X']                    = X
        self.config['feature_names']        = feature_names
        # self.config['pipeline']             = pipeline
        # self.config['union_features']       = union_features
        # self.config['transformer_list']     = transformer_list

        return X, feature_names

    def extract_raw_feature(self):
        '''
        Extract and return the basic things by loading the transformer RawFeatureExtractor(), before it's embedded in do_extract()
        But for doc2vec, I only need these sentences

            ('raw_feature_extractor', RawFeatureExtractor(self.config))
        '''
        extractor = RawFeatureExtractor(self.config)
        X_raw_feature = extractor.fit_transform(self.config['X_raw'])

        return X_raw_feature


    def split_to_instances(self, annotated_sessions):
        '''
        :param annotated_sessions:
        :return:
        X_raw = [], a list of dicts, contains all the usable information of current utterance (utterance, index, and dialogue)
        Y_raw = [], true labels, after mapping to [0,..., n-1] by LabelEncoder
        '''
        X_raw = []
        Y_raw = []

        valid_type = self.config.param['valid_type']

        for session in annotated_sessions:
            # convert the corpus to individual training instances in desired raw format ('index', 'utterance' and the contextual 'dialogue')
            for u_idx, u_ in enumerate(session):
                if u_.direction != 'user_to_sb':
                    continue
                if u_.type == None or u_.type == '' or u_.type not in valid_type:
                    continue

                raw_features = {}
                raw_features['index'] = u_idx # position of current utterance
                raw_features['utterance'] = u_ # current utterance
                raw_features['dialogue'] = session # the whole dialogue as context
                X_raw.append(raw_features)
                Y_raw.append(u_.type)

        le = preprocessing.LabelEncoder()
        le.fit(Y_raw)
        # list(le.classes_)
        Y = le.transform(Y_raw)

        self.config['label_encoder']   = le
        self.config['X_raw']           = X_raw
        self.config['Y']               = Y
        return X_raw, Y, le