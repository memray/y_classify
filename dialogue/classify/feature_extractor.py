# -*- coding: utf-8 -*-
import os

import nltk
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity

import leven
from nltk.stem.porter import *
from nltk.corpus import stopwords

from dialogue.data import data_loader
import numpy.core.numeric as _nx
from numpy.core import getlimits, umath

# initialize the Stanford wrapper
from stanford_corenlp_pywrapper import CoreNLP
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
            corenlp = CoreNLP('nerparse')  # need to override corenlp_jars

            features = np.recarray(shape=(len(X_raw),),
                                   dtype=[('current_user_utterance', object),
                                          ('next_system_utterance', object),
                                          ('next_user_utterance', object),
                                          ('user_utterance_pairs', object),
                                          ('parsed_results__current_user_utterance', object),
                                          ('parsed_results__next_system_utterance', object),
                                          ('parsed_results__next_user_utterance', object),
                                          ('noun_phrases__current_user_utterance', object),
                                          ('noun_phrases__next_system_utterance', object),
                                          ('noun_phrases__next_user_utterance', object),
                                          ('x_raw', object)])

            next_system_utterance_count = 0
            next_user_utterance_count = 0
            for i, x_raw in enumerate(X_raw):
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
                    features['user_utterance_pairs'][i] = (x_raw['utterance'].msg_text, x_raw['dialogue'][x_index+2].msg_text)
                else:
                    features['next_user_utterance'][i] = ''
                    features['user_utterance_pairs'][i] = (x_raw['utterance'].msg_text, '')

            for i, x_raw in enumerate(X_raw):
                for utterance_type in config['utterance_range']:
                    # consists of ['deps_cc', 'pos', 'lemmas', 'tokens', 'char_offsets', 'ner', 'entitymentions', 'parse', 'deps_basic', 'normner']
                    parsed_results = corenlp.parse_doc(features[utterance_type][i])

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
                        # could be multiple sentences in one utterance, so merge them into one array
                        tokens = np.concatenate([[t.lower() for t in s['tokens']] for s in parsed_results['sentences']])
                        pos    = np.concatenate([s['pos'] for s in parsed_results['sentences']])
                        features['noun_phrases__%s' % utterance_type][i] = extract_noun_phrases(tokens, pos)

            # logger.info('has_next_system_utterance_count = %d/%d' % (next_system_utterance_count, len(X_raw)))
            # logger.info('has_next_user_utterance_count = %d/%d' % (next_user_utterance_count, len(X_raw)))

            logger.info('Saving raw feature of %s to %s' % (config['data_name'], feature_cache_path))
            data_loader.serialize_to_file(features, feature_cache_path)

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
        for k, np_record in enumerate(zip(np_records[0],np_records[1],np_records[2])):
            # iterate each type of utterance
            np_dict = {}

            # 5.1 noun_phrase: one-hot representation of all noun phrases (extracted by POS-tagging patterns)
            for i in range(3):
                # iterate each phrase in the sentence, np[0] is words, np[1] is postags
                for np_ in np_record[i]:
                    np_dict['noun_phrases__%s__%s' % (self.config['utterance_range'][i], '_'.join(np_[0]))] = True
            current_np_set  = set(['_'.join(np_[0]) for np_ in np_record[0]])
            next_np_set     = set(['_'.join(np_[0]) for np_ in np_record[2]])
            inter_np        = set.intersection(current_np_set, next_np_set)

            # 5.2 noun_phrase_overlap:  True, if there is any NP overlap between two user utterances
            if len(inter_np) > 0:
                np_dict['noun_phrases__have_overlap']    = True
                np_dict['noun_phrases__#overlap']        = len(inter_np)
            else:
                np_dict['noun_phrases__have_overlap']    = False
                np_dict['noun_phrases__#overlap']        = 0

            # 5.3 np_jaccard_similarity: Jaccard similarity of NPs.
            np_dict['noun_phrases__jaccard_similarity']  = jaccard_similarity(current_np_set, next_np_set)

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

        # iterate each data sample, contains three parts (currrent_user, current_system, next_user)
        for k, parse_record in enumerate(zip(parse_records[0],parse_records[1],parse_records[2])):
            # iterate each type of utterance
            entity_dict = {}

            entity_lists = []
            # 6.1 entity: one-hot representation of all entities
            for i in range(3):
                entity_list = []
                cache = []
                # iterate each word and NER tag
                if len(parse_record[i]) > 0:
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
                entity_lists.append(entity_list)

            current_entity_set  = set([e for e in entity_lists[0]])
            next_entity_set     = set([e for e in entity_lists[2]])
            inter_entity        = set.intersection(current_entity_set, next_entity_set)

            # 6.2 entity_overlap:  True, if there is any entity overlap between two user utterances
            if len(inter_entity) > 0:
                entity_dict['entity__have_overlap']    = True
                entity_dict['entity__#overlap']        = len(inter_entity)
            else:
                entity_dict['entity__have_overlap']    = False
                entity_dict['entity__#overlap']        = 0

            # 6.3 entity_jaccard_similarity: Jaccard similarity of entities.
            entity_dict['entity__jaccard_similarity']  = jaccard_similarity(current_entity_set, next_entity_set)

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
        for k, parse_record in enumerate(zip(parse_records[0],parse_records[1],parse_records[2])):
            # iterate each type of utterance
            entity_dict = {}

            # 7.1-7.3 key syntactic components
            current_root    = set()
            next_root       = set()
            current_subj    = set()
            next_subj       = set()
            current_obj     = set()
            next_obj        = set()

            for i in range(3):
                if len(parse_record[i]) > 0:
                    # iterate each parsed sentence
                    for sent in parse_record[i]:
                        # 7.1 root_word: the word that at the root of parse tree (shot).
                        for dep in sent['deps_basic']:
                            if dep[0] == 'root':
                                entity_dict['root_word__%s__%s' % (self.config['utterance_range'][i], sent['tokens'][dep[2]])] = True
                                root_index = dep[2]
                                if i == 0:
                                    current_root.add(stemmer.stem(sent['tokens'][dep[2]].lower()))
                                elif i == 2:
                                    next_root.add(stemmer.stem(sent['tokens'][dep[2]].lower()))
                                break

                        # 7.2 subj_word: the topmost subjects
                        for dep in sent['deps_basic']:
                            if dep[0].endswith('subj') and dep[1] == root_index:
                                entity_dict['subj_word__%s__%s' % (self.config['utterance_range'][i], sent['tokens'][dep[2]])] = True
                                if i == 0:
                                    current_subj.add(stemmer.stem(sent['tokens'][dep[2]].lower()))
                                elif i == 2:
                                    next_subj.add(stemmer.stem(sent['tokens'][dep[2]].lower()))

                        # 7.3 obj_word: the topmost object
                        for dep in sent['deps_basic']:
                            if dep[0].endswith('obj') and dep[1] == root_index:
                                entity_dict['obj_word__%s__%s' % (self.config['utterance_range'][i], sent['tokens'][dep[2]])] = True
                                if i == 0:
                                    current_obj.add(stemmer.stem(sent['tokens'][dep[2]].lower()))
                                elif i == 2:
                                    next_obj.add(stemmer.stem(sent['tokens'][dep[2]].lower()))

            # 7.4-7.6 root_words_overlap: True if any of the root words of two user utterances are same.
            if len(set.intersection(current_root, next_root)) > 0:
                entity_dict['root_words_overlap'] = True
            else:
                entity_dict['root_words_overlap'] = False

            if len(set.intersection(current_subj, next_subj)) > 0:
                entity_dict['subj_words_overlap'] = True
            else:
                entity_dict['subj_words_overlap'] = False

            if len(set.intersection(current_obj, next_obj)) > 0:
                entity_dict['obj_words_overlap'] = True
            else:
                entity_dict['obj_words_overlap'] = False

            return_list.append(entity_dict)
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
    8.3 Word2Vec_features
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
    8.5 WMD_features
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
                    ('2.3-action_jaccard_similarity', Pipeline([
                        ('selector', ItemSelector(keys=['user_utterance_pairs'])),
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
                    ('4.2-edit_distance', Pipeline([
                        ('selector', ItemSelector(keys=['user_utterance_pairs'])),
                        ('edit_distance', EditDistance()),
                        ('vectorize', DictVectorizer()),
                    ]))
                )
                #   4.3 jaccard_similarity
                transformer_list.append(
                    ('4.3-jaccard_similarity', Pipeline([
                        ('selector', ItemSelector(keys=['user_utterance_pairs'])),
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
            ('8.2-lda_similarity', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'next_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, lda_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )

        # 8.3 W2V Feature
        w2v_feature_extractor = Word2VecFeature(self.config)
        for utterance_type in self.config['utterance_range']:
            transformer_list.append(
                ('8.3-w2v_feature.%s' % utterance_type, Pipeline([
                    ('selector', ItemSelector(keys=[utterance_type])),
                    ('w2v_feature', w2v_feature_extractor),
                    ('vectorize', DictVectorizer()),
                ]))
            )

        # 8.4 W2V similarity
        transformer_list.append(
            ('8.4-w2v_similarity', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'next_user_utterance'])),
                ('cosine_similarity', CosineSimilarity(self.config, w2v_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )

        # 8.5 WMD Distance
        transformer_list.append(
            ('8.5-wmd_distance', Pipeline([
                ('selector', ItemSelector(keys=['current_user_utterance', 'next_user_utterance'])),
                ('wmd_distance', WmdDistance(self.config, w2v_feature_extractor)),
                ('vectorize', DictVectorizer()),
            ]))
        )

        '''
        Run the extracting pipeline
        '''
        union_features = FeatureUnion(transformer_list=transformer_list)

        pipeline = Pipeline([
            ('raw_feature_extractor', RawFeatureExtractor(self.config)),
            # Use FeatureUnion to combine the features
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