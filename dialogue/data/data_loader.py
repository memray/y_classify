# -*- coding: utf-8 -*-
import csv
import json
import pickle
import random

import dill
import numpy as np
import six
import os
import sys
import re

from collections import Counter
from bs4 import BeautifulSoup

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return identifier

def data_loader(identifier, kwargs=None):
    '''
    load testing data dynamically
    :return:
    '''
    data_loader = get_from_module(identifier, globals(), 'data_loader', instantiate=True,
                                  kwargs=kwargs)
    return data_loader


def deserialize_from_file(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def serialize_to_file(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    print('serialize to %s' % path)
    f = open(path, 'wb')
    pickle.dump(obj, f, protocol=protocol)
    f.close()

def deserialize_from_file_by_dill(path):
    f = open(path, 'rb')
    obj = dill.load(f)
    f.close()
    return obj

def serialize_to_file_by_dill(obj, path):
    print('serialize to %s' % path)
    f = open(path, 'wb')
    dill.dump(obj, f)
    f.close()

class Utterance():
    def __init__(self, session_id , time, userid, direction, text, botlog=None, **kwargs):
        self.session_id = session_id
        self.time = time
        self.userid = userid
        self.direction = direction
        # # self.platform = record[3]
        # # self.msg_sentto = record[4]
        # self.msg_types = record[5]
        # # self.msg_sentto_displayname = record[6]
        # # self.dt_day = record[7]
        # # self.ts_in_second = record[8]
        # # self.platform_message_id = record[9]
        # # self.botlog_intent = record[10]
        # # self.botlog_slots = record[11]
        self.msg_text  = text
        self.botlog  = botlog

        # add anything maybe useful later
        for (k,v) in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        str_ = ''
        # for attr in dir(self):
        #     if not attr.startswith('__') and getattr(self, attr) != None and getattr(self, attr)!='':
        # str_ += '\t\t%s : %s\n' % (attr, getattr(self, attr))
        str_ += '\t\t%s : %s\n' % ('session_id', getattr(self, 'session_id'))
        str_ += '\t\t%s : %s\n' % ('useruuid', getattr(self, 'useruuid'))
        str_ += '\t\t%s : %s\n' % ('time', getattr(self, 'time'))
        str_ += '\t\t%s : %s\n' % ('direction', getattr(self, 'direction'))
        str_ += '\t\t%s : %s\n' % ('msg_text', getattr(self, 'msg_text'))
        return str_

class DataLoader(object):
    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.logger = self.config.logger

        self.root_dir = self.config['root_path'] #os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))
        self.__dict__.update(kwargs)
        self.name    = self.__class__.__name__
        self.session_list    = []
        self.annoteted_list  = []

    class Dialogue():
        def __init__(self, session_id):
            self.utterances = []
            self.session_id = session_id

        def load_from_file(self, log_path, label_path, dialogue_number):
            log_json = json.load(open(log_path, 'r'))
            label_json = json.load(open(label_path, 'r'))

            self.path = log_path
            self.session_id = log_json['session-id']

            # logger.info(self.session_id + '*' * 20)


            for turn_id, (log_turn, label_turn) in enumerate(zip(log_json['turns'], label_json["turns"])):
                if 'transcript' in log_turn['output']:
                    self.utterances.append(Utterance(self.session_id, turn_id, dialogue_number, 'bot_to_sb', log_turn['output']['transcript']))
                else:
                    self.utterances.append(Utterance(self.session_id, turn_id, dialogue_number, 'bot_to_sb', ''))
                # logger.info('\t\t bot: %s' % self.utterances[-1].msg_text)

                if 'transcription' in label_turn:
                    self.utterances.append(Utterance(self.session_id, turn_id, dialogue_number, 'user_to_sb', label_turn['transcription']))
                else:
                    self.utterances.append(Utterance(self.session_id, turn_id, dialogue_number, 'user_to_sb', ''))
                    # logger.info('\t\t user:  %s' % self.utterances[-1].msg_text)

        def __len__(self):
            return len(self.utterances)

        def __iter__(self):
            return iter(self.utterances)

        def __getitem__(self,index):
            return self.utterances[index]

    def __call__(self):
        '''
        All the loading procedures are implemented in corresponding load functions, e.g. load_pig_log(), load_dstc()
        :return: a list of dialogues
        '''
        if self.__class__.__name__ == 'Family_Assistant' or self.__class__.__name__ == 'Yahoo_Weather':
            self.load_pig_log()
            return self.session_list
        if self.__class__.__name__.lower().startswith('dstc'):
            self.load_dstc()
            return self.session_list
        if self.__class__.__name__ == 'Maluuba' or self.__class__.__name__ == 'MATCH' or self.__class__.__name__ == 'GHOME':
            self.load_()
            return self.session_list

        raise 'Not Recognized Dataset Name'

    def load_annotated_data(self):
        '''
        Load the annotated CSV data
        :return: a list of dialogues
        '''
        if not os.path.exists(self.annotated_data_path):
            raise Exception('annotated file does not exist: %s' % self.annotated_data_path)

        dialogue_dict = {}
        with open(self.annotated_data_path, 'r') as annotated_csv:
            csv_file = csv.reader(annotated_csv)
            for row_num, csv_row in enumerate(csv_file):
                if row_num == 0:
                    header = csv_row
                    continue

                if csv_row == None or len(csv_row) == 0:
                    continue

                row = {}
                for c_idx, c_name in enumerate(header):
                    row[c_name] = csv_row[c_idx]

                session_id = row['conversation #'] # there is something wrong with the session_id in Family_Assistant
                dialogue = dialogue_dict.get(session_id, self.Dialogue(session_id))

                utt_ = Utterance(session_id, row['time'], row['userid'], row['direction'], row['msg_text'].strip(' []"'), '')

                # put all the other fields into utt_
                for (k,v) in row.items():
                    setattr(utt_, k, v.strip())

                if utt_.direction == 'user_to_sb':
                    utt_.type = utt_.Annotation.strip()
                    # 'Corrented_by_bot' is removed in the latest update
                    # if hasattr(utt_, 'Corrected_by_bot') and utt_.Corrected_by_bot != '':
                    #     utt_.type = 'CC'
                    # (temporary) overwrite the annotation with Rui's correction
                    if hasattr(utt_, 'Rui\'s Correction') and getattr(utt_, 'Rui\'s Correction') != '':
                        utt_.type = getattr(utt_, 'Rui\'s Correction').strip()

                # merge the multiple 'bot_to_sb' messages to one
                if utt_.direction == 'bot_to_sb' and len(dialogue) > 0 and dialogue.utterances[-1].direction == 'bot_to_sb':
                    dialogue.utterances[-1].msg_text += utt_.msg_text
                else:
                    dialogue.utterances.append(utt_)

                dialogue_dict[session_id] = dialogue

        self.annoteted_list = dialogue_dict.values()
        return dialogue_dict.keys(), dialogue_dict.values()

    def stats(self):
        session_count = len(self.session_list)
        user_utterance_count = sum([len([u for u in s if u.direction=='user_to_sb']) for s in self.session_list])
        system_utterance_count = sum([len([u for u in s if u.direction=='bot_to_sb']) for s in self.session_list])

        self.logger.info('session_count = %d' % session_count)
        self.logger.info('utterance_count = %d' % (user_utterance_count+system_utterance_count))
        self.logger.info('user_utterance_count = %d' % user_utterance_count)
        self.logger.info('system_utterance_count = %d' % system_utterance_count)
        if session_count > 0:
            average_session_length = float(user_utterance_count + system_utterance_count)/float(session_count)
        else:
            average_session_length = 0
        self.logger.info('average_session_length = %.5f' % (average_session_length))

        '''
        stats of annotated data
        '''
        if not os.path.exists(os.path.join(self.config.param['experiment_path'], 'data.stats.csv')):
            print_header = True
        else:
            print_header = False

        types = list(['F', 'C', 'R', 'N', 'CC', 'A', 'Chitchat', 'G', 'O', 'Total'])
        type_set = list(['F', 'C', 'R', 'N', 'CC', 'A', 'Chitchat', 'G', 'O', 'Total'])
        labels          = np.concatenate([[u_.type for u_ in s if u_.direction == 'user_to_sb' and u_.type in type_set] for s in self.annoteted_list]).ravel()
        counter = Counter(labels)
        counter['Total'] = len(labels)
        label_values    = [str(counter[l]) for l in types]

        with open(os.path.join(self.config.param['experiment_path'], 'data.stats.csv'), 'a') as csv_file:

            if print_header:
                csv_file.write('Dataset,' + ','.join(types)+'\n')

            csv_file.write(self.config.param['data_name'] + ',' + ','.join(label_values)+'\n')

        # if len(self.annoteted_list) > 0 :
        #     logger.info('*' * 20 + ' Annotated Data '+ '*' * 20 )
        #     labels = np.concatenate([[u_.type for u_ in s if u_.direction == 'user_to_sb'] for s in self.annoteted_list]).ravel()
        #     logger.info(Counter(labels))

        # length distribution
        # count_session_length = {}
        # for s in self.session_list:
        #     count_session_length[len(s)] = count_session_length.get(len(s), 0) + 1
        # sorted_length_count = sorted(count_session_length.items(), key=lambda x:x[0])
        # for k,v in sorted_length_count:
        #     logger.info('\t%d\t%d' % (k,v))

    def load_pig_log(self):
        '''
        Load family assistant and weather data
        '''
        self.logger.info(self.name + ' : ' + self.data_path)
        session_list = []
        session_count = 0

        with open(self.data_path, 'r') as f_:
            for session_line in f_.readlines():
                delimeter_idx = session_line.find('\t')
                user_id = session_line[:delimeter_idx]
                session_content = session_line[delimeter_idx+1:]

                if (session_content == None or session_content.strip() == ''):
                    continue

                session_json = json.loads(session_content)

                for s_ in session_json:
                    session = []
                    for record in s_:
                        u_ = Utterance('%s_%s' % (record['useruuid'], record['time']) , record['time'], record['useruuid'], record['direction'], record['msg_text'], record['botlog'])
                        session.append(u_)

                    session_list.append(session)

        self.session_list =  session_list

    def load_dstc(self):
        dialogue_list = []
        data_dir = self.data_path
        DATA_NAME = self.name

        if os.path.exists(os.path.join(data_dir, DATA_NAME+'.pkl')):
            with open(os.path.join(data_dir, DATA_NAME+'.pkl'), 'rb') as f_:
                dialogue_list = pickle.load(f_)
            self.logger.info('Loading %s cache complete' % DATA_NAME)
        else:
            dialogue_count = 0
            for first_dir in os.listdir(data_dir):
                if os.path.isfile(data_dir+first_dir):
                    continue
                for second_dir in os.listdir(data_dir+first_dir):
                    if os.path.isfile(os.path.join(data_dir, first_dir, second_dir)):
                        continue
                    self.logger.info(os.path.join(data_dir, first_dir, second_dir))
                    for third_dir in os.listdir(os.path.join(data_dir, first_dir, second_dir)):
                        if DATA_NAME == 'DSTC1':
                            log_path = os.path.abspath(os.path.join(data_dir, first_dir, second_dir, third_dir, 'dstc.log.json'))
                            label_path = os.path.abspath(os.path.join(data_dir, first_dir, second_dir, third_dir, 'dstc.labels.json'))
                        elif DATA_NAME == 'DSTC2' or DATA_NAME == 'DSTC3':
                            log_path = os.path.abspath(os.path.join(data_dir, first_dir, second_dir, third_dir, 'log.json'))
                            label_path = os.path.abspath(os.path.join(data_dir, first_dir, second_dir, third_dir, 'label.json'))

                        if os.path.exists(log_path):
                            dialogue_count += 1
                            new_dialogue = self.Dialogue()
                            new_dialogue.load_from_file(log_path, label_path, dialogue_count)
                            dialogue_list.append(new_dialogue)

            with open(os.path.join(data_dir, DATA_NAME+'.pkl'), 'wb') as f_:
                pickle.dump(dialogue_list, f_)

        self.session_list = dialogue_list

    def export_ramdom_samples(self, N=100):
        '''
        Randomly sample N data points
        :return:
        '''
        # 1. convert to numpy array
        session_list = np.asarray(self.session_list)
        self.logger.info('Exporting samples to CSV file')
        # 2. load cache if exists, otherwise resample it, if N==1 means to export all
        root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))

        if N != 1:
            sample_cache_path = root_dir + '/dataset/sample/%s.N=%s.pkl' % (self.name, N)
            if os.path.exists(sample_cache_path):
                with open(sample_cache_path, 'rb') as f_:
                    sample_index = pickle.load(f_)
            else:
                sample_index = random.sample(range(len(session_list)), N)
                with open(sample_cache_path, 'wb') as f_:
                    pickle.dump(sample_index, f_)
            session_list = session_list[sample_index]

        # 3. dump the samples to a CSV file
        sample_path = root_dir + '/dataset/sample/%s.N=%s.sampled.csv' % (self.name, N)
        self.logger.info(sample_path)
        valid_count = 0

        with open(sample_path, 'w') as csvfile:
            # attrs = [attr for attr in dir(session_list[0][0]) if not attr.startswith('__')]
            # self.logger.info(attrs)
            # attrs = ['useruuid', 'time', 'direction', 'msg', 'botlog']
            attrs = ['session_id', 'userid', 'time', 'direction', 'msg_text']

            csvfile = csv.writer(csvfile)
            # csvfile = csv.writer(csvfile, delimiter='\t', lineterminator='\n')
            header = ['conversation #']
            header.extend(attrs)
            csvfile.writerow(header)

            for session_num, session in enumerate(session_list):
                for u_ in session:
                    row = [session_num + 1]
                    row.extend([getattr(u_, attr) for attr in attrs])
                    csvfile.writerow(row)
                csvfile.writerow([])

                valid_count += 1

        self.logger.info('#(valid samples) = %d' % valid_count)

class Family_Assistant(DataLoader):
    def __init__(self, **kwargs):
        super(Family_Assistant, self).__init__(**kwargs)
        self.annotated_data_path = os.path.join(self.root_dir, 'dataset', 'editorial_annotations', 'done', 'Done-Family_Assistant.N=1000.sampled.csv')

        self.data_path = os.path.join(self.root_dir, 'dataset', 'raw_datasets', 'Family_Assistant.20170306.interval=5min.session', 'part-v002-o000-r-00000')
        self.name = self.__class__.__name__

    def is_valid(self, session):
        if len(session) < 4:
            return False

        count_user_utterance = 0

        for u in session:
            if u.direction == 'user_to_sb':
                count_user_utterance += 1
            if u.direction == 'bot_to_sb' and u.botlog != None:
                bot_log_json = json.loads(u.botlog)
                if 'use_case' in bot_log_json and len(bot_log_json['use_case']) > 0 and bot_log_json['use_case'][0] != None and bot_log_json['use_case'][0].strip() == 'onboarding':
                    return False

        if count_user_utterance < 2:
            return False

        return True

    def filter_invalid_sessions(self):
        new_session_list = []

        for session_number, session in enumerate(self.session_list):
            # session = filter_on_board_utterances(BOT_NAME, session)
            if not self.is_valid(session):
                continue
            new_session_list.append(session)

        self.session_list = new_session_list

class Yahoo_Weather(DataLoader):
    def __init__(self, **kwargs):
        super(Yahoo_Weather, self).__init__(**kwargs)
        self.data_path = os.path.join(self.root_dir, 'dataset', 'raw_datasets', 'Weather.interval=5min.session', 'part-v002-o000-r-00000')
        self.name = self.__class__.__name__

class DSTC1(DataLoader):
    def __init__(self, **kwargs):
        super(DSTC1, self).__init__(**kwargs)
        self.data_path = os.path.join(self.root_dir, 'dataset', 'raw_datasets', 'DSTC1')

    def is_valid(self, session):
        count_trash_utterance = 0

        for u in session:
            if u.direction == 'user_to_sb' and u.msg_text.strip()=='NON_UNDERSTANDABLE':
                count_trash_utterance += 1
            if u.msg_text.strip() == '':
                count_trash_utterance += 1

        if len(session) < 4:
            return False

        if count_trash_utterance > 1:
            return False

        return True

    def filter_invalid_sessions(self):
        new_session_list = []

        for session_number, session in enumerate(self.session_list):
            # session = filter_on_board_utterances(BOT_NAME, session)
            if not self.is_valid(session):
                continue
            new_session_list.append(session)

        self.session_list = new_session_list

class DSTC2(DataLoader):
    def __init__(self, **kwargs):
        super(DSTC2, self).__init__(**kwargs)
        self.annotated_data_path = os.path.join(self.root_dir, 'dataset','editorial_annotations', 'done', 'Done-DSTC2.N=1000.sampled.csv')
        self.data_path = os.path.join(self.root_dir, 'dataset', 'raw_datasets', 'DSTC2')

class DSTC3(DataLoader):
    def __init__(self, **kwargs):
        super(DSTC3, self).__init__(**kwargs)
        self.annotated_data_path = os.path.join(self.root_dir, 'dataset','editorial_annotations', 'done', 'Done-Merged.DSTC3.N=1000.sampled.csv')
        self.data_path = os.path.join(self.root_dir, 'dataset', 'raw_datasets', 'DSTC3_test')

class GHOME(DataLoader):
    def __init__(self, **kwargs):
        super(GHOME, self).__init__(**kwargs)
        self.annotated_data_path = os.path.join(self.root_dir, 'dataset','editorial_annotations', 'done', 'Done-Merged.GHOME.N=1000.sampled.csv')

        self.raw_data_path = self.root_dir + '/dataset/GHome/raw-dialogue.csv'
        # self.data_path = self.root_dir + '/dataset/GHome/sorted-dialogue.csv'
        self.data_path = os.path.join(self.root_dir, 'dataset', 'raw_datasets', 'GHome', 'GHome-Utterances-All-with-Tags-Chronological.csv')
        self.need_sort = False

        if self.need_sort:
            turns_dict = {}
            in_csv = open(self.raw_data_path, 'r')
            out_csv = open(self.data_path, 'w')

            current_user = None
            current_time = None
            current_session = None
            cache = []

            for row_num, row_str in enumerate(in_csv):
                row_csv = list(csv.reader([row_str]))[0]

                if row_num == 0:
                    header = row_csv
                    out_csv.write(row_str)
                    continue
                row = {}
                for c_idx, c_name in enumerate(header):
                    row[c_name] = row_csv[c_idx]

                if row['user'] != current_user or row['datetime'] != current_time or row['session'] != current_session:
                    # for str_ in cache[::-1]:
                    for str_ in cache:
                        out_csv.write(str_)
                    cache = [row_str]
                    current_user = row['user']
                    current_time = row['datetime']
                    current_session = row['session']
                else:
                    cache.append(row_str)

            for str_ in cache:
                out_csv.write(str_)
            in_csv.close()
            out_csv.close()

    def load_(self):
        session_dict = {}
        count = 0
        with open(self.data_path, 'r') as f_:
            data_csv = csv.reader(f_)

            for row_num, row_csv in enumerate(data_csv):
                if row_num == 0:
                    header = row_csv
                    continue

                # if row_csv[12].strip().startswith('Location info'):
                #     self.logger.info(row_csv)
                #     count += 1

                row = {}
                for c_idx, c_name in enumerate(header):
                    row[c_name] = row_csv[c_idx]

                if row['r1'] == 'com.google.homeautomation': # or row['r1'] == 'Location info':
                    for i in range(1, 6):
                        row['r%d' % i] = row['r%d' % (i+1)]

                for i in range(6, 0, -1):
                    # find the index of last reply
                    if row['r%d' % i].strip() != '':
                        row['last_index'] = i

                        if row['last_index'] > 1:
                            str_ = row['r1']
                            for j in range(2, i+1):
                                str_ += ' '+row['r%d' % j]
                                row['r%d' % j] = ''

                            row['r1'] = str_
                            break

                session_id = '%s_%s' % (row['user'], row['session'])


                # filter out the empty utterances
                dialog_ = session_dict.get(session_id, self.Dialogue(session_id))
                if row['utterance'].strip() != '':
                    utt_ = Utterance(session_id, row['datetime'], row['user'], 'user_to_sb', row['utterance'], row['intent'], **row)
                    dialog_.utterances.append(utt_)
                if row['r1'].strip() != '':
                    utt_ = Utterance(session_id, row['datetime'], row['user'], 'bot_to_sb' , row['r1'], row['intent'], **row)
                    dialog_.utterances.append(utt_)

                session_dict[session_id] = dialog_

        for dialog_id, dialog in session_dict.items():
            self.session_list.append(dialog.utterances)

class Maluuba(DataLoader):
    def __init__(self, **kwargs):
        super(Maluuba, self).__init__(**kwargs)
        self.data_path = self.root_dir + '/dataset/Maluuba/frames.json'
    def load_(self):
        with open(self.data_path, 'r') as f_:
            data = json.load(f_)
            for session_dict in data:
                session = []
                for utterance in session_dict['turns']:
                    if utterance['author'] == 'user':
                        direction_ = 'user_to_sb'
                    else:
                        direction_ = 'bot_to_sb'

                    session.append(Utterance(session_dict['id'], utterance['timestamp'], session_dict['user_id'], direction_, utterance['text']))

                self.session_list.append(session)

class MATCH(DataLoader):
    def __init__(self, **kwargs):
        super(MATCH, self).__init__(**kwargs)
        self.data_path = self.root_dir + '/dataset/MATCHcorpus/'

    class Dialogue():
        def __init__(self, dialogue_id):
            self.session_id = dialogue_id
            self.utterances = []

        def __len__(self):
            return len(self.utterances)

        def __iter__(self):
            return iter(self.utterances)


    def load_(self):
        turn_dir = self.root_dir + '/dataset/MATCHcorpus/turns/'
        transcript_path = self.root_dir + '/dataset/MATCHcorpus/transcript/'

        dialogue_list = []

        for turn_file in os.listdir(turn_dir):
            self.logger.info(os.path.join(turn_dir, turn_file))

            dialogue_id = turn_file[:turn_file.find('.turns.xml')]
            dialogue = self.Dialogue(dialogue_id)
            dialogue_list.append(dialogue)

            with open(os.path.join(turn_dir, turn_file), 'r') as fp:
                turn_soup = BeautifulSoup(fp)

            # load words, starts from index=1
            words = {}
            with open(os.path.join(transcript_path, dialogue_id + '.P.words.xml'), 'r') as fp:
                word_P_soup = BeautifulSoup(fp)
                for w in word_P_soup.find_all('w'):
                    w_id = w['nite:id']
                    words[w_id] = w.text

            with open(os.path.join(transcript_path, dialogue_id + '.W.words.xml'), 'r') as fp:
                word_W_soup = BeautifulSoup(fp)
                for w in word_W_soup.find_all('w'):
                    w_id = w['nite:id']
                    words[w_id] = w.text

            # load segments
            segments = {}
            with open(os.path.join(transcript_path, dialogue_id + '.P.segments.xml'), 'r') as fp:
                segment_P_soup = BeautifulSoup(fp)
                for s in segment_P_soup.find_all('segment'):
                    segment_id = s['nite:id']

                    if len(s.find_all('nite:child')) > 0:
                        word_href = s.find_all('nite:child')[0]['href']
                        word_ids  = re.findall('user\.word\.(\d+)',word_href)
                        if len(word_ids) == 1:
                            start_id = int(word_ids[0])
                            end_id = start_id + 1
                        else:
                            start_id = int(word_ids[0])
                            end_id = int(word_ids[1]) + 1
                        segments[segment_id] = ' '.join([words[dialogue_id+'.user.word.'+str(i)] for i in range(start_id, end_id) if dialogue_id+'.user.word.'+str(i) in words])
                    else:
                        segments[segment_id] = ' '

                    # self.logger.info('%s - %s' % (segment_id, P_segments[segment_id]))

            with open(os.path.join(transcript_path, dialogue_id + '.W.segments.xml'), 'r') as fp:
                segment_W_soup = BeautifulSoup(fp)
                for s in segment_W_soup.find_all('segment'):
                    segment_id = s['nite:id']

                    if len(s.find_all('nite:child')) > 0:
                        word_href = s.find_all('nite:child')[0]['href']
                        word_ids  = re.findall('system\.word\.(\d+)',word_href)
                        if len(word_ids) == 1:
                            start_id = int(word_ids[0])
                            end_id = start_id + 1
                        else:
                            start_id = int(word_ids[0])
                            end_id = int(word_ids[1]) + 1
                        segments[segment_id] = ' '.join([words[dialogue_id+'.system.word.'+str(i)] for i in range(start_id, end_id) if dialogue_id+'.system.word.'+str(i) in words])
                    else:
                        segments[segment_id] = ' '

                    # self.logger.info('%s - %s' % (segment_id, W_segments[segment_id]))

            for u_id, turn in enumerate(turn_soup.find_all('turn')):
                turn_id = turn['nite:id']
                if turn['speaker'] == 'system':
                    direction = 'bot_to_sb'
                elif turn['speaker'] == 'user':
                    direction = 'user_to_sb'
                else:
                    raise ('WHAT\'S THIS DERECTION?')

                # self.logger.info('*' * 30)
                # self.logger.info(turn_id)

                for child in turn.find_all('nite:child'):
                    child_href = child['href']
                    url = os.path.join(transcript_path, child_href)
                    segment_id = re.search('id\((.*?)\)', url)
                    # self.logger.info(segments[segment_id.group(1)])

                    u_ = Utterance(dialogue_id , u_id, '', direction, segments[segment_id.group(1)], '')
                    dialogue.utterances.append(u_)

        self.session_list = dialogue_list

# aliases
family = Family_Assistant
weather = Yahoo_Weather
dstc1 = DSTC1
dstc2 = DSTC2
dstc3 = DSTC3
maluuba = Maluuba
match = MATCH
ghome = GHOME

DATA_NAME = 'ghome'

if __name__ == '__main__':
    loader = data_loader(DATA_NAME)
    loader()
    loader.stats()

    if DATA_NAME == 'family' or DATA_NAME == 'dstc1':
        loader.logger.info('filtering the family dataset by removing all the invalid dialogues (short or on-boarding)')
        loader.filter_invalid_sessions()

    if DATA_NAME == 'ghome':
        loader.logger.info('*'*20 + 'Filtering short sessions and Location info' + '*'*20)
        loader.session_list = list(filter(lambda x:len(x) > 4, loader.session_list))

        def count_location_info(session):
            count = 0
            for utt_number, utt in enumerate(session):
                if utt.msg_text.strip() == 'Location info':
                    count += 1
            return count

        def count_system_response(session):
            count = 0
            for utt_number, utt in enumerate(session):
                if utt.direction == 'bot_to_sb':
                    count += 1
            return count
        # filter out sessions having more than 2 'Location info'
        loader.session_list = list(filter(lambda x:count_location_info(x) < 3, loader.session_list))
        # filter out sessions having less than 3 System Responses
        loader.session_list = list(filter(lambda x:count_system_response(x) >= 2, loader.session_list))

    loader.stats()
    loader.export_ramdom_samples(1000)