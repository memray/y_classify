# -*- coding: utf-8 -*-
import csv
import json
import pickle
import random

import numpy
import six
import os
import re

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


class Utterance():
    def __init__(self, session_id , time, userid, direction, text, botlog=None):
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

class Dialogue():
    def __init__(self, path, dialogue_id):
        with open(path, 'r') as json_:
            self.json = json.load(json_)
            self.path = path
            self.session_id = self.json['session-id']

            # print(self.session_id + '*' * 20)
            self.utterances = []
            for turn_id, turn in enumerate(self.json['turns']):
                if 'transcript' in turn['output']:
                    self.utterances.append(Utterance(dialogue_id, turn_id, '', 'bot_to_sb', turn['output']['transcript']))
                else:
                    self.utterances.append(Utterance(dialogue_id, turn_id, '', 'bot_to_sb', ''))
                # print('\t\t bot: %s' % self.utterances[-1].msg_text)

                if turn['input']['live']['asr-hyps'] == []:
                    self.utterances.append(Utterance(dialogue_id, turn_id, '', 'user_to_sb', ''))
                else:
                    self.utterances.append(Utterance(dialogue_id, turn_id, '', 'user_to_sb', turn['input']['live']['asr-hyps'][0]['asr-hyp']))
                    # print('\t\t user:  %s' % self.utterances[-1].msg_text)

    def __len__(self):
        return len(self.utterances)

    def __iter__(self):
        return iter(self.utterances)

class DataLoader(object):
    def __init__(self, **kwargs):
        self.root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))
        self.__dict__.update(kwargs)
        self.name    = self.__class__.__name__
        self.session_list    = []

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
        if self.__class__.__name__ == 'Maluuba':
            self.load_()
            return self.session_list
        if self.__class__.__name__ == 'MATCH':
            self.load_()
            return self.session_list

        raise 'Not Recognized Dataset Name'

    def load_pig_log(self):
        '''
        Load family assistant and weather data
        '''
        print(self.name + ' : ' + self.data_path)
        session_list = []
        session_count = 0

        with open(self.data_path, 'r') as f_:
            for session_line in f_.readlines():
                delimeter_idx = session_line.find('\t')
                user_id = session_line[:delimeter_idx]
                session_content = session_line[delimeter_idx+1:]

                if (session_content == None or session_content.strip() == ''):
                    continue

                session = []
                utterance_list = json.loads(session_content)[0]

                for record in utterance_list:
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
            print('Loading %s cache complete' % DATA_NAME)
        else:
            dialogue_count = 0
            for first_dir in os.listdir(data_dir):
                if os.path.isfile(data_dir+first_dir):
                    continue
                for second_dir in os.listdir(data_dir+first_dir):
                    if os.path.isfile(os.path.join(data_dir, first_dir, second_dir)):
                        continue
                    print(os.path.join(data_dir, first_dir, second_dir))
                    for third_dir in os.listdir(os.path.join(data_dir, first_dir, second_dir)):
                        if DATA_NAME == 'DSTC1':
                            file_path = os.path.abspath(os.path.join(data_dir, first_dir, second_dir, third_dir, 'dstc.log.json'))
                        elif DATA_NAME == 'DSTC2' or DATA_NAME == 'DSTC3':
                            file_path = os.path.abspath(os.path.join(data_dir, first_dir, second_dir, third_dir, 'log.json'))

                        if os.path.exists(file_path):
                            dialogue_count += 1
                            dialogue_list.append(self.Dialogue(file_path, dialogue_count))

            with open(os.path.join(data_dir, DATA_NAME+'.pkl'), 'wb') as f_:
                pickle.dump(dialogue_list, f_)

        self.session_list = dialogue_list

    def export_ramdom_samples(self, N=100):
        '''
        Randomly sample N data points
        :return:
        '''
        # 1. convert to numpy array
        session_list = numpy.asarray(self.session_list)
        print('Exporting samples to CSV file')
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
        print(sample_path)
        valid_count = 0

        with open(sample_path, 'w') as csvfile:
            # attrs = [attr for attr in dir(session_list[0][0]) if not attr.startswith('__')]
            # print(attrs)
            # attrs = ['useruuid', 'time', 'direction', 'msg', 'botlog']
            attrs = ['session_id', 'userid', 'time', 'direction', 'msg_text']

            csvfile = csv.writer(csvfile)
            # csvfile = csv.writer(csvfile, delimiter='\t', lineterminator='\n')
            csvfile.writerow(attrs)

            for session_num, session in enumerate(session_list):
                for u_ in session:
                    csvfile.writerow([getattr(u_, attr) for attr in attrs])
                csvfile.writerow([])

                valid_count += 1

        print('#(valid samples) = %d' % valid_count)

class Family_Assistant(DataLoader):
    def __init__(self, **kwargs):
        super(Family_Assistant, self).__init__(**kwargs)
        self.data_path = self.root_dir + '/dataset/Family_Assistant.20170307.haslengthfilter.interval=5min.session/part-v002-o000-r-00000'
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
        self.data_path = self.root_dir + '/dataset/Weather.interval=5min.session/part-v002-o000-r-00000'
        self.name = self.__class__.__name__

class DSTC1(DataLoader):
    def __init__(self, **kwargs):
        super(DSTC1, self).__init__(**kwargs)
        self.data_path = self.root_dir + '/dataset/DSTC1/'

class DSTC2(DataLoader):
    def __init__(self, **kwargs):
        super(DSTC2, self).__init__(**kwargs)
        self.data_path = self.root_dir + '/dataset/DSTC2/'


class DSTC3(DataLoader):
    def __init__(self, **kwargs):
        super(DSTC3, self).__init__(**kwargs)
        self.data_path = self.root_dir + '/dataset/DSTC3_test/'

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
            print(os.path.join(turn_dir, turn_file))

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

                    # print('%s - %s' % (segment_id, P_segments[segment_id]))

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

                    # print('%s - %s' % (segment_id, W_segments[segment_id]))

            for u_id, turn in enumerate(turn_soup.find_all('turn')):
                turn_id = turn['nite:id']
                if turn['speaker'] == 'system':
                    direction = 'bot_to_sb'
                elif turn['speaker'] == 'user':
                    direction = 'user_to_sb'
                else:
                    raise ('WHAT\'S THIS DERECTION?')

                # print('*' * 30)
                # print(turn_id)

                for child in turn.find_all('nite:child'):
                    child_href = child['href']
                    url = os.path.join(transcript_path, child_href)
                    segment_id = re.search('id\((.*?)\)', url)
                    # print(segments[segment_id.group(1)])

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

DATA_NAME = 'family'
if __name__ == '__main__':
    loader = data_loader(DATA_NAME)
    loader()
    print(len(loader.session_list))

    if DATA_NAME == 'family':
        loader.filter_invalid_sessions()
        print(len(loader.session_list))

    loader.export_ramdom_samples(100)