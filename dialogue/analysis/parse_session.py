# -*- coding: utf-8 -*-
import csv
import json
import pickle
import random

import numpy
import os, re

class Utterance():
    def __init__(self, record):
        for k,v in record.items():
            setattr(self, k, v)

        # self.time = record[0]
        # self.useruuid = record[1]
        # self.direction = record[2]
        # # self.platform = record[3]
        # # self.msg_sentto = record[4]
        # self.msg_types = record[5]
        # # self.msg_sentto_displayname = record[6]
        # # self.dt_day = record[7]
        # # self.ts_in_second = record[8]
        # # self.platform_message_id = record[9]
        # # self.botlog_intent = record[10]
        # # self.botlog_slots = record[11]
        # self.msg_text  = record[12]

    def __str__(self):
        str_ = ''
        # for attr in dir(self):
        #     if not attr.startswith('__') and getattr(self, attr) != None and getattr(self, attr)!='':
                # str_ += '\t\t%s : %s\n' % (attr, getattr(self, attr))
        str_ += '\t\t%s : %s\n' % ('useruuid', getattr(self, 'useruuid'))
        str_ += '\t\t%s : %s\n' % ('time', getattr(self, 'time'))
        str_ += '\t\t%s : %s\n' % ('direction', getattr(self, 'direction'))
        str_ += '\t\t%s : %s\n' % ('msg_text', getattr(self, 'msg_text'))
        return str_

def str_to_session(session_content):
    '''
    parse string to a session list
    :param session_content:
    '''
    session = []
    utterance_list = json.loads(session_content)[0]

    for record in utterance_list:
        u_ = Utterance(record)
        session.append(u_)

    return session

# Session number distribution
def session_number_distribution(session_dict):
    session_count = {}
    for k, v in session_dict.items():
        session_count[len(v)] = session_count.get(len(v), 0) + 1
    sorted_count = sorted(session_count.items(), key=lambda k:k[0])
    print('Session Number Distribution')
    for n_session, amount in sorted_count:
        print('%d\t%d' % (n_session, amount))

# check the most active users
def most_active_user(session_dict):
    user_session_count = {}
    for k, v in session_dict.items():
        user_session_count[k] = len(v)
    user_session_count = sorted(user_session_count.items(), key=lambda k:k[1], reverse=True)
    print('The most active users')
    for user_id, amount in user_session_count[:10]:
        print('%s\t%d' % (user_id, amount))

# check session length distribution
def session_length_distribution(session_dict):
    session_length_count = {}
    for user_id, sessions in session_dict.items():
        for session in sessions:
            # print(len(session))
            session_length_count[len(session)] = session_length_count.get(len(session), 0) + 1
    print('Session Length Distribution')
    sorted_count = sorted(session_length_count.items(), key=lambda k:k[0])
    for n_utterance, amount in sorted_count:
        print('%s\t%d' % (n_utterance, amount))


def print_session_at_length_K(session_dict, K = 2, equal = False):
    '''
    print sessions whose length is K
    '''
    print('Printing sessions of length %d' % K)
    for user_id, sessions in session_dict.items():
        for session in sessions:
            if (equal and len(session) == K) or (not equal and len(session) >= K):
                print('-'*20 + user_id + '-'*20)
                for u in session:
                    print(u)

def is_valid_session(session):
    is_valid = True
    min_session_length = 4 # the minimum length

    # 1. filter the sessions of which length is less than 4
    if len(session) < min_session_length:
        is_valid = False

    # 2. filter the sessions which have less 2 user utterances
    number_user_message = 0
    for u in session:
        if u.direction == 'user_to_sb':
            number_user_message += 1
            # 3. filter the sessions which have user messages msg_text==None
            if u.msg_text == None or u.msg_text == '':
                is_valid = False
        # 4. not a on-board session (this is not useful, too broad about the 'onboard')
        # else:
        #     if u.botlog != None and u.botlog.find('onboard') != -1:
        #         is_valid = False
        # key: user_case value: onboarding

    if number_user_message < 3:
        is_valid = False

    return is_valid

def filter_invalid_session(session_dict):
    new_session_dict = {}

    for user_id, sessions in session_dict.items():
        new_sessions = []
        for session in sessions:
            if is_valid_session(session):
                new_sessions.append(session)
        if len(new_sessions) > 0:
            new_session_dict[user_id] = new_sessions

    return new_session_dict


def basic_statistics(session_dict):
    user_count = len(session_dict)
    session_count = sum([len(s) for s in session_dict.values()])
    user_utterance_count = sum([sum([len([u for u in s if u.direction=='user_to_sb']) for s in sl]) for sl in session_dict.values()])
    system_utterance_count = sum([sum([len([u for u in s if u.direction=='bot_to_sb']) for s in sl]) for sl in session_dict.values()])

    print('user_count = %d' % user_count)
    print('session_count = %d' % session_count)
    print('utterance_count = %d' % (user_utterance_count+system_utterance_count))
    print('user_utterance_count = %d' % user_utterance_count)
    print('system_utterance_count = %d' % system_utterance_count)
    if session_count > 0:
        average_session_length = float(system_utterance_count)/float(session_count)
    else:
        average_session_length = 0
    print('average_session_length = %.5f' % (average_session_length))

def JaccardDistance(str1, str2):
    str1 = set(re.split('\W+', str1.lower()))
    str2 = set(re.split('\W+', str2.lower()))
    if len(str1 | str2) > 0:
        return float(len(str1 & str2)) / len(str1 | str2)
    return 0.0

def find_repetition_session(session_dict, SIMILARITY_THRESHOLD = 0.8):
    new_session_dict = {}

    for user_id, sessions in session_dict.items():
        new_sessions = []
        for session in sessions:
            max_jaccard = 0
            last_user_utterance = None
            last_user_utterance_id = None
            # most_similar_pairs = []
            most_similar_idxs = []

            for utterance_id, utt in enumerate(session):
                if utt.direction == 'user_to_sb':
                    if utt.msg_text == None:
                        last_user_utterance = None
                        last_user_utterance_id = None
                        continue

                    # ignore the utterances that length is less than 5 (simple commands like Skip)
                    if last_user_utterance == None or len(set(re.split('\W+', last_user_utterance.msg_text))) <= 5 or len(set(re.split('\W+', utt.msg_text))) <= 5:
                        last_user_utterance = utt
                        last_user_utterance_id = utterance_id
                        continue

                    else:
                        current_jaccard = JaccardDistance(last_user_utterance.msg_text, utt.msg_text)
                        if current_jaccard > max_jaccard:
                            max_jaccard = current_jaccard
                            # most_similar_pairs.append((last_user_utterance, utt))
                            most_similar_idxs.append((last_user_utterance_id, utterance_id))

                        last_user_utterance = utt
                        last_user_utterance_id = utterance_id

            if max_jaccard >= SIMILARITY_THRESHOLD:
                new_sessions.append(session)
                '''
                similar_idx_dict0 = {}
                similar_idx_dict1 = {}
                print("================== Find similar pair! ==================")
                print('Session length:\t%d' % len(session))
                for m_id, most_similar_idx in enumerate(most_similar_idxs):
                    str1 = set(re.split('\W+', session[most_similar_idx[0]].msg_text.lower()))
                    str2 = set(re.split('\W+', session[most_similar_idx[1]].msg_text.lower()))
                    similar_idx_dict0[most_similar_idx[0]] = m_id
                    similar_idx_dict1[most_similar_idx[1]] = m_id

                    print('Jaccard similarity:\t%f' % max_jaccard)
                    print('String 1: id = %d \t contect = %s' % (most_similar_idx[0], str1))
                    print('String 2: id = %d \t contect = %s' % (most_similar_idx[1], str2))
                    print('Intersection:\t%s' % (str1 & str2))
                    print('Union:\t\t\t%s\n' % (str1 | str2))

                for u_id, u_ in enumerate(session):
                    if u_id in similar_idx_dict0:
                        print('\t\t' + '-' * 25 + ' START - HEAD %d ' % similar_idx_dict0[u_id] + '-' * 25)
                    if u_id in similar_idx_dict1:
                        print('\t\t' + '-' * 25 + ' START - TAIL %d ' % similar_idx_dict1[u_id] + '-' * 25)
                    print(u_)
                    if u_id in similar_idx_dict0:
                        print('\t\t' + '-' * 25 + ' END - HEAD %d ' % similar_idx_dict0[u_id] + '-' * 25)
                    if u_id in similar_idx_dict1:
                        print('\t\t' + '-' * 25 + ' END - TAIL %d ' % similar_idx_dict1[u_id] + '-' * 25)
                # '''

        if len(new_sessions) > 0:
            new_session_dict[user_id] = new_sessions
    return new_session_dict


def filter_on_board_utterances(BOT_NAME, session):
    new_session = []
    for u in session:
        if BOT_NAME == 'Family_Assistant':
            if u.msg_text.startswith('[Hi') or u.msg_text.startswith('["Cool! Let\'s start with your shopping list') or u.msg_text.startswith('["Ok! Let\'s start with grocery list') or u.msg_text.startswith('["Cool! I\'ll introduce shopping list') or u.msg_text.startswith('["Cool! I\'ll introduce shopping list') or u.msg_text.startswith('["Great to meet you'):
                new_session = []
        new_session.append(u)
    return new_session


def is_post_valid(BOT_NAME, session):
    is_valid = True

    user_count = 0
    nontrivial_count = 0
    for u in session:
        if u.direction == 'user_to_sb':
            user_count += 1

            if BOT_NAME == 'Family_Assistant':
                t = u.msg_text.lower()
                # check appearance of complete commands
                if not t.startswith('"skip"') and not t.startswith('"help"') and not t.startswith('"show') and not t.startswith('"remove"'): # and not t.startswith('"add"')
                    nontrivial_count += 1

        if u.direction == 'bot_to_sb':
            if BOT_NAME == 'Family_Assistant':
                # check if it contains on-boarding utterances
                if u.msg_text.find('your family assistant') != -1 or u.msg_text.find('get started') != -1 or u.msg_text.startswith('["Cool! Let\'s start with') or u.msg_text.startswith('["Ok! Let\'s start with') or u.msg_text.startswith('["Cool! I\'ll introduce shopping list') or u.msg_text.startswith('["Cool! I\'ll introduce shopping list') or u.msg_text.startswith('["Great to meet you'):
                    is_valid = False
                if u.msg_text.find('start with grocery') != -1 or u.msg_text.find('My verification code') != -1:
                    is_valid = False

    if nontrivial_count < 2:
        is_valid = is_valid and False

    if len(session) <= 4 or user_count <= 2:
        is_valid = is_valid and False
    else:
        is_valid = is_valid and True

    return is_valid


def find_nontrivial_session(session_dict):
    '''
    filter on_board utterance and trivial sessions
    :param session_dict:
    :return:
    '''
    new_session_dict = {}

    for user_id, sessions in session_dict.items():
        new_sessions = []
        for session in sessions:
            # session = filter_on_board_utterances(BOT_NAME, session)

            if not is_post_valid(BOT_NAME, session):
                continue

            new_sessions.append(session)

        if len(new_sessions) > 0:
            new_session_dict[user_id] = new_sessions

    return new_session_dict

def export_ramdom_samples(session_list, BOT_NAME, N=100):
    '''
    Randomly sample N data points
    :return:
    '''
    # 1. convert to numpy array
    session_list = numpy.asarray(session_list)
    print('Exporting samples to CSV file')
    # 2. load cache if exists, otherwise resample it, if N==1 means to export all
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))

    if N != 1:
        sample_cache_path = root_dir + '/dataset/sample/%s.N=%s.pkl' % (BOT_NAME, N)
        if os.path.exists(sample_cache_path):
            with open(sample_cache_path, 'rb') as f_:
                sample_index = pickle.load(f_)
        else:
            sample_index = random.sample(range(len(session_list)), N)
            with open(sample_cache_path, 'wb') as f_:
                pickle.dump(sample_index, f_)
        session_list = session_list[sample_index]

    # 3. dump the samples to a CSV file
    sample_path = root_dir + '/dataset/sample/%s.N=%s.sampled.csv' % (BOT_NAME, N)
    valid_count = 0

    with open(sample_path, 'w') as csvfile:
        # attrs = [attr for attr in dir(session_list[0][0]) if not attr.startswith('__')]
        # print(attrs)
        # attrs = ['useruuid', 'time', 'direction', 'msg', 'botlog']
        attrs = ['useruuid', 'time', 'direction', 'msg_text']

        csvfile = csv.writer(csvfile)
        csvfile.writerow(attrs)

        for session in session_list:
            for u_ in session:
                csvfile.writerow([getattr(u_, attr) for attr in attrs])
            csvfile.writerow([])

            valid_count += 1

    print('#(valid samples) = %d' % valid_count)

BOT_INDEX = 0
BOT_NAMES = ['Family_Assistant', 'Monkey_Pets', 'Weather']
BOT_NAME  = BOT_NAMES[BOT_INDEX]

root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))
FAMILY_PATH = root_dir + '/dataset/Family_Assistant.interval=5min.session/part-v002-o000-r-00000'
MONKEY_PATH = root_dir + '/dataset/Monkey_Pets.interval=5min.session/part-v002-o000-r-00000'
WEATHER_PATH = root_dir + '/dataset/Weather.interval=5min.session/part-v002-o000-r-00000'
PATHS = [FAMILY_PATH, MONKEY_PATH, WEATHER_PATH]
file_dir  = PATHS[BOT_INDEX]

print(BOT_NAME + ' : ' + file_dir)

if __name__ == '__main__':

    session_dict = {}

    with open(file_dir, 'r') as f_:
        for session_line in f_.readlines():
            delimeter_idx = session_line.find('\t')
            user_id = session_line[:delimeter_idx]
            session_content = session_line[delimeter_idx+1:]

            session_list = session_dict.get(user_id, [])

            if (session_content == None or session_content.strip() == ''):
                continue

            new_session = str_to_session(session_content)
            if len(new_session) > 4:
                session_list.append(new_session)
            session_dict[user_id] = session_list

    print('%' * 20 + 'RAW Data' + '%' * 20)
    basic_statistics(session_dict)

    # filter the sessions that have only one direction (not a dialogue)
    session_dict = filter_invalid_session(session_dict)
    print('%' * 20 + 'Valid Data' + '%' * 20)
    basic_statistics(session_dict)
    # session_length_distribution(session_dict)

    # session_number_distribution(session_dict)
    # most_active_user(session_dict)
    # session_length_distribution(session_dict)


    session_dict = find_nontrivial_session(session_dict)
    print('%' * 20 + 'Data after filtering trivia' + '%' * 20)
    basic_statistics(session_dict)
    # session_length_distribution(nontrivial_session)

    # high_repetition_session_dict = find_repetition_session(nontrivial_session)
    # print('%' * 20 + 'Data after Jaccard Filtering' + '%' * 20)
    # basic_statistics(high_repetition_session_dict)
    # session_length_distribution(high_repetition_session_dict)

    # basic_statistics(high_repetition_session_dict)
    # print_session_at_length_K(valid_session_dict, K=4, equal=True)

    session_list = []
    # flatten the sessions
    for sessions in session_dict.values():
        session_list.extend(sessions)

    export_ramdom_samples(session_list, BOT_NAME, N=1)