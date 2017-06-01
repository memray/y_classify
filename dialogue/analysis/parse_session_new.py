# -*- coding: utf-8 -*-
import csv
import json

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
        for attr in dir(self):
            if not attr.startswith('__') and getattr(self, attr) != None and getattr(self, attr)!='':
                str_ += '\t\t%s : %s\n' % (attr, getattr(self, attr))
        return str_

def str_to_session(session_content):
    '''
    parse string to a session list
    :param session_content:
    '''
    session = []
    utterance_list = json.loads(session_content)

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
    for user_id, session in session_dict.items():
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

def filter_invalid_session(session_dict):
    min_session_length = 4 # the minimum length
    new_session_dict = {}

    for user_id, sessions in session_dict.items():
        new_sessions = []
        for session in sessions:
            # filter by session
            if len(session) < min_session_length:
                continue

            # filter by direction (both directions must occur)
            dir_dict = set()
            for utt in session:
                if utt.direction!=None and utt.direction!='':
                    dir_dict.add(utt.direction)
            if len(dir_dict) <= 1:
                continue

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
    print('average_session_length = %.5f' % (float(system_utterance_count)/float(session_count)))

def JaccardDistance(str1, str2):
    str1 = set(re.split('\W+', str1.lower()))
    str2 = set(re.split('\W+', str2.lower()))
    if len(str1 | str2) > 0:
        return float(len(str1 & str2)) / len(str1 | str2)
    return 0.0

def find_repetition_session(session_dict):
    new_session_dict = {}
    SIMILARITY_THRESHOLD = 0.5

    for user_id, sessions in session_dict.items():
        new_sessions = []
        for session in sessions:
            max_jaccard = 0
            last_user_utterance = None
            last_user_utterance_id = None
            most_similar_pair = None
            most_similar_ids = None

            for utterance_id, utt in enumerate(session):
                if utt.direction == 'user_to_sb':
                    # ignore the utterances that length is less than 3 (simple commands like Skip)
                    if len(re.split('\W+', utt.msg_text.lower())) < 3:
                        continue

                    if last_user_utterance == None:
                        last_user_utterance = utt
                        last_user_utterance_id = utterance_id
                        continue
                    else:
                        current_jaccard = JaccardDistance(last_user_utterance.msg_text, session[utterance_id].msg_text)
                        if current_jaccard > max_jaccard:
                            max_jaccard = current_jaccard
                            most_similar_pair = (last_user_utterance, session[utterance_id])
                            most_similar_ids = (last_user_utterance_id, utterance_id)

                        last_user_utterance = session[utterance_id]
                        last_user_utterance_id = utterance_id

            if max_jaccard >= SIMILARITY_THRESHOLD:
                new_sessions.append(session)

                print("================== Find similar pair! ==================")
                str1 = set(re.split('\W+', most_similar_pair[0].msg_text.lower()))
                str2 = set(re.split('\W+', most_similar_pair[1].msg_text.lower()))

                print('Session length:\t%d' % len(session))
                print('Jaccard similarity:\t%f' % max_jaccard)
                print('Intersection:\t%s' % (str1 & str2))
                print('Union:\t\t\t%s' % (str1 | str2))

                for u_id, u_ in enumerate(session):
                    if u_id in most_similar_ids:
                        print('\t\t' + '-' * 25 + ' START '+ '-' * 25)
                        print(str(u_) + '\t\t' + '-' * 25 + ' END '+ '-' * 25)
                    else:
                        print(u_)

        if len(new_sessions) > 0:
            new_session_dict[user_id] = new_sessions
    return new_session_dict

root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))
FAMILY_PATH = root_dir + '/dataset/Family_Assistant.interval=5min.session/part-v002-o000-r-00000'
WEATHER_PATH = root_dir + '/dataset/Weather.interval=5min.session/part-v002-o000-r-00000'
MONKEY_PATH = root_dir + '/dataset/Monkey_Pets.interval=5min.session/part-v002-o000-r-00000'

file_dir = WEATHER_PATH

if __name__ == '__main__':

    session_dict = {}

    with open(file_dir, 'r') as f_:
        for session_line in f_.readlines():
            delimeter_idx = session_line.find('\t')
            user_id = session_line[:delimeter_idx]
            session_content = session_line[delimeter_idx:]

            session_list = session_dict.get(user_id, [])
            session_list.append(str_to_session(session_content))
            session_dict[user_id] = session_list

    print('%' * 20 + 'RAW Data' + '%' * 20)
    basic_statistics(session_dict)
    # filter the sessions that have only one direction (not a dialogue)
    valid_session_dict = filter_invalid_session(session_dict)

    print('%' * 20 + 'Valid Data' + '%' * 20)
    basic_statistics(valid_session_dict)

    # session_number_distribution(session_dict)
    # most_active_user(session_dict)
    # session_length_distribution(session_dict)

    high_repetition_session_dict = find_repetition_session(valid_session_dict)

    print('%' * 20 + 'Data after Jaccard Filtering' + '%' * 20)
    basic_statistics(high_repetition_session_dict)
    # print_session_at_length_K(valid_session_dict, K=4, equal=True)