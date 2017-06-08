import json
import pickle

import os
import re
from dialogue.analysis.parse_session import export_ramdom_samples

class Utterance():
    def __init__(self, time, userid, direction, text):
        self.time = time
        self.useruuid = userid
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

class Dialogue():
    def __init__(self, path):
        with open(path, 'r') as json_:
            self.json = json.load(json_)
            self.path = path
            self.session_id = self.json['session-id']

            # print(self.session_id + '*' * 20)
            self.utterances = []
            for turn_id, turn in enumerate(self.json['turns']):
                if 'transcript' in turn['output']:
                    self.utterances.append(Utterance(turn_id, '', 'bot_to_sb', turn['output']['transcript']))
                else:
                    self.utterances.append(Utterance(turn_id, '', 'bot_to_sb', ''))
                # print('\t\t bot: %s' % self.utterances[-1].msg_text)

                if turn['input']['live']['asr-hyps'] == []:
                    self.utterances.append(Utterance(turn_id, '', 'user_to_sb', ''))
                else:
                    self.utterances.append(Utterance(turn_id, '', 'user_to_sb', turn['input']['live']['asr-hyps'][0]['asr-hyp']))
                # print('\t\t user:  %s' % self.utterances[-1].msg_text)
    def __len__(self):
        return len(self.utterances)

    def __iter__(self):
        return iter(self.utterances)


def load_data(DATA_NAME, data_dir):
    dialogue_list = []

    if os.path.exists(os.path.join(data_dir, DATA_NAME+'.pkl')):
        with open(os.path.join(data_dir, DATA_NAME+'.pkl'), 'rb') as f_:
            dialogue_list = pickle.load(f_)
        print('Loading %s cache complete' % DATA_NAME)
    else:
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
                    elif DATA_NAME == 'DSTC2':
                        file_path = os.path.abspath(os.path.join(data_dir, first_dir, second_dir, third_dir, 'log.json'))

                    if os.path.exists(file_path):
                        dialogue_list.append(Dialogue(file_path))

        with open(os.path.join(data_dir, DATA_NAME+'.pkl'), 'wb') as f_:
            pickle.dump(dialogue_list, f_)

    return dialogue_list


def basic_statistics(session_list):
    session_count = len(session_list)
    utterance_count = sum([len(session.utterances) for session in session_list])

    print('session_count = %d' % session_count)
    print('utterance_count = %d' % (utterance_count))
    if session_count > 0:
        average_session_length = float(utterance_count)/float(session_count)
    else:
        average_session_length = 0
    print('average_session_length = %.5f' % (average_session_length))

def JaccardDistance(str1, str2):
    str1 = set(re.split('\W+', str1.lower()))
    str2 = set(re.split('\W+', str2.lower()))
    if len(str1 | str2) > 0:
        return float(len(str1 & str2)) / len(str1 | str2)
    return 0.0

def find_repetition_session(session_list, SIMILARITY_THRESHOLD = 0.8):
    new_sessions = []

    for session in session_list:
        max_jaccard = 0
        last_user_utterance = None
        last_user_utterance_id = None
        # most_similar_pairs = []
        most_similar_idxs = []

        for utterance_id, utt in enumerate(session.utterances):
            if utt.direction == 'user_to_sb':
                if utt.msg_text == None:
                    last_user_utterance = None
                    last_user_utterance_id = None
                    continue

                # ignore the utterances that length is less than 5 (simple commands like Skip)
                if last_user_utterance == None or len(set(re.split('\W+', last_user_utterance.msg_text))) <= 2 or len(set(re.split('\W+', utt.msg_text))) <= 2:
                # if last_user_utterance == None:
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
                str1 = set(re.split('\W+', session.utterances[most_similar_idx[0]].msg_text.lower()))
                str2 = set(re.split('\W+', session.utterances[most_similar_idx[1]].msg_text.lower()))
                similar_idx_dict0[most_similar_idx[0]] = m_id
                similar_idx_dict1[most_similar_idx[1]] = m_id

                print('Jaccard similarity:\t%f' % max_jaccard)
                print('String 1: id = %d \t contect = %s' % (most_similar_idx[0], str1))
                print('String 2: id = %d \t contect = %s' % (most_similar_idx[1], str2))
                print('Intersection:\t%s' % (str1 & str2))
                print('Union:\t\t\t%s\n' % (str1 | str2))

            for u_id, u_ in enumerate(session.utterances):
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

    return new_sessions




DATA_INDEX = 1
DATA_NAMES = ['DSTC1', 'DSTC2']
DATA_NAME  = DATA_NAMES[DATA_INDEX]

root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir+os.sep+os.pardir))
DSTC1_PATH = root_dir + '/dataset/DSTC1/'
DSTC2_PATH = root_dir + '/dataset/DSTC2/'
PATHS = [DSTC1_PATH, DSTC2_PATH]
data_dir  = PATHS[DATA_INDEX]

print(DATA_NAME + ' : ' + data_dir)


if __name__ == '__main__':
    dialogue_list = load_data(DATA_NAME, data_dir)

    print('%' * 20 + 'RAW Data' + '%' * 20)
    basic_statistics(dialogue_list)

    # high_repetition_dialogues = find_repetition_session(dialogue_list)
    # print('%' * 20 + 'Data after Jaccard Filtering' + '%' * 20)
    # basic_statistics(high_repetition_dialogues)

    export_ramdom_samples(dialogue_list, DATA_NAME, N=100)