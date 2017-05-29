from __future__ import division
import string
import sys
import math
import re

# -*- coding: utf-8 -*-

@outputSchema('session:bag{t:('
              'time:                    chararray, '
              'useruuid:                chararray, '
              'direction:               chararray, '
              'platform:                chararray, '
              'msg_sentto:              chararray,'
              'msg_types:               chararray, '
              'msg_sentto_displayname:  chararray, '
              'dt_day:                  chararray, '
              'ts_in_second:            int, '
              'platform_message_id:     chararray, '
              'botlog_intent:           chararray, '
              'botlog_slots:            chararray, '
              'msg_text:                chararray'
              ')}')

def split_session(user_utterances, max_session_interval):
    '''
    Given a handful of utterances of one user, segment the utterances into sessions
    :return: a list of sessions, each session consists of a few utterances
    '''
    session_list = []
    current_session = []
    last_utterance_time = None

    for utterance in user_utterances:
        (time, useruuid, direction, platform, msg_sentto, msg_types, msg_sentto_displayname, dt_day, ts_in_second, platform_message_id, botlog_intent, botlog_slots, msg_text) = utterance

        if len(current_session) == 0:
            current_session.append(utterance)
            last_utterance_time = ts_in_second
            continue

        if ts_in_second - last_utterance_time <= max_session_interval:
            current_session.append(utterance)
            last_utterance_time = ts_in_second
        else:
            session_list.append(current_session)
            current_session = []
            current_session.append(utterance)
            last_utterance_time = ts_in_second

    if len(current_session) > 0:
        session_list.append(current_session)

    return session_list








