from __future__ import division
import com.xhaus.jyson.JysonCodec as json

import string
import sys
import math
import re

# -*- coding: utf-8 -*-

@outputSchema('session_json: chararray')

def split_session(user_utterances):
    '''
    Given a handful of utterances of one user, segment the utterances into sessions
    :return: a list of sessions, each session consists of a few utterances
    '''
    max_session_interval = 300 # in second, try two interval values: 300 (5 mins) and 1800 (30 mins)
    session_list = []
    current_session = []
    last_utterance_time = None

    # iterate all the utterances and identify sessions
    for utterance in user_utterances:
        (time, dt_day, ts_in_second, useruuid, direction, msg, msg_sentto, msg_types, msg_sentto_displayname, platform_message_id, msg_text, is_suggested_response, botlog, botlog_intent, botlog_slots) = utterance

        # convert to a dict
        field_names = ['time', 'dt_day', 'ts_in_second', 'useruuid', 'direction', 'msg', 'msg_sentto', 'msg_types', 'msg_sentto_displayname', 'platform_message_id', 'msg_text', 'is_suggested_response', 'botlog', 'botlog_intent', 'botlog_slots'];
        utterance_dict = {}
        for v_id, val in enumerate(utterance):
            utterance_dict[field_names[v_id]] = val

        # ignore the "show-typing" message
        # if msg_types == '["show-typing"]':
        #     continue

        # put in the first utterance
        if len(current_session) == 0:
            current_session.append(utterance_dict)
            last_utterance_time = ts_in_second
            continue

        if ts_in_second - last_utterance_time <= max_session_interval:
            current_session.append(utterance_dict)
            last_utterance_time = ts_in_second
        else:
            '''
            identify the validity of current session
            '''
            # keep the valid sessions and messages only
            # if is_valid_session(current_session):
            session_list.append(current_session)

            current_session = []
            current_session.append(utterance_dict)
            last_utterance_time = ts_in_second

    # reach the end of data
    # if is_valid_session(current_session):
    if len(current_session) > 0:
        session_list.append(current_session)

    # if empty return None
    if len(session_list) > 0:
        return json.dumps(session_list)
    else:
        return None