from __future__ import division
import string
import sys
import math
import re

# -*- coding: utf-8 -*-

@outputSchema("record:{(n_message:int, n_notification:int, n_delivery:int, n_read:int, n_bottosb:int, n_bottouser:int, "
              "n_bottouser_post:int, visit_span:int, delivery_span:int, read_span:int)}")


def count_event(events):
    n_message = 0
    n_notification = 0
    n_delivery = 0
    n_read = 0
    n_bottosb = 0
    n_bottouser = 0
    n_bottouser_post = 0

    first_message_time = None
    first_notification_time = None
    notification_delivery_time = None
    notification_read_time = None
    visit_span = None
    delivery_span = None
    read_span = None

    for event in events:
        (useruuid, dt_day, ts_in_second, event_trigger, direction, platform_message_id) = event

        if (direction == 'user_to_sb') and (event_trigger == 'message'):
            n_message = n_message + 1
            if first_message_time is None:
                first_message_time  = ts_in_second

        if (event_trigger == 'notification'):
            n_notification = n_notification + 1
            if first_notification_time is None:
                first_notification_time  = ts_in_second

        if (event_trigger == 'delivery_receipt'):
            n_delivery = n_delivery + 1
            if (notification_delivery_time is None) and (first_notification_time is not None) and (ts_in_second >= first_notification_time):
                notification_delivery_time  = ts_in_second

        if (event_trigger == 'read_receipt'):
            n_read = n_read + 1
            if (notification_read_time is None) and (first_notification_time is not None) and (ts_in_second >= first_notification_time):
                notification_read_time  = ts_in_second

        if (direction == 'bot_to_sb') and (event_trigger == 'message'):
            n_bottosb = n_bottosb + 1

        if (direction == 'bot_to_user'):
            n_bottouser = n_bottouser + 1
            if platform_message_id is not None:
                n_bottouser_post += 1


    if (first_message_time is not None) and (first_notification_time is not None):
        visit_span = first_message_time - first_notification_time

    if (notification_delivery_time is not None) and (first_notification_time is not None):
        delivery_span = notification_delivery_time - first_notification_time

    if (notification_read_time is not None) and (first_notification_time is not None):
        read_span = notification_read_time - first_notification_time


    return ((n_message, n_notification, n_delivery, n_read, n_bottosb, n_bottouser, n_bottouser_post, visit_span, delivery_span, read_span))
