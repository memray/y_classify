/*
    Pig script to count the number of different types of events per user per day
 */

register 'eventudf.py' using jython as eventudf;


set default_parallel 10
%default reduceNum 10

%default OUTPUT '/user/zhenyue/bots/noti.weather.fb.perDay'

data = LOAD 'uapi_analytics.uapi_logs' USING org.apache.hive.hcatalog.pig.HCatLoader();

data_filtered = filter data by (
    dt >= '2017-03-01-00' and dt < '2017-03-20-00'
	and msg_sentto_displayname matches '.*Weather.*'
--	and mtestid matches '.*B3433.*'
    and msg_sentto_env == 'prod'
    and platform == 'facebook'
	);


data_processed = foreach data_filtered generate
	(chararray) useruuid,
	(chararray) SUBSTRING (dt,0,10) as dt_day,
	(int)(ts/1000) as ts_in_second,
	(chararray) event_trigger,
	(chararray) direction,
	(chararray) platform_message_id;


data_group = GROUP data_processed BY (useruuid, dt_day);

data_group_processed = FOREACH data_group  {
               ordered = ORDER $1 BY ts_in_second ASC;
               GENERATE FLATTEN (group) AS (useruuid, dt_day),
               ordered AS events;
               }


data_perDay = FOREACH data_group_processed GENERATE useruuid, dt_day, FLATTEN(eventudf.count_event(events))
                  AS (n_message, n_notification, n_delivery, n_read, n_bottosb, n_bottouser, n_bottouser_post,
                  visit_span, delivery_span, read_span);


perDay = DISTINCT data_perDay PARALLEL 1;
STORE perDay INTO '$OUTPUT.0301-0319' USING org.apache.pig.piggybank.storage.PigStorageSchema();

