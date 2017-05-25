/*
    Pig script to count the number of different types of events per user per day
 */

--register 'eventudf.py' using jython as eventudf;

%default BOT_NAME 'Family Assistant';
%default time_start '2017-04-01-00';
%default time_end '2017-04-30-24';

SET default_parallel 10;
%default reduceNum 10;
%default OUTPUT '/user/rmeng/$BOT_NAME';

data = LOAD 'uapi_analytics.uapi_logs' USING org.apache.hive.hcatalog.pig.HCatLoader();

data_filtered = filter data by (
	msg_sentto_displayname matches $BOT_NAME
    and msg_sentto_env != NULL
    and msg_sentto_env == 'prod'
--    and dt >= time_start
--    and dt < time_end
    and (direction == 'bot_to_sb' or direction == 'user_to_sb')
	);

data_processed = foreach data_filtered generate
    -- metadata
	(chararray) time,
	(chararray) useruuid,
	(chararray) direction,
	(chararray) platform,
	(chararray) msg_sentto,
	(chararray) msg_types,
	(chararray) msg_sentto_displayname,
	-- date
	(chararray) SUBSTRING (dt,0,10) as dt_day,
	-- time stamp
	(int)(ts/1000) as ts_in_second,
	(chararray) platform_message_id,
	-- text
	(chararray) msg_text,
	-- NLU
	(chararray) botlog_intent,
	(chararray) botlog_slots;


data_group = GROUP data_processed BY (useruuid, dt_day);
--
--data_group_processed = FOREACH data_group  {
--               ordered = ORDER $1 BY ts_in_second ASC;
--               GENERATE FLATTEN (group) AS (useruuid, dt_day),
--               ordered AS events;
--               }
--
--
--data_perDay = FOREACH data_group_processed GENERATE useruuid, dt_day, FLATTEN(eventudf.count_event(events))
--                  AS (n_message, n_notification, n_delivery, n_read, n_bottosb, n_bottouser, n_bottouser_post,
--                  visit_span, delivery_span, read_span);
--
--perDay = DISTINCT data_perDay PARALLEL 1;

STORE data_group INTO '$OUTPUT.20170525-0319' USING org.apache.pig.piggybank.storage.PigStorageSchema();

