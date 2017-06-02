/*
    Pig script to count the number of different types of events per user per day
 */

register /homes/rmeng/lib/jyson-1.0.2.jar;
register 'udf_session_monkeypets.py' using jython as sessionudf;

%default BOT_NAME 'Monkey_Pets';
%default time_start '2017-04-01-00';
%default time_end '2017-04-30-24';

SET default_parallel 10;
%default reduceNum 10;
%default OUTPUT '/user/rmeng/$BOT_NAME.interval=5min.session';

rmf $OUTPUT

data = LOAD 'uapi_analytics.uapi_logs' USING org.apache.hive.hcatalog.pig.HCatLoader();

data_filtered = filter data by (
	msg_sentto_displayname matches 'Family.*Assistant'
    AND msg_sentto_env == 'prod'
--    AND platform == 'facebook'
--    AND msg_text IS NOT NULL # filter later
--    AND dt >= '$time_start'
--    AND dt < '$time_end'
    AND (direction == 'bot_to_sb' or direction == 'user_to_sb')
    AND (event_trigger == 'message' OR event_trigger == 'notification')
);

data_processed = foreach data_filtered generate
	-- date
	(chararray) time,
	(chararray) SUBSTRING (dt,0,10) as dt_day,
	(int)(ts/1000) as ts_in_second,
    -- metadata
	(chararray) useruuid,
	(chararray) direction,
    -- message
	(chararray) msg,
	(chararray) msg_sentto,
	(chararray) msg_types,
	(chararray) msg_sentto_displayname,
	(chararray) platform_message_id,
	(chararray) msg_text,
	(boolean) is_suggested_response,
	
	-- botlog
	(chararray) botlog,
	(chararray) botlog_intent,
	(chararray) botlog_slots;

-- Group utterances by useruuid
data_group = GROUP data_processed BY (useruuid);

-- For each group (utterances of one user), order utterances by time and do sessionization
data_group_sessionized = FOREACH data_group  {
   ordered_groups = ORDER $1 BY time ASC;
   GENERATE FLATTEN ($0) AS userid, sessionudf.split_session(ordered_groups);
           }

-- Reduce results
reduced_data = DISTINCT data_group_sessionized PARALLEL 1;

-- Write results into JSON
STORE reduced_data INTO '$OUTPUT' USING org.apache.pig.piggybank.storage.PigStorageSchema();
--STORE reduced_data INTO '$OUTPUT' USING org.apache.pig.builtin.JsonStorage();