/*
    Pig script to count the number of different types of events per user per day
 */

register 'udf_session.py' using jython as sessionudf;

%default BOT_NAME 'Family_Assistant';
%default time_start '2017-04-01-00';
%default time_end '2017-04-30-24';
%default MAX_SESSION_INTERVAL 300; -- in second, try two interval values: 300 (5 mins) and 1800 (30 mins)

SET default_parallel 10;
%default reduceNum 10;
%default OUTPUT '/user/rmeng/$BOT_NAME.interval=5min.sessionized';

rmf $OUTPUT

data = LOAD 'uapi_analytics.uapi_logs' USING org.apache.hive.hcatalog.pig.HCatLoader();

data_filtered = filter data by (
	msg_sentto_displayname matches 'Family.*Assistant'
    and msg_sentto_env == 'prod'
    and msg_text IS NOT NULL
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
	-- NLU,
	(chararray) botlog_intent,
	(chararray) botlog_slots,
	-- text
	(chararray) msg_text;

-- Group utterances by useruuid
data_group = GROUP data_processed BY (useruuid);

-- For each group (utterances of one user), order utterances by time and do sessionization
data_group_sessionized = FOREACH data_group  {
                               ordered_groups = ORDER $1 BY ts_in_second ASC;
                               GENERATE FLATTEN ($0) AS useruuid, sessionudf.split_session(ordered_groups, MAX_SESSION_INTERVAL);
                         }

-- Reduce results
reduced_data = DISTINCT data_group_sessionized PARALLEL 1;

-- Write results into JSON
--STORE reduced_data INTO '$OUTPUT' USING org.apache.pig.piggybank.storage.PigStorageSchema();
STORE reduced_data INTO '$OUTPUT' USING org.apache.pig.builtin.JsonStorage();
