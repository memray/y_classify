/*
    Pig script to count the number of different types of events per user per day
 */

register /homes/rmeng/lib/jyson-1.0.2.jar;
register 'udf_check_count.py' using jython as sessionudf;

%default BOT_NAME 'Family_Assistant';
%default time_start '2017-03-07-00';
%default time_end '2017-04-30-24';

SET default_parallel 10;
%default reduceNum 10;
%default OUTPUT '/user/rmeng/$BOT_NAME.20170307.log_count_after_filter';

rmf $OUTPUT

data = LOAD 'uapi_analytics.uapi_logs' USING org.apache.hive.hcatalog.pig.HCatLoader();

data_filtered = filter data by (
	msg_sentto_displayname matches 'Family.*Assistant'
    AND msg_sentto_env == 'prod'
--    AND platform == 'facebook'
    AND msg_text IS NOT NULL
    AND dt >= '$time_start'
--    AND dt < '$time_end'
    AND (direction == 'bot_to_sb' OR direction == 'user_to_sb')
--    AND (event_trigger == 'message' OR event_trigger == 'notification') -- doesn't matter, all records belong to these two events
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

--LOG_COUNT = FOREACH (GROUP data_processed ALL) GENERATE COUNT(data_processed);

-- Group utterances by useruuid
data_group = GROUP data_processed BY (useruuid);

-- For each group (utterances of one user), order utterances by time and do sessionization
log_count_each_user = FOREACH data_group  {
   ordered_groups = ORDER $1 BY time ASC;
   GENERATE FLATTEN ($0) AS userid, sessionudf.split_session(ordered_groups) AS (u_count: int);
           }

log_count_group = GROUP log_count_each_user ALL;
LOG_COUNT_AFTER_SESSION = FOREACH log_count_group GENERATE SUM(log_count_each_user.u_count);

-- Reduce results
--reduced_data = DISTINCT data_group_sessionized PARALLEL 1;

--output_union = UNION LOG_COUNT, LOG_COUNT_AFTER_SESSION;
-- Write results into JSON
STORE LOG_COUNT_AFTER_SESSION INTO '$OUTPUT' USING org.apache.pig.piggybank.storage.PigStorageSchema();
--STORE reduced_data INTO '$OUTPUT' USING org.apache.pig.builtin.JsonStorage();