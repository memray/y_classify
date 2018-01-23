#!/usr/bin/env bash

export EXP_MODE='cross_validation'

# 0.all, [1-8] each feature set, [9-13] is combined features([1,3,4] + one of [2,5,6,7,8]), [14] is [1,3,4]
START=0
END=14

#for ((CONTEXT=0;CONTEXT<=3;CONTEXT++));
#do
#    export CONTEXT
#    for ((FEATURE_ID=START;FEATURE_ID<=END;FEATURE_ID++)); do
#        export FEATURE_ID;
#        echo "$EXP_MODE, Context=$CONTEXT, Feature=$FEATURE_ID";
#        sbatch --export=CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,EXP_MODE=$EXP_MODE --job-name=$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.run --output=slurm_log/$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.run_task_log.out run_task.sbatch;
#    done
#done

#export CONTEXT=0

for ((CONTEXT=1;CONTEXT<=3;CONTEXT++));
do
export CONTEXT
for ((FEATURE_ID=START;FEATURE_ID<=END;FEATURE_ID++));
do
    export FEATURE_ID;
    echo "$EXP_MODE, Context=$CONTEXT, Feature=$FEATURE_ID, with similarity";
    sbatch --export=CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,EXP_MODE=$EXP_MODE --job-name=$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.similarity.run_task_log.out run_task.similarity.sbatch;
done
done
