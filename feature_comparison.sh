#!/usr/bin/env bash

EXP_MODE='cross_validation'
# 0.all, [1-8] each feature set, [9-13] combined features
START=14
END=14

for ((CONTEXT=0;CONTEXT<=0;CONTEXT++));
do
    export CONTEXT
    for ((FEATURE_ID=START;FEATURE_ID<=END;FEATURE_ID++)); do
        export FEATURE_ID;
        echo "Context=$CONTEXT, Feature=$FEATURE_ID";
        sbatch --export=CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,EXP_MODE=$EXP_MODE --job-name=$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.run --output=slurm_log/$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.run_task_log.out run_task.sbatch;
    done
done

export CONTEXT=0
for ((FEATURE_ID=START;FEATURE_ID<=END;FEATURE_ID++));
do
    export FEATURE_ID;
    echo "Context=$CONTEXT, Feature=$FEATURE_ID, with similarity";
    sbatch --export=CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,EXP_MODE=$EXP_MODE --job-name=$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.similarity.run_task_log.out run_task.similarity.sbatch;
done