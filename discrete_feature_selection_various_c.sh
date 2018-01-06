#!/usr/bin/env bash

export EXP_MODE='discrete_feature_selection'
# 0.all, [1-8] each feature set, [9-13] combined features
# do feature selection on 9 and 11

export CONTEXT=0
declare -a FEATURE_IDs=(0 10 11 12 13)
declare -a power=(8)
export C=8

for FEATURE_ID in "${FEATURE_IDs[@]}"
do
    export FEATURE_ID
    for NUMBER_TO_KEEP in "${power[@]}"
    do
        export NUMBER_TO_KEEP;
        echo "$EXP_MODE, Feature_Number_TO_KEEP=$NUMBER_TO_KEEP, Context=$CONTEXT, Feature_ID=$FEATURE_ID, with similarity";
        sbatch --export=CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,NUMBER_TO_KEEP=$NUMBER_TO_KEEP,EXP_MODE=$EXP_MODE --job-name=$EXP_MODE.C-$C.feature_num-$NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/$EXP_MODE.C-$C.feature_num-$NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.out run_task.similarity.sbatch;
    done
done