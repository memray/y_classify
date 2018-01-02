#!/usr/bin/env bash

# 0.all, [1-8] each feature set, [9-13] combined features
# do feature selection on 9 and 11

export CONTEXT=0
declare -a FEATURE_IDs=(14) #(0 9 10 11 12 13)
declare -a power=(4 5 6 7 8 9 10 11 12 -1)

for FEATURE_ID in "${FEATURE_IDs[@]}"
do
    export FEATURE_ID
    for NUMBER_TO_KEEP in "${power[@]}"
    do
        export NUMBER_TO_KEEP;
        echo "Feature Selection, Feature_Number_TO_KEEP=$NUMBER_TO_KEEP, Context=$CONTEXT, Feature_ID=$FEATURE_ID, with similarity";
        sbatch --export=CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,NUMBER_TO_KEEP=$NUMBER_TO_KEEP,EXP_MODE='feature_selection' --job-name=feature_selection.no_regularization.feature_num-$NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/feature_selection.no_regularization.feature_num-$NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.out run_task.similarity.sbatch;
    done
done