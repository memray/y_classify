#!/usr/bin/env bash

# 0.all, [1-8] each feature set, [9-13] combined features
# do feature selection on 9 and 11
export FEATURE_ID=11

export CONTEXT=0
declare -a power=(4 5 6 7 8 9 10 11 12)

for feature_number in "${arr[@]}"
do
    export feature_number;
    echo "Feature Selection, Feature Number=$feature_number, Context=$CONTEXT, Feature=$FEATURE_ID, with similarity";
    sbatch --export=CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,FEATURE_NUMBER=$feature_number,EXP_MODE='feature_selection' --job-name=context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/context-$CONTEXT.feature-$FEATURE_ID.similarity.run_task_log.out run_task.similarity.sbatch;
done