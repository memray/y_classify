#!/usr/bin/env bash

export EXP_MODE='continuous_feature_selection'
# 0.all, [1-8] each feature set, [9-13] combined features
# do feature selection on 9 and 11
# 20180123, do feature selection on 15-18: [1,2,3,4] + one of [5,6,7,8]

export CONTEXT=0
declare -a FEATURE_IDs=(16 17 18)
declare -a power=(3 4 5 6 7 8) # (2 3 4 5 6 7 8 9 10 -1)

export DISCRETE_NUMBER_TO_KEEP=7
for FEATURE_ID in "${FEATURE_IDs[@]}"
do
    export FEATURE_ID
    for PCA_COMPONENT in "${power[@]}"
    do
        export PCA_COMPONENT;
        echo "$EXP_MODE, PCA_COMPONENT=$PCA_COMPONENT, Feature_Number_TO_KEEP=$DISCRETE_NUMBER_TO_KEEP, Context=$CONTEXT, Feature_ID=$FEATURE_ID, with similarity";
        echo "sbatch --export=EXP_MODE=$EXP_MODE,CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,NUMBER_TO_KEEP=$DISCRETE_NUMBER_TO_KEEP,PCA_COMPONENT=$PCA_COMPONENT --job-name=$EXP_MODE.pca_component-$PCA_COMPONENT.feature_num-$DISCRETE_NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/$EXP_MODE.pca_component-$PCA_COMPONENT.feature_num-$DISCRETE_NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.out run_task.similarity.sbatch;"
#        sbatch --export=EXP_MODE=$EXP_MODE,CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,NUMBER_TO_KEEP=$DISCRETE_NUMBER_TO_KEEP,PCA_COMPONENT=$PCA_COMPONENT --job-name=$EXP_MODE.pca_component-$PCA_COMPONENT.feature_num-$DISCRETE_NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/$EXP_MODE.pca_component-$PCA_COMPONENT.feature_num-$DISCRETE_NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.out run_task.similarity.sbatch;
        sbatch --export=EXP_MODE=$EXP_MODE,CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,NUMBER_TO_KEEP=$DISCRETE_NUMBER_TO_KEEP,PCA_COMPONENT=$PCA_COMPONENT --job-name=$EXP_MODE.pca_component-$PCA_COMPONENT.feature_num-$DISCRETE_NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/$EXP_MODE.pca_component-$PCA_COMPONENT.feature_num-$DISCRETE_NUMBER_TO_KEEP.context-$CONTEXT.feature-$FEATURE_ID.similarity.out run_task.sbatch;
    done
done