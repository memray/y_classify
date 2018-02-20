#!/usr/bin/env bash

export EXP_MODE='cross_validation'
export CONTEXT=3; # context=all
export FEATURE_ID=2;
export CLASSIFIER="rf.#tree=1024";
echo "Context=$CONTEXT, Feature=$FEATURE_ID, Classifier=$CLASSIFIER";

sbatch --export=CONTEXT=$CONTEXT,FEATURE_ID=$FEATURE_ID,EXP_MODE=$EXP_MODE --job-name=classifier-$CLASSIFIER.$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.similarity.run --output=slurm_log/classifier-$CLASSIFIER.$EXP_MODE.context-$CONTEXT.feature-$FEATURE_ID.similarity.run_task_log.out run_task.similarity.sbatch;