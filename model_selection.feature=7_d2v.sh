#!/usr/bin/env bash

export CONTEXT=0;
export i=7;
export CLASSIFIER="random_forest";
echo "Context=$CONTEXT, Feature=$i";
sbatch --export=CONTEXT=$CONTEXT,i=$i --job-name=context-$CONTEXT.feature-$i.run --output=slurm_log/classifier-$CLASSIFIER.context-$CONTEXT.feature-$i.out run_task.sbatch;
