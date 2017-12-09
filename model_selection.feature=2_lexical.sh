#!/usr/bin/env bash

export CONTEXT=0;
export i=2;
export CLASSIFIER="lr_svm_c=0";
echo "Context=$CONTEXT, Feature=$i";
sbatch --export=CONTEXT=$CONTEXT,i=$i --job-name=context-$CONTEXT.feature-$i.run --output=slurm_log/classifier-$CLASSIFIER.context-$CONTEXT.feature-$i.out run_task.sbatch;
