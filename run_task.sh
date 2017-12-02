#!/usr/bin/env bash

START=1
END=1

for ((CONTEXT=1;CONTEXT<=3;CONTEXT++));
do
    export CONTEXT
    for ((i=START;i<=END;i++)); do
        export i;
        echo "Context=$CONTEXT, Feature=$i";
        sbatch --export=CONTEXT=$CONTEXT,i=$i --job-name=context=$CONTEXT.feature=$i.run --output=slurm_log/context=$CONTEXT.feature=$i.run_task_log.out run_task.sbatch;
    done
done

export CONTEXT=0
for ((i=START;i<=END;i++));
do
    export i;
    echo "Context=$CONTEXT, Feature=$i, with similarity";
    sbatch --export=CONTEXT=$CONTEXT,i=$i --job-name=context=$CONTEXT.feature=$i.similarity.run --output=slurm_log/context=$CONTEXT.feature=$i.similarity.run_task_log.out run_task.similarity.sbatch;
done