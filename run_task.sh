#!/usr/bin/env bash

START=1
END=1

for ((CONTEXT=1;CONTEXT<=3;CONTEXT++));
do
    export CONTEXT
    for ((i=START;i<=END;i++)); do
        export i;
        echo "Context=$CONTEXT, Feature=$i";
        sbatch --export=CONTEXT=$CONTEXT,i=$i run_task.sbatch;
    done
done

export CONTEXT=0
for ((i=START;i<=END;i++));
do
    export i;
    echo "Context=$CONTEXT, Feature=$i, with similarity";
    sbatch --export=CONTEXT=$CONTEXT,i=$i run_task.similarity.sbatch;
done