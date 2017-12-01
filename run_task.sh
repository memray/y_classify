#!/usr/bin/env bash

START=1
END=8

for ((i=START;i<=END;i++)); do
    export i;
    for ((CONTEXT=1;CONTEXT<=3;i++)); do
        export CONTEXT=0
        sbatch run_task.sbatch;
    done
done

export CONTEXT=0
for ((i=START;i<=END;i++)); do
    export i;
    sbatch run_task.similarity.sbatch;
done