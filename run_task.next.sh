#!/usr/bin/env bash

START=1
END=8

for ((i=START;i<=END;i++)); do
    export i;
    sbatch run_task.next.sbatch;
done