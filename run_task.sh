#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=y_classify
#SBATCH --output=tdr_dag.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH –-mem=16g

# Load modules
#module restore

# Run the job
srun python -m dialogue.classify.task_runner

