#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=y_classify
#SBATCH --output=last.run_task_log.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5GB

# Load modules
#module restore

# Run the job
srun python -m dialogue.classify.task_runner -selected_feature_set_id 1 2 3 4 5 6 7 8 -selected_context_id 2

