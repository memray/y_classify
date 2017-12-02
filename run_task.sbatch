#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=y_classify
#SBATCH --output=slurm_log/context=$CONTEXT.feature=$i.run_task_log.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

# Load modules
#module restore

# Run the job
echo "python -m dialogue.classify.task_runner -selected_feature_set_id $i -selected_context_id $CONTEXT"
srun python -m dialogue.classify.task_runner -selected_feature_set_id $i -selected_context_id $CONTEXT