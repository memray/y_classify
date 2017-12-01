#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=y_classify
#SBATCH --output=next.run_task_log.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

# Load modules
#module restore

# Run the job
END=8
for ((i=1;i<=END;i++)); do
    echo "python -m dialogue.classify.task_runner -selected_feature_set_id $i -selected_context_id 0"
    srun python -m dialogue.classify.task_runner -selected_feature_set_id $i -selected_context_id 0
done
