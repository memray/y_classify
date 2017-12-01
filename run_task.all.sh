#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=y_classify
#SBATCH --output=tdr_dag.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

# Load modules
#module restore

# Run the job
END=8
for ((i=1;i<=END;i++)); do
    srun python -m dialogue.classify.task_runner -selected_feature_set_id $i -selected_context_id 3
done

