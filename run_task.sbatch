#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
#module restore

export PYTHONPATH="/ihome/pbrusilosky/rum20/.conda/envs/py35/bin/python"

# Run the job
COMMAND="/ihome/pbrusilosky/rum20/.conda/envs/py35/bin/python -m dialogue.classify.task_runner -selected_feature_set_id $i -selected_context_id $CONTEXT"
echo COMMAND
srun COMMAND
