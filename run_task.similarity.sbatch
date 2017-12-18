#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --partition=high-mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB

# set max wallclock time
#SBATCH --time=12:00:00
# send mail to this address
#SBATCH --mail-user=memray0@gmail.com

# Load modules
#module restore

# Run the job
COMMAND="/ihome/pbrusilosky/rum20/packages/anaconda3/bin/python -m dialogue.classify.task_runner -selected_feature_set_id $i -add_similarity_feature -selected_context_id $CONTEXT"
echo $COMMAND
srun $COMMAND
