#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --partition=high-mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB

# set max wallclock time
# send mail to this address
#SBATCH --mail-user=memray0@gmail.com

# Load modules
#module restore

# Run the job
COMMAND="/ihome/pbrusilosky/rum20/packages/anaconda3/bin/python -m dialogue.classify.task_runner -experiment_mode $EXP_MODE -selected_feature_set_id $FEATURE_ID -add_similarity_feature -selected_context_id $CONTEXT -experiment_mode $EXP_MODE" -k_feature_to_keep $FEATURE_NUMBER
echo $COMMAND
srun $COMMAND
