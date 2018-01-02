#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --partition=high-mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB

# set max wallclock time
#SBATCH --time=24:00:00
# send mail to this address
#SBATCH --mail-user=memray0@gmail.com

# Load modules
#module restore

# Run the job
if [ "$EXP_MODE" = "cross_validation" ]; then
    COMMAND="/ihome/pbrusilosky/rum20/packages/anaconda3/bin/python -m dialogue.classify.task_runner -experiment_mode $EXP_MODE -selected_feature_set_id $FEATURE_ID  -selected_context_id $CONTEXT"
else
    COMMAND="/ihome/pbrusilosky/rum20/packages/anaconda3/bin/python -m dialogue.classify.task_runner -experiment_mode $EXP_MODE -selected_feature_set_id $FEATURE_ID  -selected_context_id $CONTEXT -k_feature_to_keep $NUMBER_TO_KEEP"
fi

echo $COMMAND
srun $COMMAND
