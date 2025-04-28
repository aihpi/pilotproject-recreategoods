#!/bin/bash -eux

# ==============================
# SLURM Queue Configuration 
# ==============================
# Set default queue
QUEUE="defq"

# Check if queue is provided as command-line argument
if [ $# -ge 1 ]; then
  QUEUE="$1"
fi

# Select the appropriate SLURM script based on queue
if [ "$QUEUE" == "defq" ]; then
  SLURM_SCRIPT="webapp/defq.slurm"
  echo "Using defq queue (Partition: defq, QOS: normal)"
elif [ "$QUEUE" == "shortq" ]; then
  SLURM_SCRIPT="webapp/shortq.slurm"
  echo "Using shortq queue (Partition: shortq, QOS: shortjob)"
else
  echo "Invalid queue: $QUEUE. Valid options are 'defq' or 'shortq'. Using defq as default."
  QUEUE="defq"
  SLURM_SCRIPT="webapp/defq.slurm"
  echo "Using defq queue (Partition: defq, QOS: normal)"
fi

# Submit the job
job_id=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
echo "Submitted job with ID: $job_id" 