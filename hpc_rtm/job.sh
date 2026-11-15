#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# usage: sbatch --job-name=rtm-NAME --output=$R/logs/NAME.slurm.log job.sh NAME [extra args]
R=/l/users/<user>/punch_rtm
NAME=$1; shift
source ~/apps/miniforge3/etc/profile.d/conda.sh
conda activate punchrtm
export PYTHONUNBUFFERED=1
LOCAL_RUN=${TMPDIR:-/tmp}/rtm_$SLURM_JOB_ID; mkdir -p $LOCAL_RUN; export TMPDIR=$LOCAL_RUN
publish() { [ -f $LOCAL_RUN/output.log ] && timeout 60 cp $LOCAL_RUN/output.log $R/logs/.$NAME.part && timeout 60 mv $R/logs/.$NAME.part $R/logs/$NAME.log; }
( while true; do sleep 30; publish; done ) &
PUB=$!
echo "node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)  start: $(date)"
python -u $R/code/extract.py $R/lists/$NAME.txt $R/videos $R/out "$@" > $LOCAL_RUN/output.log 2>&1
RC=$?
kill $PUB 2>/dev/null; publish; rm -rf $LOCAL_RUN
echo "exit code: $RC  end: $(date)"
