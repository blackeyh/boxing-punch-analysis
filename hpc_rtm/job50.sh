#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# RTMPose on EVERY frame (50 fps) of the Olympic bouts -> out50/
R=/l/users/<user>/punch_rtm; NAME=$1
source ~/apps/miniforge3/etc/profile.d/conda.sh; conda activate punchrtm
export PYTHONUNBUFFERED=1
LOCAL_RUN=${TMPDIR:-/tmp}/rtm50_$SLURM_JOB_ID; mkdir -p $LOCAL_RUN; export TMPDIR=$LOCAL_RUN; mkdir -p $R/out50
echo "node: $(hostname) start: $(date)"
python -u $R/code/extract.py $R/lists/$NAME.txt $R/videos $R/out50 --step 1 > $LOCAL_RUN/output.log 2>&1 &
PID=$!
( while kill -0 $PID 2>/dev/null; do sleep 60; timeout 60 cp $LOCAL_RUN/output.log $R/logs/.$NAME.part && timeout 60 mv $R/logs/.$NAME.part $R/logs/$NAME.log; done ) &
wait $PID; RC=$?; timeout 120 cp $LOCAL_RUN/output.log $R/logs/$NAME.log; rm -rf $LOCAL_RUN; echo "exit code: $RC end: $(date)"
