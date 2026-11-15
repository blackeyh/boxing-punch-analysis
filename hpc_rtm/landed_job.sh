#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
R=/l/users/<user>/punch_rtm
PART=$1
source ~/apps/miniforge3/etc/profile.d/conda.sh
conda activate punchvid
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
LOCAL_RUN=${TMPDIR:-/tmp}/landed_$SLURM_JOB_ID; mkdir -p $LOCAL_RUN
echo "node: $(hostname) start: $(date)"
python -u $R/code/landed_feats.py $R/lists/landed_clips.json $R/videos $LOCAL_RUN/landed_feats_$PART.npz $PART 2 > $LOCAL_RUN/out.log 2>&1
RC=$?
timeout 300 cp $LOCAL_RUN/out.log $R/logs/landed_$PART.log; timeout 600 cp $LOCAL_RUN/landed_feats_$PART.npz $R/out/ 2>/dev/null; rm -rf $LOCAL_RUN
echo "exit code: $RC end: $(date)"
