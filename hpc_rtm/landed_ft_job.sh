#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# usage: sbatch landed_ft_job.sh extract [LIST.json OUT.npy] | sbatch --gres=gpu:1 landed_ft_job.sh finetune HELD MODEL [extra args] | ... chain HELD MODEL
R=/l/users/<user>/punch_rtm; MODE=$1; shift
source ~/apps/miniforge3/etc/profile.d/conda.sh; conda activate punchvid
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
LOCAL_RUN=${TMPDIR:-/tmp}/landedft_$SLURM_JOB_ID; mkdir -p $LOCAL_RUN; export TMPDIR=$LOCAL_RUN
echo "node: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none) start: $(date)"
if [ "$MODE" = extract ]; then
  LIST=${1:-landed_clips.json}; OUTN=${2:-landed_glove_20x224.npy}; NAME=landed_extract_${OUTN%.npy}
  python -u $R/code/landed_extract_clips.py $R/lists/$LIST $R/videos $R/clips/$OUTN > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = predict ]; then
  LIST=$1; NPY=$2; OUTN=$3; TAG=$4; MODEL=$5; NAME=${OUTN%.npz}
  python -u $R/code/landed_predict.py $R/lists/$LIST $R/clips/$NPY $LOCAL_RUN/$NAME.npz $R/models $TAG $MODEL > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = chain ]; then
  HELD=$1; MODEL=$2; TAG=${3:-chain}; shift 3; NAME=landed_${TAG}_${MODEL}_${HELD}
  python -u $R/code/landed_finetune.py $R/lists/landed_clips.json $R/clips/landed_glove_20x224.npy $LOCAL_RUN/$NAME.npz $HELD --model $MODEL \
    --extra-json $R/lists/chain_clips.json --extra-npy $R/clips/chain_20x224.npy --save-model $R/models/landed_${TAG}_${MODEL}_heldout_${HELD}.pt "$@" > $LOCAL_RUN/out.log 2>&1 &
else
  HELD=$1; MODEL=$2; shift 2; NAME=landed_ft_${MODEL}_${HELD}
  python -u $R/code/landed_finetune.py $R/lists/landed_clips.json $R/clips/landed_glove_20x224.npy $LOCAL_RUN/$NAME.npz $HELD --model $MODEL "$@" > $LOCAL_RUN/out.log 2>&1 &
fi
PID=$!
( while kill -0 $PID 2>/dev/null; do sleep 60; timeout 60 cp $LOCAL_RUN/out.log $R/logs/.$NAME.part && timeout 60 mv $R/logs/.$NAME.part $R/logs/$NAME.log; done ) &
wait $PID; RC=$?
timeout 120 cp $LOCAL_RUN/out.log $R/logs/$NAME.log; [ -f $LOCAL_RUN/$NAME.npz ] && timeout 300 cp $LOCAL_RUN/$NAME.npz $R/out/
rm -rf $LOCAL_RUN; echo "exit code: $RC end: $(date)"
