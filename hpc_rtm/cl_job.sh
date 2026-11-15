#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
# usage: sbatch cl_job.sh extract LIST.json OUT.npy T PRE | sbatch --gres=gpu:1 cl_job.sh train HELD MODEL | sbatch --gres=gpu:1 cl_job.sh scan LIST.json CLIPS.npy OUT.npz C0 OFFSETS [--model-gym G]
R=/l/users/<user>/punch_rtm; MODE=$1; shift
source ~/apps/miniforge3/etc/profile.d/conda.sh; conda activate punchvid
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
LOCAL_RUN=${TMPDIR:-/tmp}/cl_$SLURM_JOB_ID; mkdir -p $LOCAL_RUN; export TMPDIR=$LOCAL_RUN
echo "node: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none) start: $(date)"
if [ "$MODE" = extract ]; then
  NAME=cl_extract_${2%.npy}
  python -u $R/code/extract_clips2.py $R/lists/$1 $R/videos $R/clips/$2 $3 $4 > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = extract3 ]; then       # extract3 LIST.json OUT.npy T PRE: clips with the puncher's upper arm drawn in
  NAME=cl_extract_${2%.npy}
  python -u $R/code/extract_clips3.py $R/lists/$1 $R/videos $R/clips/$2 $3 $4 > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = trainarm ]; then
  HELD=$1; MODEL=$2; NAME=cl_arm_${MODEL}_${HELD}
  python -u $R/code/contact_landed_finetune.py $R/lists/landed_clips_arm.json $R/clips/landed_arm_44x224.npy $LOCAL_RUN/$NAME.npz $HELD --model $MODEL \
    --chain-json $R/lists/chain_clips_v5.json --chain-npy $R/clips/chain_v5_44x224.npy --save-model $R/models/cl_arm_${MODEL}_heldout_${HELD}.pt > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = traintc ]; then
  HELD=$1; MODEL=$2; NAME=cl_tc_${MODEL}_${HELD}
  python -u $R/code/contact_landed_finetune.py $R/lists/landed_clips_tc.json $R/clips/landed_tc_44x224.npy $LOCAL_RUN/$NAME.npz $HELD --model $MODEL \
    --chain-json $R/lists/chain_clips_v4.json --chain-npy $R/clips/chain_v4_44x224.npy --save-model $R/models/cl_tc_${MODEL}_heldout_${HELD}.pt > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = trainx ]; then        # trainx NAME HELD [extra args]: referee clips only (no chain scan, model not saved)
  NAME=$1; HELD=$2; shift 2
  python -u $R/code/contact_landed_finetune.py $R/lists/landed_clips.json $R/clips/landed_glove_44x224.npy $LOCAL_RUN/$NAME.npz $HELD --model large "$@" > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = trainy ]; then        # trainy NAME HELD REF.json REF.npy [extra args]: any clip set, no chain scan unless given in the extra args
  NAME=$1; HELD=$2; RJ=$3; RN=$4; shift 4
  python -u $R/code/contact_landed_finetune.py $R/lists/$RJ $R/clips/$RN $LOCAL_RUN/$NAME.npz $HELD --model large "$@" > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = type ]; then          # type NAME [extra args]: punch type (straight / hook / uppercut) from the BoxingWeb clips
  NAME=$1; shift
  python -u $R/code/type_finetune.py $R/lists/bw_clips.json $R/clips/bw_44x224.npy $LOCAL_RUN/$NAME.npz --save-model $R/models/$NAME.pt "$@" > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = trainbw ]; then
  HELD=$1; MODEL=$2; NAME=cl_bw_${MODEL}_${HELD}
  python -u $R/code/contact_landed_finetune.py $R/lists/landed_clips.json $R/clips/landed_glove_44x224.npy $LOCAL_RUN/$NAME.npz $HELD --model $MODEL \
    --chain-json $R/lists/chain_clips_v3.json --chain-npy $R/clips/chain_v3_44x224.npy --save-model $R/models/cl_bw_${MODEL}_heldout_${HELD}.pt \
    --bw-json $R/lists/bw_clips.json --bw-npy $R/clips/bw_44x224.npy --bw-scan $R/out/scan_bw_${HELD}.npz --bw-min 0.8 > $LOCAL_RUN/out.log 2>&1 &
elif [ "$MODE" = scan ]; then
  NAME=${3%.npz}
  python -u $R/code/cl_scan.py $R/lists/$1 $R/clips/$2 $LOCAL_RUN/$NAME.npz $R/models $4 $5 $6 $7 > $LOCAL_RUN/out.log 2>&1 &
else
  HELD=$1; MODEL=$2; NAME=cl_${MODEL}_${HELD}
  python -u $R/code/contact_landed_finetune.py $R/lists/landed_clips.json $R/clips/landed_glove_44x224.npy $LOCAL_RUN/$NAME.npz $HELD --model $MODEL \
    --chain-json $R/lists/chain_clips_v3.json --chain-npy $R/clips/chain_v3_44x224.npy --save-model $R/models/cl_${MODEL}_heldout_${HELD}.pt > $LOCAL_RUN/out.log 2>&1 &
fi
PID=$!
( while kill -0 $PID 2>/dev/null; do sleep 60; timeout 60 cp $LOCAL_RUN/out.log $R/logs/.$NAME.part && timeout 60 mv $R/logs/.$NAME.part $R/logs/$NAME.log; done ) &
wait $PID; RC=$?
timeout 120 cp $LOCAL_RUN/out.log $R/logs/$NAME.log; [ -f $LOCAL_RUN/$NAME.npz ] && timeout 300 cp $LOCAL_RUN/$NAME.npz $R/out/
rm -rf $LOCAL_RUN; echo "exit code: $RC end: $(date)"
