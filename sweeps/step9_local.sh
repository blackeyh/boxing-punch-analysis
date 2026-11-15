#!/bin/bash
# Step 9 run LOCALLY on the Mac (MPS): the HPC's /l storage got stuck on a Lustre lock held by an
# unkillable (state D) process from a dead sync connection, freezing all HPC jobs that touch it.
# Same commands as sweeps/step9_conf_and_ensemble.txt; results in runs/s9-<name>-local/.
set -e
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/bin/python
run() { name=$1; shift; echo "=== $name: $* ($(date +%H:%M:%S))"; $PY deep.py cnn --seeds 0 1 2 3 4 "$@" --out runs/$name > runs/$name.log 2>&1; grep "==>" runs/$name.log; }
run s9-2d-local                    --input 2d
run s9-lift-rot-local              --input lift-rot
run s9-mp3d-rot-fill-local         --input mp3d-rot --conf-fill 0.5
run s9-mp3d-rot-fill-conf-local    --input mp3d-rot --conf-fill 0.5 --conf-input
run s9-2d-mp3d-rot-fill-conf-local --input 2d+mp3d-rot --conf-fill 0.5 --conf-input
echo "=== all done $(date +%H:%M:%S)"
