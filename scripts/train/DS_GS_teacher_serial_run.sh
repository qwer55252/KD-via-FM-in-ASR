#!/usr/bin/env bash
# /workspace/KD-via-FM-in-ASR/scripts/train/run_gs_chain.sh
set -Eeuo pipefail

A="/workspace/KD-via-FM-in-ASR/scripts/train/DS_GSxs_teacher.sh"
B="/workspace/KD-via-FM-in-ASR/scripts/train/DS_GSs_teacher.sh"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] START A: $A"
bash "$A"
echo "[$(ts)] DONE  A"

echo "[$(ts)] START B: $B"
bash "$B"
echo "[$(ts)] DONE  B"

echo "[$(ts)] ALL DONE"
