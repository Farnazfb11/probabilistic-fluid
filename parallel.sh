#!/usr/bin/env bash
set -euo pipefail

BASE="/media/nvme7n1/fbtelki/dyldo_full/data"
P=4
STRAT="class"
TAG="collapsedlit_onlyid_numeric"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

OUTDIR="/media/nvme7n1/fbtelki/dyldo_full/summaries_${STRAT}_${TAG}_${RUN_TAG}"
LOGDIR="${OUTDIR}/log"
BUILD_SCRIPT="main_build.py"

mkdir -p "$OUTDIR" "$LOGDIR"

export BASE P STRAT TAG RUN_TAG OUTDIR LOGDIR BUILD_SCRIPT

find "$BASE" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | LC_ALL=C sort | \
xargs -I{} -P "$P" bash -lc '
week="{}"

D1="$BASE/$week/data.nq.gz"
D2="$BASE/$week/data-seedlist.nq.gz"

LOG="$LOGDIR/${week}.log"

if [[ ! -f "$D1" || ! -f "$D2" ]]; then
    echo "[SKIP] $week missing files" | tee -a "$LOG"
    exit 0
fi

echo "[START] $week" | tee -a "$LOG"

python3 -u "$BUILD_SCRIPT" \
  -i "$D1" "$D2" \
  -s "$STRAT" \
  -o "$OUTDIR/" \
  --limit 0 \
  --only-id \
  --use-numeric-ids \
  >> "$LOG" 2>&1

echo "[DONE] $week" | tee -a "$LOG"
