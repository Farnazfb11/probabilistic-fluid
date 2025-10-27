set -Eeuo pipefail
source .venv/bin/activate || { echo "[fatal] venv not found"; exit 1; }
export VENV_PY="$(which python)"
export BASE="/media/nvme7n1/fbtelki/dyldo_full/data"

# make top-level dirs once (ok if exist)
mkdir -p summaries logs

# reduce parallelism a bit to avoid OOM; adjust if stable
PARALLEL=8

find "$BASE" -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]{4}-[0-9]{2}-[0-9]{2}$' \
| sort \
| xargs -I{} -P "$PARALLEL" bash -lc '
  set -Eeuo pipefail
  # ensure dirs exist in this subshell BEFORE using tee
  mkdir -p summaries logs

  W=$(basename "{}")
  IN1="{}/data.nq.gz"
  IN2="{}/data-seedlist.nq.gz"
  OUT="summaries/${W}_attributeclass_onlyid.json"
  LOG="logs/${W}.log"

  # Skip if already good (nodes only; no edges saved)
  if [[ -s "$OUT" ]] && jq -e ".nodes|length>0" "$OUT" >/dev/null 2>&1; then
    echo "[$W] already done"
    exit 0
  fi

  # Missing/corrupt handling — IMPORTANT: blank missing paths too
  if [[ ! -f "$IN1" ]]; then echo "[$W] missing $IN1" | tee -a logs/_missing.log; IN1=""; fi
  if [[ ! -f "$IN2" ]]; then echo "[$W] missing $IN2" | tee -a logs/_missing.log; IN2=""; fi
  if [[ -n "$IN1" ]] && ! gzip -t "$IN1" >/dev/null 2>&1; then echo "[$W] CORRUPT: $IN1" | tee -a logs/_corrupt.log; IN1=""; fi
  if [[ -n "$IN2" ]] && ! gzip -t "$IN2" >/dev/null 2>&1; then echo "[$W] CORRUPT: $IN2" | tee -a logs/_corrupt.log; IN2=""; fi

  # If both inputs are now empty, skip
  [[ -n "$IN1" || -n "$IN2" ]] || { echo "[$W] all inputs missing/corrupt → skip"; exit 0; }

  # Build argv safely
  args=( --input )
  [[ -n "$IN1" ]] && args+=( "$IN1" )
  [[ -n "$IN2" ]] && args+=( "$IN2" )

  echo "[$W] starting" | tee "$LOG"

  # Run with venv python
  "'"$VENV_PY"'" main_build.py \
      "${args[@]}" \
      --strategy attributeclass \
      --out summaries/ \
      --only-id --use-numeric-ids --limit 0 \
      >>"$LOG" 2>&1

  rc=$?
  if (( rc == 0 )); then
    if jq -e ".nodes|length==0" "$OUT" >/dev/null 2>&1; then
      echo "[$W] ⚠ empty payload (0 nodes) — will redo" | tee -a logs/_empty_payload.log
    else
      echo "[$W] ✔ done"
    fi
  else
    echo "[$W] ✖ failed (rc=$rc) — see $LOG"
  fi
'
