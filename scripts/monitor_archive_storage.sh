#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/llm/AnhNT/RiskGuard-AIGI"
DISK1_ROOT="/home/llm/RiskGuard-AIGI-data/genimage_disk1"
DISK2_ROOT="/home/llm/disk2/AnhNT/RiskGuard-AIGI-data/genimage_disk2"
LOG_DIR="${PROJECT_ROOT}/logs/genimage_download"
OUT_CSV="${LOG_DIR}/disk_usage_history.csv"
PAUSE_FILE="${LOG_DIR}/PAUSE_RESERVE_BREACH"
LOCK_FILE="${LOG_DIR}/monitor.lock"
RESERVE_BYTES="${GENIMAGE_RESERVE_BYTES:-161061273600}"
INTERVAL_SECONDS="${GENIMAGE_MONITOR_INTERVAL_SECONDS:-60}"
RUN_ONCE="false"

if [[ "${1:-}" == "--once" ]]; then
  RUN_ONCE="true"
fi

mkdir -p "$LOG_DIR" "$DISK1_ROOT" "$DISK2_ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

if [[ ! -f "$OUT_CSV" ]]; then
  printf 'timestamp,device,mount_point,total_bytes,used_bytes,available_bytes,disk1_downloaded_bytes,disk2_downloaded_bytes\n' > "$OUT_CSV"
fi

sum_downloaded_bytes() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    printf '0\n'
    return
  fi
  find "$root" -type f ! -name '*.rclone-meta.json' -printf '%s\n' \
    | awk '{sum += $1} END {printf "%.0f\n", sum + 0}'
}

while true; do
  timestamp="$(date -Iseconds)"
  disk1_downloaded_bytes="$(sum_downloaded_bytes "$DISK1_ROOT")"
  disk2_downloaded_bytes="$(sum_downloaded_bytes "$DISK2_ROOT")"

  while read -r device total used available mount_point; do
    [[ "$device" == "Filesystem" ]] && continue
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$timestamp" "$device" "$mount_point" "$total" "$used" "$available" \
      "$disk1_downloaded_bytes" "$disk2_downloaded_bytes" >> "$OUT_CSV"
    if (( available < RESERVE_BYTES )); then
      printf '%s reserve breach on %s: available_bytes=%s reserve_bytes=%s\n' \
        "$timestamp" "$mount_point" "$available" "$RESERVE_BYTES" >> "${LOG_DIR}/reserve_breach.log"
      touch "$PAUSE_FILE"
    fi
  done < <(df -B1 --output=source,size,used,avail,target / /home/llm/disk2 | tail -n +2)

  if [[ "$RUN_ONCE" == "true" ]]; then
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
