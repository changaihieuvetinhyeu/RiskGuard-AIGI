#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/llm/AnhNT/RiskGuard-AIGI"
GENIMAGE_ID="${GENIMAGE_ID:-1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS}"
ASSIGNMENT_CSV="${PROJECT_ROOT}/artifacts/genimage_two_disk_assignment.csv"
LOG_DIR="${PROJECT_ROOT}/logs/genimage_download"
QUEUE_TSV="${LOG_DIR}/download_queue.tsv"
PAUSE_FILE="${LOG_DIR}/PAUSE_RESERVE_BREACH"
STATE_FILE="${LOG_DIR}/download_state.csv"
RESERVE_BYTES="${GENIMAGE_RESERVE_BYTES:-161061273600}"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

if [[ ! -f "$STATE_FILE" ]]; then
  printf 'timestamp,event,assigned_disk,remote_path,remote_object_id,destination,size_bytes,message\n' > "$STATE_FILE"
fi

scripts/monitor_archive_storage.sh &
MONITOR_PID="$!"

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

python3 - "$ASSIGNMENT_CSV" "$QUEUE_TSV" <<'PY'
import csv
import re
import sys
from pathlib import Path

assignment_csv = Path(sys.argv[1])
queue_tsv = Path(sys.argv[2])

def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

with assignment_csv.open(newline="", encoding="utf-8") as handle, queue_tsv.open("w", encoding="utf-8") as out:
    rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (row["assigned_disk"], row["generator"], row["remote_path"], row["remote_object_id"]))
    for row in rows:
        log_file = Path("logs/genimage_download") / f"{safe(row['archive_family'])}.log"
        fields = [
            row["generator"],
            row["archive_family"],
            row["remote_path"],
            row["size_bytes"],
            row["assigned_disk"],
            row["physical_destination"],
            row["remote_object_id"],
            row.get("remote_md5", ""),
            row.get("duplicate_remote_path", "false"),
            str(log_file),
        ]
        out.write("\t".join(fields) + "\n")
PY

total_files="$(wc -l < "$QUEUE_TSV" | tr -d ' ')"
current_index=0

meta_matches() {
  local meta_path="$1"
  local remote_path="$2"
  local remote_object_id="$3"
  local size_bytes="$4"
  local remote_md5="$5"
  python3 - "$meta_path" "$remote_path" "$remote_object_id" "$size_bytes" "$remote_md5" <<'PY'
import json
import sys
from pathlib import Path

meta_path, remote_path, remote_object_id, size_bytes, remote_md5 = sys.argv[1:]
path = Path(meta_path)
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ok = (
    data.get("remote_path") == remote_path
    and data.get("remote_object_id") == remote_object_id
    and str(data.get("size_bytes")) == size_bytes
    and data.get("remote_md5", "") == remote_md5
)
raise SystemExit(0 if ok else 1)
PY
}

write_meta() {
  local meta_path="$1"
  local remote_path="$2"
  local remote_object_id="$3"
  local size_bytes="$4"
  local remote_md5="$5"
  local assigned_disk="$6"
  local destination="$7"
  python3 - "$meta_path" "$remote_path" "$remote_object_id" "$size_bytes" "$remote_md5" "$assigned_disk" "$destination" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

meta_path, remote_path, remote_object_id, size_bytes, remote_md5, assigned_disk, destination = sys.argv[1:]
payload = {
    "remote_path": remote_path,
    "remote_object_id": remote_object_id,
    "size_bytes": int(size_bytes),
    "remote_md5": remote_md5,
    "assigned_disk": assigned_disk,
    "destination": destination,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
Path(meta_path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
}

log_state() {
  local event="$1"
  local assigned_disk="$2"
  local remote_path="$3"
  local remote_object_id="$4"
  local destination="$5"
  local size_bytes="$6"
  local message="$7"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -Iseconds)" "$event" "$assigned_disk" "$remote_path" "$remote_object_id" \
    "$destination" "$size_bytes" "$message" >> "$STATE_FILE"
}

check_reserve_before_file() {
  local assigned_disk="$1"
  local destination="$2"
  local size_bytes="$3"
  local mount_point="/"
  if [[ "$assigned_disk" == "disk2" ]]; then
    mount_point="/home/llm/disk2"
  fi
  local available
  available="$(df -B1 --output=avail "$mount_point" | tail -n 1 | tr -dc '0-9')"
  local required="$size_bytes"
  if [[ -f "$destination" ]]; then
    local current_size
    current_size="$(stat -c '%s' "$destination")"
    if (( current_size < size_bytes )); then
      required=$(( size_bytes - current_size ))
    fi
  fi
  if (( available - required < RESERVE_BYTES )); then
    touch "$PAUSE_FILE"
    log_state "pause_reserve_risk" "$assigned_disk" "" "" "$destination" "$size_bytes" \
      "available_bytes=${available};required_bytes=${required};reserve_bytes=${RESERVE_BYTES}"
    return 1
  fi
}

while IFS=$'\t' read -r generator archive_family remote_path size_bytes assigned_disk destination remote_object_id remote_md5 duplicate_remote_path log_file; do
  current_index=$((current_index + 1))
  if [[ -f "$PAUSE_FILE" ]]; then
    log_state "paused" "$assigned_disk" "$remote_path" "$remote_object_id" "$destination" "$size_bytes" \
      "pause_file_present=${PAUSE_FILE}"
    exit 75
  fi

  mkdir -p "$(dirname "$destination")" "$(dirname "$log_file")"
  meta_path="${destination}.rclone-meta.json"

  if [[ -f "$destination" ]]; then
    actual_size="$(stat -c '%s' "$destination")"
    if [[ "$actual_size" == "$size_bytes" ]]; then
      if meta_matches "$meta_path" "$remote_path" "$remote_object_id" "$size_bytes" "$remote_md5"; then
        log_state "skip_existing" "$assigned_disk" "$remote_path" "$remote_object_id" "$destination" "$size_bytes" \
          "index=${current_index}/${total_files};metadata_match=true"
        continue
      fi
      if [[ -n "$remote_md5" ]]; then
        local_md5="$(md5sum "$destination" | awk '{print $1}')"
        if [[ "$local_md5" == "$remote_md5" ]]; then
          write_meta "$meta_path" "$remote_path" "$remote_object_id" "$size_bytes" "$remote_md5" "$assigned_disk" "$destination"
          log_state "skip_existing" "$assigned_disk" "$remote_path" "$remote_object_id" "$destination" "$size_bytes" \
            "index=${current_index}/${total_files};md5_match=true"
          continue
        fi
      fi
    fi
  fi

  check_reserve_before_file "$assigned_disk" "$destination" "$size_bytes"
  log_state "start" "$assigned_disk" "$remote_path" "$remote_object_id" "$destination" "$size_bytes" \
    "index=${current_index}/${total_files};archive_family=${archive_family};generator=${generator}"

  common_flags=(
    --drive-root-folder-id "$GENIMAGE_ID"
    --transfers 2
    --checkers 4
    --drive-chunk-size 64M
    --retries 20
    --low-level-retries 50
    --timeout 10m
    --contimeout 60s
    --stats 30s
    --stats-one-line
    --checksum
    --log-level INFO
    --log-file "$log_file"
  )

  if [[ "$duplicate_remote_path" == "true" ]]; then
    rclone backend copyid dmhung: "$remote_object_id" "$destination" "${common_flags[@]}"
  else
    rclone copyto "dmhung:${remote_path}" "$destination" "${common_flags[@]}"
  fi

  actual_size="$(stat -c '%s' "$destination")"
  if [[ "$actual_size" != "$size_bytes" ]]; then
    log_state "size_mismatch_after_copy" "$assigned_disk" "$remote_path" "$remote_object_id" "$destination" "$size_bytes" \
      "actual_size=${actual_size}"
    exit 1
  fi
  write_meta "$meta_path" "$remote_path" "$remote_object_id" "$size_bytes" "$remote_md5" "$assigned_disk" "$destination"
  log_state "complete" "$assigned_disk" "$remote_path" "$remote_object_id" "$destination" "$size_bytes" \
    "index=${current_index}/${total_files}"
done < "$QUEUE_TSV"

find \
  /home/llm/RiskGuard-AIGI-data/genimage_disk1 \
  /home/llm/disk2/AnhNT/RiskGuard-AIGI-data/genimage_disk2 \
  -type f ! -name '*.rclone-meta.json' \
  -printf '%p,%s\n' \
  | sort \
  > artifacts/genimage_local_two_disk_inventory.csv

rclone size dmhung: \
  --drive-root-folder-id "$GENIMAGE_ID" \
  --fast-list \
  | tee artifacts/genimage_remote_size_after_download.txt

python3 scripts/verify_archive_downloads.py
