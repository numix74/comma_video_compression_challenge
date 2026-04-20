#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="$1"; OUTPUT_DIR="$2"; FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.mkv"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  printf "Inflating %s ... " "$line"
  cd "$ROOT"
  python3 -m submissions.adv_frames.inflate "$SRC" "$DST"
  echo "done"
done < "$FILE_LIST"
