#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUBMISSION_DIR="${HERE}/submissions/baseline"
VIDEO_NAMES_FILE="${HERE}/public_test_video_names.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submission-dir|--submission_dir)
      SUBMISSION_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--submission-dir <dir>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

ARCHIVE_ZIP="${SUBMISSION_DIR}/archive.zip"
ARCHIVE_DIR="${SUBMISSION_DIR}/archive"
INFLATED_DIR="${SUBMISSION_DIR}/inflated"

if [ ! -f "$ARCHIVE_ZIP" ]; then
  echo "ERROR: ${ARCHIVE_ZIP} not found" >&2
  exit 1
fi

# unzip
rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"
unzip -o "$ARCHIVE_ZIP" -d "$ARCHIVE_DIR"

# inflate
cd "$HERE"
MODULE_PATH="${SUBMISSION_DIR#${HERE}/}"
python -m "${MODULE_PATH//\//.}.inflate" \
  --data-dir "$ARCHIVE_DIR" \
  --output-dir "$INFLATED_DIR" \
  --file-list "$VIDEO_NAMES_FILE"

# assert all videos have been inflated
MISSING=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  RAW_PATH="${INFLATED_DIR}/$(dirname "$line")/video.raw"
  if [ ! -f "$RAW_PATH" ]; then
    echo "ERROR: missing inflated file: ${RAW_PATH}" >&2
    MISSING=$((MISSING + 1))
  fi
done < "$VIDEO_NAMES_FILE"

if [ "$MISSING" -gt 0 ]; then
  echo "ERROR: ${MISSING} video(s) not inflated" >&2
  exit 1
fi

echo "All videos inflated to ${INFLATED_DIR}"

# evaluate
python "$HERE/evaluate.py" \
  --submission-dir "$SUBMISSION_DIR" \
  --uncompressed-dir "$HERE/test_videos" \
  --report "$SUBMISSION_DIR/report.txt" \
  --video-names-file "$VIDEO_NAMES_FILE"

echo "Evaluation complete. Report saved to ${SUBMISSION_DIR}/report.txt"
