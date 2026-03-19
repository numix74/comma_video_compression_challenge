#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/.." && pwd)"

IN_DIR="${PD}/test_videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
OUTDIR="${PD}/submission"
JOBS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    --out-dir|--out_dir)
      OUTDIR="${2%/}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--jobs <n>] [--video-names-file <file>] [--out-dir <dir>]" >&2
      exit 2 ;;
  esac
done

OUTDIR="$(mkdir -p "$OUTDIR" && cd "$OUTDIR" && pwd)"
TMPDIR="$(mktemp -d)"

export IN_DIR TMPDIR JOBS

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  OUT="${TMPDIR}/${rel}"
  mkdir -p "$(dirname "$OUT")"

  echo "→ ${IN}  →  ${OUT}"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -c:v libx265 -preset ultrafast -crf 30 \
    -g 1 -bf 0 -x265-params "keyint=1:min-keyint=1:scenecut=0:frame-threads=1:log-level=warning" \
    -r 20 -f hevc "$OUT"
' _ {}

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
(
  cd "$TMPDIR"
  cp -r . "$OUTDIR"
)
rm -rf "$TMPDIR"
echo "All done. Saved $OUTDIR"
