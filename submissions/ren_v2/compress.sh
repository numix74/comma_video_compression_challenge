#!/usr/bin/env bash
# ren_v2/compress.sh
# Improvements over neural_inflate:
#   - CRF 33 → 36  (saves ~20-25% bitrate)
#   - chroma-qp-offset=6  (saves ~3-5% additional on chroma, safe because
#     PoseNet already subsamples chroma 2x via YUV6 format)
#   - keyint 180 → 240  (fewer I-frames, saves ~2%)
#   - same ROI denoise preprocessing as neural_inflate
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/ren_v2"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--jobs <n>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"
export IN_DIR ARCHIVE_DIR PD

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  PRE_IN="$IN"

  echo "→ ${IN}  →  ${OUT}"

  # Step 2: Downscale + AV1 encode
  # Key changes vs neural_inflate:
  #   crf 33 -> 36          (~20-25% smaller file)
  #   keyint 180 -> 240     (~2% smaller file, ~12s GOP at 20fps)
  #   tune=0 (PSNR mode)   PoseNet measures MSE → PSNR-optimised encode
  #   NOTE: chroma-qp-offset removed from svtav1-params (causes segfault)
  # Encoder priority: libsvtav1 (fast) → libaom-av1 (toujours disponible)
  FFMPEG="${PD}/ffmpeg-new"
  export LD_LIBRARY_PATH="${PD}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$FFMPEG" -version &>/dev/null || FFMPEG="ffmpeg"

  # Detect available AV1 encoder
  if "$FFMPEG" -encoders 2>/dev/null | grep -q libsvtav1; then
    AV1_ENCODER=libsvtav1
    AV1_OPTS=(-preset 0 -crf 44 -svtav1-params "film-grain=0:keyint=240:scd=0:tune=0")
  else
    AV1_ENCODER=libaom-av1
    AV1_OPTS=(-cpu-used 4 -crf 44 -b:v 0 -g 240 -tune psnr)
  fi

  "$FFMPEG" -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$PRE_IN" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -c:v "$AV1_ENCODER" "${AV1_OPTS[@]}" \
    -r 20 "$OUT"

' _ {}

# Copy the trained REN model into the archive (it will be found by inflate.py)
MODEL_SRC="${HERE}/ren_model.int8.bz2"
if [ -f "$MODEL_SRC" ]; then
  cp "$MODEL_SRC" "${ARCHIVE_DIR}/ren_model.int8.bz2"
  echo "Model copied to archive: $(du -h "$MODEL_SRC" | cut -f1)"
else
  echo "WARNING: ${MODEL_SRC} not found. Run train_ren.py first." >&2
  echo "         The inflate step will fail without the model." >&2
fi

# zip archive
cd "$ARCHIVE_DIR"
if command -v zip &>/dev/null; then
  zip -r "${HERE}/archive.zip" .
else
  python3 -c "
import zipfile, os
with zipfile.ZipFile('${HERE}/archive.zip', 'w', zipfile.ZIP_STORED) as zf:
    for f in os.listdir('.'):
        zf.write(f)
"
fi
echo "Compressed to ${HERE}/archive.zip"
SIZE_KB=$(du -k "${HERE}/archive.zip" | cut -f1)
echo "Archive size: ${SIZE_KB} KB"
