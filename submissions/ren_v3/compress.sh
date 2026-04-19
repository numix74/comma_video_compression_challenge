#!/usr/bin/env bash
# ren_v3/compress.sh — OC-REN : AV1 + pose targets oracle
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/ren_v3"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)        IN_DIR="${2%/}"; shift 2 ;;
    --jobs)                   JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file) VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR" "$TMP_DIR"

export IN_DIR ARCHIVE_DIR PD

# Step 1: Encoder chaque vidéo en AV1
head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"; [[ -z "$rel" ]] && exit 0
  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  PRE_IN="'"${TMP_DIR}"'/${BASE}.pre.mkv"

  echo "→ ${IN}  →  ${OUT}"

  python3 "'"${HERE}"'/../neural_inflate/preprocess.py" \
    --input "$IN" --output "$PRE_IN" \
    --outside-luma-denoise 2.5 --outside-chroma-mode medium \
    --feather-radius 24 --outside-blend 0.50

  FFMPEG="${PD}/ffmpeg-new"
  export LD_LIBRARY_PATH="${PD}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$FFMPEG" -version &>/dev/null || FFMPEG="ffmpeg"

  if "$FFMPEG" -encoders 2>/dev/null | grep -q libsvtav1; then
    AV1_ENCODER=libsvtav1
    AV1_OPTS=(-preset 0 -crf 36 -svtav1-params "film-grain=22:keyint=240:scd=0:tune=0")
  else
    AV1_ENCODER=libaom-av1
    AV1_OPTS=(-cpu-used 4 -crf 36 -b:v 0 -g 240 -tune psnr)
  fi

  "$FFMPEG" -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$PRE_IN" \
    -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
    -pix_fmt yuv420p -c:v "$AV1_ENCODER" "${AV1_OPTS[@]}" \
    -r 20 "$OUT"

  rm -f "$PRE_IN"
' _ {}

# Step 2: Extraire les cibles PoseNet oracle depuis la vidéo originale
echo "Extracting PoseNet oracle targets..."
python3 "${HERE}/extract_targets.py" \
  --video "${IN_DIR}/0.mkv" \
  --archive-dir "$ARCHIVE_DIR"

# Step 3: Copier le modèle OC-REN entraîné
MODEL_SRC="${HERE}/ren_model.int8.bz2"
if [ -f "$MODEL_SRC" ]; then
  cp "$MODEL_SRC" "${ARCHIVE_DIR}/ren_model.int8.bz2"
  echo "Model: $(du -h "$MODEL_SRC" | cut -f1)"
else
  echo "WARNING: ${MODEL_SRC} not found. Run train_ren.py first." >&2
fi

# Step 4: Zipper l'archive
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

echo "Archive: ${HERE}/archive.zip ($(du -k "${HERE}/archive.zip" | cut -f1) KB)"
echo "Contents:"
ls -lh "${ARCHIVE_DIR}/"
