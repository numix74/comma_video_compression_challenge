#!/usr/bin/env python
"""
adv_frames/inflate.py — Décompression triviale : décodage AV1 pur.
Aucun modèle neuronal. Aucun GPU requis.
"""
import sys, os
import av
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from frame_utils import camera_size

W, H = camera_size  # 1164 × 874


def inflate(src: str, dst: str):
    container = av.open(src)
    stream = container.streams.video[0]
    n = 0
    with open(dst, "wb") as f:
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            # Resize si nécessaire (ne devrait pas arriver)
            if arr.shape[1] != W or arr.shape[0] != H:
                from PIL import Image
                arr = np.array(Image.fromarray(arr).resize((W, H), Image.LANCZOS))
            f.write(arr.tobytes())
            n += 1
    container.close()
    return n


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = inflate(src, dst)
    print(f"saved {n} frames → {dst}")
