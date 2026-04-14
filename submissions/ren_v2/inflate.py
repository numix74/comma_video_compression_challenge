#!/usr/bin/env python
"""
ren_v2/inflate.py — Residual Enhancement Network v2

Improvements over neural_inflate:
  - features: 32 → 48  (~37K params, ~50 KB int8.bz2, cost < 0.02 pts)
  - 4 conv layers instead of 3  (more expressivity for CRF-36 artifacts)
  - LeakyReLU(0.1) instead of ReLU  (better gradient flow; no dead neurons)
  - Same int8.bz2 / f16.bz2 / raw loading logic as neural_inflate

The REN is trained specifically on CRF-36 compressed frames (not CRF-33).
Using a CRF-33 REN on CRF-36 input would be suboptimal.
"""
import os, io, bz2, struct, av, torch, numpy as np
import torch.nn as nn
from PIL import Image
from frame_utils import camera_size, yuv420_to_rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))


class REN(nn.Module):
    """
    Residual Enhancement Network v2.

    Architecture: PixelUnshuffle(2) → HaarGain → 4×Conv(features) → PixelShuffle(2)
    - Works in pixel-shuffled space (stride-2 subpixels) for efficiency
    - Residual connection: output = input + learned_correction
    - LeakyReLU avoids dead neurons on negative residuals
    - HaarGain: per-channel trainable gain before the CNN body.
      AV1 compression attenuates high-frequency Haar channels (y10, y01, y11)
      more than y00 (mean). After PixelUnshuffle(2), the 12 channels are
      {R,G,B} × {y00, y10, y01, y11}. A learned gain amplifies the HF channels
      that PoseNet relies on (especially y10/y01 = horizontal/vertical gradients).
      Cost: +12 parameters (~48 bytes in int8.bz2 — negligible).
    """
    def __init__(self, features=48):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        # Per-channel gain applied after PixelUnshuffle, before the CNN body.
        # Initialised to 1.0 (identity). Learns to amplify HF Haar channels.
        self.haar_gain = nn.Parameter(torch.ones(1, 12, 1, 1))
        self.body = nn.Sequential(
            nn.Conv2d(12, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, 12, 3, padding=1),
        )
        self.up = nn.PixelShuffle(2)
        # Zero-init the last layer so training starts from identity
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        x_norm = x / 255.0
        shuffled = self.down(x_norm)           # (B, 12, H/2, W/2)
        scaled   = shuffled * self.haar_gain   # per-channel amplification
        residual = self.up(self.body(scaled))  # (B, 3, H, W)
        return (x_norm + residual).clamp(0, 1) * 255.0


MODEL = None


def _load_f16_bz2(path):
    with open(path, 'rb') as f:
        data = bz2.decompress(f.read())
    sd = torch.load(io.BytesIO(data), map_location=DEVICE, weights_only=True)
    return {k: v.float() for k, v in sd.items()}


def _load_int8_bz2(path):
    with open(path, 'rb') as f:
        raw = bz2.decompress(f.read())
    buf = io.BytesIO(raw)
    n_tensors = struct.unpack('<I', buf.read(4))[0]
    sd = {}
    for _ in range(n_tensors):
        name_len = struct.unpack('<I', buf.read(4))[0]
        name = buf.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<I', buf.read(4))[0]
        shape = [struct.unpack('<I', buf.read(4))[0] for _ in range(n_dims)]
        scale = struct.unpack('<f', buf.read(4))[0]
        data_len = struct.unpack('<I', buf.read(4))[0]
        data = np.frombuffer(buf.read(data_len), dtype=np.int8)
        sd[name] = torch.from_numpy(data.astype(np.float32)).reshape(shape) * scale
    return sd


def get_model(archive_dir=None):
    global MODEL
    if MODEL is not None:
        return MODEL
    candidates = []
    for d in ([archive_dir] if archive_dir else []) + [os.path.join(HERE, 'archive'), HERE]:
        candidates.append((os.path.join(d, 'ren_model.int8.bz2'), 'int8'))
        candidates.append((os.path.join(d, 'ren_model.pt.bz2'), 'f16'))
        candidates.append((os.path.join(d, 'ren_model.pt'), 'raw'))
    for path, fmt in candidates:
        if os.path.exists(path):
            MODEL = REN(features=48).to(DEVICE).eval()
            if fmt == 'int8':
                MODEL.load_state_dict(_load_int8_bz2(path))
            elif fmt == 'f16':
                MODEL.load_state_dict(_load_f16_bz2(path))
            else:
                MODEL.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            n_params = sum(p.numel() for p in MODEL.parameters())
            print(f"[ren_v2] Loaded REN ({n_params:,} params) from {path}")
            return MODEL
    raise FileNotFoundError(
        "ren_model not found. Searched: " + ", ".join(p for p, _ in candidates)
    )


def decode_and_resize_to_file(video_path: str, dst: str):
    target_w, target_h = camera_size
    fmt = 'hevc' if video_path.endswith('.hevc') else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    n = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)
            H, W, _ = t.shape
            if H != target_h or W != target_w:
                pil = Image.fromarray(t.numpy())
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    x = get_model(os.path.dirname(video_path))(x)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    n = decode_and_resize_to_file(src, dst)
    print(f"saved {n} frames → {dst}")
