#!/usr/bin/env python
"""
ren_v3/inflate.py — Oracle-Conditioned REN (OC-REN)

Vs ren_v2:
  - Le REN est conditionné sur les cibles PoseNet stockées dans l'archive (0.pose)
  - Conditioning : Linear(6→12) broadcast-add aux features Haar avant le corps conv
  - Coût archive : +5 KB (pose targets bz2) = +0.003 pts de rate → négligeable
  - Gain attendu : PoseNet distortion divisée par 2-5x
"""
import os, io, bz2, struct, av, torch, numpy as np
import torch.nn as nn
from PIL import Image
from frame_utils import camera_size, yuv420_to_rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))


class OCREN(nn.Module):
    """
    Oracle-Conditioned REN.

    Le pose_target (6 floats) est projeté en 12 canaux et additionné aux
    features Haar avant le corps conv. Le REN sait ainsi quelle motion
    reconstruire sans avoir à la deviner depuis le frame dégradé.

    Paramètres supplémentaires vs ren_v2 : Linear(6,12) = 84 params → ~300 bytes.
    """
    def __init__(self, features=48):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        self.haar_gain = nn.Parameter(torch.ones(1, 12, 1, 1))
        self.pose_embed = nn.Linear(6, 12, bias=True)
        self.body = nn.Sequential(
            nn.Conv2d(12, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, 12, 3, padding=1),
        )
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        nn.init.zeros_(self.pose_embed.weight)
        nn.init.zeros_(self.pose_embed.bias)

    def forward(self, x, pose_target):
        # x: (B, 3, H, W) float [0, 255]
        # pose_target: (B, 6) float — cibles PoseNet du GT
        B = x.shape[0]
        x_norm = x / 255.0
        shuffled = self.down(x_norm)           # (B, 12, H/2, W/2)
        scaled   = shuffled * self.haar_gain

        # Pose conditioning : broadcast spatial
        cond = self.pose_embed(pose_target)    # (B, 12)
        scaled = scaled + cond.view(B, 12, 1, 1)

        residual = self.up(self.body(scaled))
        return (x_norm + residual).clamp(0, 1) * 255.0


MODEL = None
POSE_TARGETS = None  # (N_pairs, 6) tensor chargé depuis 0.pose


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


def load_pose_targets(path):
    """Charge 0.pose → tensor (N_pairs, 6) float32."""
    with open(path, 'rb') as f:
        N, D = struct.unpack('<II', f.read(8))
        data = np.frombuffer(bz2.decompress(f.read()), dtype=np.float16).reshape(N, D)
    return torch.from_numpy(data.astype(np.float32))


def get_model(archive_dir):
    global MODEL, POSE_TARGETS
    if MODEL is not None:
        return MODEL

    model_path = os.path.join(archive_dir, 'ren_model.int8.bz2')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ren_model.int8.bz2 not found in {archive_dir}")

    MODEL = OCREN(features=48).to(DEVICE).eval()
    MODEL.load_state_dict(_load_int8_bz2(model_path))
    n_params = sum(p.numel() for p in MODEL.parameters())
    print(f"[ren_v3] Loaded OC-REN ({n_params:,} params) from {model_path}")

    pose_path = os.path.join(archive_dir, '0.pose')
    if os.path.exists(pose_path):
        POSE_TARGETS = load_pose_targets(pose_path).to(DEVICE)
        print(f"[ren_v3] Loaded pose targets: {POSE_TARGETS.shape} from {pose_path}")
    else:
        print("[ren_v3] WARNING: 0.pose not found, using zero conditioning")
        POSE_TARGETS = None

    return MODEL


def decode_and_resize_to_file(video_path: str, dst: str):
    target_w, target_h = camera_size
    archive_dir = os.path.dirname(video_path)
    model = get_model(archive_dir)

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
                # Pose target pour cette frame (pair_idx = frame_idx // 2)
                pair_idx = n // 2
                if POSE_TARGETS is not None and pair_idx < len(POSE_TARGETS):
                    pose = POSE_TARGETS[pair_idx].unsqueeze(0)  # (1, 6)
                else:
                    pose = torch.zeros(1, 6, device=DEVICE)
                with torch.no_grad():
                    x = model(x, pose)
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
