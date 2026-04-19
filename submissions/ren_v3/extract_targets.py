#!/usr/bin/env python
"""
Extrait les cibles PoseNet depuis la vidéo originale et les sauvegarde dans archive/.
Format 0.pose : header (N, D) uint32 + données float16 bz2
Taille : ~5-7 KB pour 600 paires → coût rate = +0.003 pts (négligeable)
"""
import os, sys, struct, bz2, argparse
import torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT)

from frame_utils import yuv420_to_rgb, camera_size
from modules import DistortionNet, posenet_sd_path, segnet_sd_path
import av


def extract_frames(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(yuv420_to_rgb(frame))
    container.close()
    return frames


def compute_pose_targets(frames, device, batch_size=16):
    dn = DistortionNet().eval().to(device)
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    posenet = dn.posenet

    n_frames = len(frames)
    n_pairs = n_frames // 2
    mH, mW = 384, 512

    all_poses = []
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        B = end - start

        batch = []
        for i in range(start, end):
            f0 = frames[i * 2].float().to(device).permute(2, 0, 1)
            f1 = frames[i * 2 + 1].float().to(device).permute(2, 0, 1)
            f0 = F.interpolate(f0.unsqueeze(0), size=(mH, mW), mode='bilinear', align_corners=False).squeeze(0)
            f1 = F.interpolate(f1.unsqueeze(0), size=(mH, mW), mode='bilinear', align_corners=False).squeeze(0)
            batch.append(torch.stack([f0, f1]))  # (2, 3, mH, mW)

        pair_batch = torch.stack(batch)  # (B, 2, 3, mH, mW)

        with torch.no_grad():
            # rgb_to_yuv6 logic inline (matches modules.py)
            from frame_utils import rgb_to_yuv6
            x = pair_batch.view(B * 2, 3, mH, mW)
            yuv = rgb_to_yuv6(x)  # (B*2, 6, mH//2, mW//2)
            yuv = yuv.view(B, 12, mH // 2, mW // 2)
            out = posenet(yuv)
            poses = out['pose'][:, :6].cpu()  # (B, 6)

        all_poses.append(poses)
        if (start // batch_size) % 10 == 0:
            print(f"  Pairs {start}-{end}/{n_pairs}")

    return torch.cat(all_poses, dim=0)  # (N_pairs, 6)


def save_pose_targets(poses, path):
    N, D = poses.shape
    data = poses.numpy().astype(np.float16)
    compressed = bz2.compress(data.tobytes(), compresslevel=9)
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', N, D))
        f.write(compressed)
    size_kb = os.path.getsize(path) / 1024
    rate_cost = (size_kb / 1024) / 37.5 * 25
    print(f"  Saved {path}: {size_kb:.1f} KB → coût rate +{rate_cost:.4f} pts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=os.path.join(ROOT, 'videos', '0.mkv'))
    parser.add_argument('--archive-dir', default=os.path.join(HERE, 'archive'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.archive_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Extracting frames from {args.video}...")
    frames = extract_frames(args.video)
    print(f"  {len(frames)} frames")

    print(f"Computing PoseNet targets (device={device})...")
    poses = compute_pose_targets(frames, device)
    print(f"  {poses.shape[0]} pose targets computed")

    base = os.path.splitext(os.path.basename(args.video))[0]
    save_pose_targets(poses, os.path.join(args.archive_dir, f'{base}.pose'))
    print("Done.")


if __name__ == '__main__':
    main()
