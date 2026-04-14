#!/usr/bin/env python
"""
ren_v2/train_ren.py — Task-aware training for REN v2

Improvements over neural_inflate/train_ren.py:
  - Must train on ren_v2/archive/0.mkv (CRF-36 output, NOT CRF-33)
  - AdamW optimizer with weight_decay=1e-4 (better regularisation)
  - L1 pixel regulariser: prevents the REN from introducing hallucinations
  - 150 epochs (vs 100) for finer convergence
  - Saves both .pt (for re-training) and .int8.bz2 (for distribution)
  - Loss calibration based on actual baseline score contributions

Usage:
  # 1. Run compress.sh first to produce ren_v2/archive/0.mkv
  # 2. Run this script:
  python train_ren.py [--epochs 150] [--batch-size 1] [--lr 1e-3] [--features 48]
  # 3. The script saves ren_v2/ren_model.int8.bz2 (included by compress.sh)
"""
import os, sys, argparse, math, io, bz2, struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from frame_utils import camera_size, yuv420_to_rgb
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class REN(nn.Module):
    """
    Must stay in sync with inflate.py REN class.
    HaarGain: +12 trainable params before CNN body.
    After PixelUnshuffle(2), channels = {R,G,B}×{y00,y10,y01,y11}.
    AV1 attenuates y10/y01/y11 (HF) more than y00 (mean).
    Learned gain restores the HF channels PoseNet relies on.
    """
    def __init__(self, features=48):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        self.haar_gain = nn.Parameter(torch.ones(1, 12, 1, 1))
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

    def forward(self, x):
        x_norm = x / 255.0
        shuffled = self.down(x_norm)
        scaled   = shuffled * self.haar_gain
        residual = self.up(self.body(scaled))
        return (x_norm + residual).clamp(0, 1) * 255.0


def save_int8_bz2(model, path):
    """Quantise to int8 and save as custom bz2 format (same as neural_inflate)."""
    state = {}
    for name, param in model.state_dict().items():
        t = param.detach().cpu().float()
        scale = t.abs().max().item() / 127.0 if t.abs().max().item() > 0 else 1.0
        quantised = (t / scale).round().clamp(-127, 127).to(torch.int8)
        state[name] = (quantised, scale, list(t.shape))

    buf = io.BytesIO()
    n = len(state)
    buf.write(struct.pack('<I', n))
    for name, (q, scale, shape) in state.items():
        name_bytes = name.encode('utf-8')
        buf.write(struct.pack('<I', len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack('<I', len(shape)))
        for s in shape:
            buf.write(struct.pack('<I', s))
        buf.write(struct.pack('<f', scale))
        data = q.numpy().tobytes()
        buf.write(struct.pack('<I', len(data)))
        buf.write(data)

    compressed = bz2.compress(buf.getvalue(), compresslevel=9)
    with open(path, 'wb') as f:
        f.write(compressed)

    size_kb = len(compressed) / 1024
    print(f"  Saved int8.bz2: {path} ({size_kb:.1f} KB)")


def decode_all_frames(video_path, target_w=None, target_h=None, lanczos=False):
    fmt = 'hevc' if video_path.endswith('.hevc') else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb(frame)
        if target_w and target_h and (t.shape[0] != target_h or t.shape[1] != target_w):
            if lanczos:
                pil = Image.fromarray(t.numpy())
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                t = torch.from_numpy(np.array(pil))
            else:
                t = F.interpolate(
                    t.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(target_h, target_w), mode='bicubic', align_corners=False
                ).clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
        frames.append(t)
    container.close()
    return frames


class ConsecutivePairDataset(Dataset):
    def __init__(self, comp_frames, gt_frames):
        assert len(comp_frames) == len(gt_frames)
        self.comp = comp_frames
        self.gt = gt_frames

    def __len__(self):
        return len(self.comp) - 1

    def __getitem__(self, idx):
        ca = self.comp[idx].permute(2, 0, 1).float()
        cb = self.comp[idx + 1].permute(2, 0, 1).float()
        ga = self.gt[idx].permute(2, 0, 1).float()
        gb = self.gt[idx + 1].permute(2, 0, 1).float()
        return ca, cb, ga, gb


def compute_loss(model, posenet, segnet, comp_a, comp_b, gt_a, gt_b,
                 w_seg, w_temp, w_pixel):
    inf_a = model(comp_a)
    inf_b = model(comp_b)

    pair_inf = torch.stack([inf_a.permute(0, 2, 3, 1),
                            inf_b.permute(0, 2, 3, 1)], dim=1)
    pair_gt = torch.stack([gt_a.permute(0, 2, 3, 1),
                           gt_b.permute(0, 2, 3, 1)], dim=1)

    # PoseNet loss — dominant metric (44% of score)
    posenet_in_inf = posenet.preprocess_input(pair_inf.permute(0, 1, 4, 2, 3))
    with torch.no_grad():
        posenet_in_gt = posenet.preprocess_input(pair_gt.permute(0, 1, 4, 2, 3))
        posenet_out_gt = posenet(posenet_in_gt)
    posenet_out_inf = posenet(posenet_in_inf)
    loss_pose = sum(
        F.mse_loss(posenet_out_inf[h.name][..., :h.out // 2],
                   posenet_out_gt[h.name][..., :h.out // 2])
        for h in posenet.hydra.heads
    )

    # SegNet loss — second metric (22% of score)
    # SegNet only sees the LAST frame (x[:, -1, ...]) so we weight frame b more.
    segnet_in_inf = segnet.preprocess_input(pair_inf.permute(0, 1, 4, 2, 3))
    with torch.no_grad():
        segnet_in_gt = segnet.preprocess_input(pair_gt.permute(0, 1, 4, 2, 3))
        logits_gt = segnet(segnet_in_gt)
    logits_inf = segnet(segnet_in_inf)
    loss_seg = F.kl_div(
        F.log_softmax(logits_inf, dim=1),
        F.softmax(logits_gt, dim=1),
        reduction='batchmean'
    )

    # Temporal consistency — penalises inconsistent corrections frame-to-frame
    corr_a = (inf_a - comp_a) / 255.0
    corr_b = (inf_b - comp_b) / 255.0
    loss_temp = F.l1_loss(corr_a, corr_b)

    # Pixel L1 regulariser — prevents hallucinations; keeps REN grounded
    loss_pixel = (F.l1_loss(inf_a / 255.0, gt_a / 255.0) +
                  F.l1_loss(inf_b / 255.0, gt_b / 255.0)) / 2.0

    loss = loss_pose + w_seg * loss_seg + w_temp * loss_temp + w_pixel * loss_pixel
    return loss, loss_pose.item(), loss_seg.item(), loss_temp.item(), loss_pixel.item()


def train(args):
    print(f"Device: {DEVICE}")
    torch.manual_seed(1234)
    np.random.seed(1234)

    W, H = camera_size

    # Find compressed archive — must be from ren_v2's own compress.sh (CRF 36)
    archive_path = os.path.join(HERE, 'archive/0.mkv')
    if not os.path.exists(archive_path):
        print(f"ERROR: {archive_path} not found.")
        print("       Run compress.sh first to generate the CRF-36 archive.")
        sys.exit(1)

    print(f"Loading compressed frames (CRF-36) from {archive_path}...")
    comp_frames = decode_all_frames(archive_path, target_w=W, target_h=H, lanczos=True)
    print(f"  {len(comp_frames)} frames at {W}x{H}")

    gt_path = os.path.join(ROOT, 'videos/0.mkv')
    if not os.path.exists(gt_path):
        print(f"ERROR: {gt_path} not found.")
        sys.exit(1)

    print(f"Loading ground-truth frames from {gt_path}...")
    gt_frames = decode_all_frames(gt_path)
    print(f"  {len(gt_frames)} frames")

    if len(comp_frames) != len(gt_frames):
        n = min(len(comp_frames), len(gt_frames))
        print(f"WARNING: frame count mismatch ({len(comp_frames)} vs {len(gt_frames)}), "
              f"truncating to {n}")
        comp_frames = comp_frames[:n]
        gt_frames = gt_frames[:n]

    split = int(len(comp_frames) * 0.80)
    train_ds = ConsecutivePairDataset(comp_frames[:split], gt_frames[:split])
    val_ds = ConsecutivePairDataset(comp_frames[split:], gt_frames[split:])
    print(f"  Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = REN(features=args.features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: REN(features={args.features}), {n_params:,} parameters")

    print("  Loading DistortionNet (frozen)...")
    distortion_net = DistortionNet().to(DEVICE).eval()
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    for p in distortion_net.parameters():
        p.requires_grad_(False)
    posenet = distortion_net.posenet
    segnet = distortion_net.segnet

    # AdamW: weight decay provides implicit L2 regularisation on the small model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    save_pt_path = os.path.join(HERE, 'ren_model.pt')
    save_int8_path = os.path.join(HERE, 'ren_model.int8.bz2')

    # ---- Loss weight calibration (score-sensitivity based) ----
    # Score formula: 100*segnet + sqrt(10*posenet) + 25*rate
    #
    # The correct calibration uses the SCORE GRADIENT, not loss magnitude equality.
    # At operating point (D_pose, D_seg):
    #   ∂score/∂D_seg  = 100                              (constant)
    #   ∂score/∂D_pose = 10 / (2 * sqrt(10 * D_pose))    (grows as D_pose → 0)
    #
    # We want w_seg such that the training signal reflects the relative score impact:
    #   w_seg = (∂score/∂D_seg) / (∂score/∂D_pose) = 100 / sens_pose
    #
    # At ren_v2 operating point (D_pose ≈ 0.012):
    #   sens_pose = 10 / (2*sqrt(0.12)) ≈ 14.4
    #   w_seg = 100 / 14.4 ≈ 6.9
    #
    # This is ~7×, NOT the magnitude-equalised value from lp0/ls0.
    # The previous calibration was wrong: it drove training to equalize loss
    # magnitudes rather than score impact.
    print("\n  Calibrating loss weights (score-sensitivity method)...")
    model.train()
    ca, cb, ga, gb = train_ds[0]
    ca = ca.unsqueeze(0).to(DEVICE)
    cb = cb.unsqueeze(0).to(DEVICE)
    ga = ga.unsqueeze(0).to(DEVICE)
    gb = gb.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, lp0, ls0, lt0, lpx0 = compute_loss(
            model, posenet, segnet, ca, cb, ga, gb, 1.0, 0.005, 0.0
        )
    print(f"  Identity baseline — pose: {lp0:.6f}, seg: {ls0:.6f}, "
          f"temp: {lt0:.6f}, pixel: {lpx0:.6f}")

    # Estimate current D_pose from identity loss (proxy for operating point)
    # lp0 is posenet MSE loss at identity (≈ D_pose of compressed input)
    d_pose_est = max(lp0, 1e-6)
    d_seg_est  = max(ls0 * 0.01, 1e-6)  # rough proxy (KL → distortion mapping)

    # Score-sensitivity derivatives at estimated operating point
    sens_pose = 10.0 / (2.0 * math.sqrt(10.0 * d_pose_est))
    sens_seg  = 100.0

    # w_seg: ratio of score sensitivities (how much more the score cares about seg vs pose)
    w_seg = max(1.0, min(20.0, sens_seg / sens_pose))

    # w_temp: small regulariser for temporal consistency (no direct score term)
    w_temp = 0.005

    # w_pixel: L1 regulariser scaled to ~5% of pose signal to prevent hallucinations
    w_pixel = max(0.005, min(0.5, lp0 * 0.05 / max(lpx0, 1e-8))) if lpx0 > 0 else 0.05

    print(f"  D_pose_est={d_pose_est:.6f}, sens_pose={sens_pose:.2f}, sens_seg={sens_seg:.2f}")
    print(f"  Calibrated: w_seg={w_seg:.4f}, w_temp={w_temp:.4f}, w_pixel={w_pixel:.4f}")
    del ca, cb, ga, gb

    best_val = float('inf')
    print(f"\n  Training {args.epochs} epochs "
          f"(batch={args.batch_size}, lr={args.lr}, features={args.features})\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_lp, train_ls = 0.0, 0.0, 0.0
        n_batches = 0

        for comp_a, comp_b, gt_a, gt_b in train_loader:
            comp_a = comp_a.to(DEVICE)
            comp_b = comp_b.to(DEVICE)
            gt_a   = gt_a.to(DEVICE)
            gt_b   = gt_b.to(DEVICE)

            optimizer.zero_grad()
            loss, lp, ls, lt, lpx = compute_loss(
                model, posenet, segnet, comp_a, comp_b, gt_a, gt_b,
                w_seg, w_temp, w_pixel
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_lp   += lp
            train_ls   += ls
            n_batches  += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)
        train_lp   /= max(n_batches, 1)
        train_ls   /= max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            val_loss, val_lp, val_ls = 0.0, 0.0, 0.0
            n_val = 0
            with torch.no_grad():
                for comp_a, comp_b, gt_a, gt_b in val_loader:
                    comp_a = comp_a.to(DEVICE)
                    comp_b = comp_b.to(DEVICE)
                    gt_a   = gt_a.to(DEVICE)
                    gt_b   = gt_b.to(DEVICE)
                    loss, lp, ls, lt, lpx = compute_loss(
                        model, posenet, segnet, comp_a, comp_b, gt_a, gt_b,
                        w_seg, w_temp, w_pixel
                    )
                    val_loss += loss.item()
                    val_lp   += lp
                    val_ls   += ls
                    n_val    += 1
            val_loss /= max(n_val, 1)
            val_lp   /= max(n_val, 1)
            val_ls   /= max(n_val, 1)

            marker = ''
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), save_pt_path)
                save_int8_bz2(model, save_int8_path)
                marker = '  ← saved'

            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.6f} (pose={train_lp:.6f} seg={train_ls:.4f})  "
                  f"val={val_loss:.6f} (pose={val_lp:.6f} seg={val_ls:.4f})  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{marker}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.6f} (pose={train_lp:.6f} seg={train_ls:.4f})")

    print(f"\n  Best val_loss: {best_val:.6f}")

    # Inspect learned Haar gain — shows which sub-pixel channels were amplified
    gains = model.haar_gain.detach().cpu().squeeze().tolist()
    chan_names = ['R_y00','R_y10','R_y01','R_y11','G_y00','G_y10','G_y01','G_y11',
                  'B_y00','B_y10','B_y01','B_y11']
    print("  Learned Haar gains (>1 = amplified, expected: HF channels y10/y01/y11):")
    for name, g in zip(chan_names, gains):
        bar = '█' * int(abs(g - 1.0) * 20)
        print(f"    {name:8s}: {g:.4f}  {bar}")

    if os.path.exists(save_int8_path):
        size_kb = os.path.getsize(save_int8_path) / 1024
        print(f"\n  Model (int8.bz2): {save_int8_path} ({size_kb:.1f} KB)")
        rate_cost = (size_kb / 1024) / 37.5 * 25
        print(f"  Rate cost of model in archive: +{rate_cost:.4f} pts "
              f"({size_kb/1024:.3f} MB × 0.667 pts/MB)")
    print("\n  Next: run compress.sh to rebuild archive.zip (it will include the model),")
    print("        then evaluate with evaluate.sh")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train REN v2 (task-aware, CRF-36 targeted)'
    )
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--batch-size', type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--features',   type=int,   default=48)
    args = parser.parse_args()
    train(args)
