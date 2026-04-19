#!/usr/bin/env python
"""
ren_v3/train_ren.py — Entraînement de l'OC-REN (Oracle-Conditioned REN)

Vs ren_v2:
  - Le REN reçoit les cibles PoseNet GT comme signal de conditionnement
  - pose_embed: Linear(6, 12) broadcast-add aux features Haar
  - Les cibles sont calculées une fois depuis la vidéo GT (pas le compressé)
  - Même calibration de loss que ren_v2

Usage:
  # 1. Lancer compress.sh pour produire archive/0.mkv
  # 2. Les pose targets sont extraits automatiquement si absents
  python train_ren.py [--epochs 150] [--batch-size 8] [--lr 1e-3] [--features 48]
"""
import os, sys, argparse, math, io, bz2, struct, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from frame_utils import camera_size, yuv420_to_rgb, rgb_to_yuv6
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OCREN(nn.Module):
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
        B = x.shape[0]
        x_norm = x / 255.0
        shuffled = self.down(x_norm)
        scaled   = shuffled * self.haar_gain
        cond = self.pose_embed(pose_target).view(B, 12, 1, 1)
        scaled = scaled + cond
        residual = self.up(self.body(scaled))
        return (x_norm + residual).clamp(0, 1) * 255.0


def save_int8_bz2(model, path):
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
    print(f"  Saved int8.bz2: {path} ({len(compressed)/1024:.1f} KB)")


def load_pose_targets_file(path):
    with open(path, 'rb') as f:
        N, D = struct.unpack('<II', f.read(8))
        data = np.frombuffer(bz2.decompress(f.read()), dtype=np.float16).reshape(N, D)
    return torch.from_numpy(data.astype(np.float32))


def decode_all_frames(video_path, target_w=None, target_h=None):
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb(frame)
        if target_w and target_h and (t.shape[0] != target_h or t.shape[1] != target_w):
            pil = Image.fromarray(t.numpy())
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            t = torch.from_numpy(np.array(pil))
        frames.append(t)
    container.close()
    return frames


def compute_pose_targets(frames, posenet, device, mH=384, mW=512, batch_size=16):
    """Calcule les cibles PoseNet GT pour toutes les paires."""
    n_pairs = len(frames) // 2
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
            batch.append(torch.stack([f0, f1]))
        pair_batch = torch.stack(batch)
        with torch.no_grad():
            x = pair_batch.view(B * 2, 3, mH, mW)
            yuv = rgb_to_yuv6(x).view(B, 12, mH // 2, mW // 2)
            out = posenet(yuv)
            poses = out['pose'][:, :6].cpu()
        all_poses.append(poses)
    return torch.cat(all_poses, dim=0)  # (N_pairs, 6)


class PairDataset(Dataset):
    def __init__(self, comp_frames, gt_frames, pose_targets):
        assert len(comp_frames) == len(gt_frames)
        self.comp = comp_frames
        self.gt = gt_frames
        self.pose = pose_targets  # (N_pairs, 6)

    def __len__(self):
        return len(self.comp) - 1

    def __getitem__(self, idx):
        ca = self.comp[idx].permute(2, 0, 1).float()
        cb = self.comp[idx + 1].permute(2, 0, 1).float()
        ga = self.gt[idx].permute(2, 0, 1).float()
        gb = self.gt[idx + 1].permute(2, 0, 1).float()
        pair_idx = idx // 2
        pose = self.pose[min(pair_idx, len(self.pose) - 1)]
        return ca, cb, ga, gb, pose


def compute_loss(model, posenet, segnet, comp_a, comp_b, gt_a, gt_b, pose,
                 w_seg, w_temp, w_pixel):
    inf_a = model(comp_a, pose)
    inf_b = model(comp_b, pose)

    pair_inf = torch.stack([inf_a.permute(0, 2, 3, 1), inf_b.permute(0, 2, 3, 1)], dim=1)
    pair_gt  = torch.stack([gt_a.permute(0, 2, 3, 1),  gt_b.permute(0, 2, 3, 1)],  dim=1)

    posenet_in_inf = posenet.preprocess_input(pair_inf.permute(0, 1, 4, 2, 3))
    with torch.no_grad():
        posenet_in_gt  = posenet.preprocess_input(pair_gt.permute(0, 1, 4, 2, 3))
        posenet_out_gt = posenet(posenet_in_gt)
    posenet_out_inf = posenet(posenet_in_inf)
    loss_pose = sum(
        F.mse_loss(posenet_out_inf[h.name][..., :h.out // 2],
                   posenet_out_gt[h.name][..., :h.out // 2])
        for h in posenet.hydra.heads
    )

    segnet_in_inf = segnet.preprocess_input(pair_inf.permute(0, 1, 4, 2, 3))
    with torch.no_grad():
        segnet_in_gt = segnet.preprocess_input(pair_gt.permute(0, 1, 4, 2, 3))
        logits_gt    = segnet(segnet_in_gt)
    logits_inf = segnet(segnet_in_inf)
    loss_seg = F.kl_div(
        F.log_softmax(logits_inf, dim=1),
        F.softmax(logits_gt, dim=1),
        reduction='batchmean'
    )

    corr_a = (inf_a - comp_a) / 255.0
    corr_b = (inf_b - comp_b) / 255.0
    loss_temp = F.l1_loss(corr_a, corr_b)

    loss_pixel = (F.l1_loss(inf_a / 255.0, gt_a / 255.0) +
                  F.l1_loss(inf_b / 255.0, gt_b / 255.0)) / 2.0

    loss = loss_pose + w_seg * loss_seg + w_temp * loss_temp + w_pixel * loss_pixel
    return loss, loss_pose.item(), loss_seg.item()


def train(args):
    print(f"Device: {DEVICE}")
    torch.manual_seed(1234)
    np.random.seed(1234)

    W, H = camera_size

    archive_path = os.path.join(HERE, 'archive/0.mkv')
    if not os.path.exists(archive_path):
        print(f"ERROR: {archive_path} not found. Run compress.sh first.")
        sys.exit(1)

    print(f"Loading compressed frames from {archive_path}...")
    comp_frames = decode_all_frames(archive_path, target_w=W, target_h=H)
    print(f"  {len(comp_frames)} frames")

    gt_path = os.path.join(ROOT, 'videos/0.mkv')
    print(f"Loading GT frames from {gt_path}...")
    gt_frames = decode_all_frames(gt_path)
    print(f"  {len(gt_frames)} frames")

    n = min(len(comp_frames), len(gt_frames))
    comp_frames, gt_frames = comp_frames[:n], gt_frames[:n]

    print("Loading DistortionNet...")
    dn = DistortionNet().to(DEVICE).eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    for p in dn.parameters():
        p.requires_grad_(False)
    posenet = dn.posenet
    segnet  = dn.segnet

    pose_path = os.path.join(HERE, 'archive/0.pose')
    if os.path.exists(pose_path):
        print(f"Loading pose targets from {pose_path}...")
        pose_targets = load_pose_targets_file(pose_path)
    else:
        print("Computing pose targets from GT video...")
        pose_targets = compute_pose_targets(gt_frames, posenet, DEVICE)
        # Sauvegarder pour réutilisation
        import struct as st, bz2 as bz
        data = pose_targets.numpy().astype(np.float16)
        compressed = bz.compress(data.tobytes(), compresslevel=9)
        os.makedirs(os.path.join(HERE, 'archive'), exist_ok=True)
        with open(pose_path, 'wb') as f:
            f.write(st.pack('<II', *pose_targets.shape))
            f.write(compressed)
        print(f"  Saved {pose_path} ({os.path.getsize(pose_path)/1024:.1f} KB)")
    print(f"  {pose_targets.shape[0]} pose targets")

    split = int(n * 0.80)
    train_ds = PairDataset(comp_frames[:split], gt_frames[:split], pose_targets)
    val_ds   = PairDataset(comp_frames[split:], gt_frames[split:], pose_targets)
    print(f"  Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = OCREN(features=args.features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: OC-REN(features={args.features}), {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    save_pt_path   = os.path.join(HERE, 'ren_model.pt')
    save_int8_path = os.path.join(HERE, 'ren_model.int8.bz2')

    # Calibration des poids de loss (identique à ren_v2)
    print("\n  Calibrating loss weights...")
    model.train()
    ca, cb, ga, gb, pose = train_ds[0]
    ca   = ca.unsqueeze(0).to(DEVICE)
    cb   = cb.unsqueeze(0).to(DEVICE)
    ga   = ga.unsqueeze(0).to(DEVICE)
    gb   = gb.unsqueeze(0).to(DEVICE)
    pose_b = pose.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, lp0, ls0 = compute_loss(model, posenet, segnet, ca, cb, ga, gb, pose_b, 1.0, 0.005, 0.0)
    d_pose_est = max(lp0, 1e-6)
    sens_pose  = 10.0 / (2.0 * math.sqrt(10.0 * d_pose_est))
    w_seg      = max(1.0, min(20.0, 100.0 / sens_pose))
    w_temp     = 0.005
    w_pixel    = max(0.005, min(0.5, lp0 * 0.05 / max(ls0 * 0.01, 1e-8)))
    print(f"  w_seg={w_seg:.2f}, w_temp={w_temp}, w_pixel={w_pixel:.4f}")
    del ca, cb, ga, gb, pose_b

    drive_dir = args.drive_dir
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        print(f"  Drive backup: {drive_dir}")

    best_val = float('inf')
    print(f"\n  Training {args.epochs} epochs (batch={args.batch_size}, lr={args.lr})\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_lp, train_ls = 0.0, 0.0, 0.0
        n_batches = 0

        for ca, cb, ga, gb, pose in train_loader:
            ca   = ca.to(DEVICE); cb = cb.to(DEVICE)
            ga   = ga.to(DEVICE); gb = gb.to(DEVICE)
            pose = pose.to(DEVICE)

            optimizer.zero_grad()
            loss, lp, ls = compute_loss(model, posenet, segnet, ca, cb, ga, gb, pose,
                                        w_seg, w_temp, w_pixel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item(); train_lp += lp; train_ls += ls
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
                for ca, cb, ga, gb, pose in val_loader:
                    ca   = ca.to(DEVICE); cb = cb.to(DEVICE)
                    ga   = ga.to(DEVICE); gb = gb.to(DEVICE)
                    pose = pose.to(DEVICE)
                    loss, lp, ls = compute_loss(model, posenet, segnet, ca, cb, ga, gb, pose,
                                                w_seg, w_temp, w_pixel)
                    val_loss += loss.item(); val_lp += lp; val_ls += ls
                    n_val += 1
            val_loss /= max(n_val, 1)
            val_lp   /= max(n_val, 1)
            val_ls   /= max(n_val, 1)

            marker = ''
            is_best = val_loss < best_val
            force_save = (args.save_every > 0 and epoch % args.save_every == 0)

            if is_best or force_save:
                if is_best:
                    best_val = val_loss
                torch.save(model.state_dict(), save_pt_path)
                save_int8_bz2(model, save_int8_path)
                marker = '  ← best' if is_best else '  ← checkpoint'
                if drive_dir:
                    shutil.copy(save_int8_path, os.path.join(drive_dir, 'ren_model.int8.bz2'))
                    shutil.copy(save_pt_path,   os.path.join(drive_dir, 'ren_model.pt'))
                    marker += ' + drive'

            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f} (pose={train_lp:.4f} seg={train_ls:.4f})  "
                  f"val={val_loss:.4f} (pose={val_lp:.4f} seg={val_ls:.4f})  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{marker}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f} (pose={train_lp:.4f} seg={train_ls:.4f})")

    print(f"\n  Best val_loss: {best_val:.6f}")
    if os.path.exists(save_int8_path):
        size_kb = os.path.getsize(save_int8_path) / 1024
        rate_cost = (size_kb / 1024) / 37.5 * 25
        print(f"  Model: {size_kb:.1f} KB → rate cost +{rate_cost:.4f} pts")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--batch-size', type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--features',   type=int,   default=48)
    parser.add_argument('--drive-dir',  type=str,   default='',
                        help='Dossier Google Drive pour backup des checkpoints')
    parser.add_argument('--save-every', type=int,   default=10,
                        help='Forcer sauvegarde Drive toutes les N epochs (0=désactivé)')
    args = parser.parse_args()
    train(args)
