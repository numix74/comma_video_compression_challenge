#!/usr/bin/env python
"""
adv_frames/compress.py — Adversarial Frame Compression

Pour chaque paire de frames consécutives, on cherche par descente de gradient
les frames RGB les plus compressibles qui satisfont SegNet et PoseNet.

Pipeline :
  1. Charger SegNet + PoseNet (offline, ne vont PAS dans l'archive)
  2. Pour chaque paire (t, t+1) :
     a. Init x* = version basse fréquence de la frame originale
     b. Optimiser : loss = α·CE(SegNet) + β·MSE(PoseNet) + γ·HF_energy
     c. Boucle codec : AV1 encode→decode, vérifier loss, re-optimiser si dégradé
  3. Encoder la séquence de frames optimisées en AV1 agressif
  4. Stocker : archive/0.mkv (vidéo adversariale) + archive/poses.npy.br

Usage :
  python compress.py [--crf 55] [--iters 80] [--batch-size 4] [--device cuda]
"""
import os, sys, io, bz2, subprocess, shutil, argparse, tempfile
import numpy as np
import torch
import torch.nn.functional as F
import brotli
from pathlib import Path
from safetensors.torch import load_file
import av
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb, rgb_to_yuv6
from modules import SegNet, PoseNet

HERE   = Path(__file__).resolve().parent
W, H   = camera_size          # 1164 × 874
SEG_W, SEG_H = 512, 384

# ── Hyperparamètres ────────────────────────────────────────────────────────────
ALPHA  = 1.0    # poids CE SegNet
BETA   = 10.0   # poids MSE PoseNet (plus fort car √ dans score)
GAMMA  = 0.05   # poids énergie haute fréquence (encourage compression)


def get_ffmpeg():
    local = ROOT / "ffmpeg-new"
    if local.exists() and os.access(local, os.X_OK):
        # Vérifier que les dépendances sont présentes
        import subprocess as sp
        r = sp.run([str(local), "-version"], capture_output=True)
        if r.returncode == 0:
            return str(local)
    return shutil.which("ffmpeg") or "ffmpeg"


def load_video_frames(path: Path, device) -> torch.Tensor:
    """Charge toutes les frames RGB en mémoire → (N, H, W, 3) uint8."""
    frames = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        t = yuv420_to_rgb(frame)           # (H, W, 3) uint8 tensor
        frames.append(t)
    container.close()
    return torch.stack(frames)             # (N, H, W, 3)


def hf_energy(x: torch.Tensor) -> torch.Tensor:
    """Énergie haute fréquence via Laplacien — pénalise les détails fins."""
    # x : (B, 3, H, W) float [0,255]
    kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3).expand(3, 1, 3, 3)
    lap = F.conv2d(x / 255.0, kernel, padding=1, groups=3)
    return lap.pow(2).mean()


def optimize_pair(
    frame_t:  torch.Tensor,   # (H, W, 3) uint8
    frame_t1: torch.Tensor,   # (H, W, 3) uint8
    segnet:   SegNet,
    posenet:  PoseNet,
    gt_mask_t:  torch.Tensor, # (SEG_H, SEG_W) long — cible SegNet frame t
    gt_mask_t1: torch.Tensor, # (SEG_H, SEG_W) long — cible SegNet frame t+1
    gt_pose:    torch.Tensor, # (1, 6) float  — cible PoseNet
    iters: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Renvoie (opt_t, opt_t1) — tenseurs float (H, W, 3) ∈ [0,255]."""

    # Initialisation basse-fréquence pour encourager la compressibilité
    def blur_init(f):
        x = f.float().permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
        x = F.avg_pool2d(x, 8, stride=1, padding=4)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x.squeeze(0).permute(1, 2, 0).clamp(0, 255)       # (H,W,3)

    xt  = blur_init(frame_t).requires_grad_(True)
    xt1 = blur_init(frame_t1).requires_grad_(True)

    optimizer = torch.optim.Adam([xt, xt1], lr=4.0, betas=(0.9, 0.99))

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)

        # ── SegNet loss ────────────────────────────────────────────────────
        seg_in_t  = F.interpolate(
            xt.permute(2,0,1).unsqueeze(0), size=(SEG_H, SEG_W), mode='bilinear'
        )
        seg_in_t1 = F.interpolate(
            xt1.permute(2,0,1).unsqueeze(0), size=(SEG_H, SEG_W), mode='bilinear'
        )
        logits_t  = segnet(seg_in_t)
        logits_t1 = segnet(seg_in_t1)
        loss_seg  = F.cross_entropy(logits_t,  gt_mask_t.unsqueeze(0))  \
                  + F.cross_entropy(logits_t1, gt_mask_t1.unsqueeze(0))

        # ── PoseNet loss ───────────────────────────────────────────────────
        pair_btchw = torch.stack([
            xt.permute(2,0,1),
            xt1.permute(2,0,1),
        ], dim=0).unsqueeze(0)  # (1, 2, 3, H, W)
        pose_in = posenet.preprocess_input(pair_btchw)  # (1, 12, H/2, W/2)
        pred_pose = posenet(pose_in)["pose"][..., :6]
        loss_pose = F.mse_loss(pred_pose, gt_pose)

        # ── Énergie haute fréquence ────────────────────────────────────────
        loss_hf = hf_energy(xt.permute(2,0,1).unsqueeze(0)) \
                + hf_energy(xt1.permute(2,0,1).unsqueeze(0))

        loss = ALPHA * loss_seg + BETA * loss_pose + GAMMA * loss_hf
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            xt.clamp_(0, 255)
            xt1.clamp_(0, 255)

    return xt.detach(), xt1.detach()


def encode_video_lossless(frames_np: np.ndarray, out_path: Path, fps: int):
    """Encode en FFV1 lossless pour checkpoint intermédiaire."""
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "ffv1", str(out_path)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(input=frames_np.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg lossless failed (code {proc.returncode})")


def encode_video_av1(frames_np: np.ndarray, out_path: Path, fps: int, crf: int):
    """Encode un array (N, H, W, 3) uint8 en AV1 via ffmpeg."""
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "pipe:0",
        "-vf", "format=yuv420p",
        "-c:v", "libaom-av1",
        "-crf", str(crf), "-b:v", "0",
        "-cpu-used", "4", "-g", "240", "-tune", "psnr",
        str(out_path)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(input=frames_np.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode})")


def decode_video_frames(path: Path) -> np.ndarray:
    """Décode une vidéo AV1 → (N, H, W, 3) uint8."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(arr)
    container.close()
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crf",        type=int,   default=55)
    parser.add_argument("--iters",      type=int,   default=80)
    parser.add_argument("--codec-iters",type=int,   default=2,
                        help="Nombre de boucles encode/decode/re-optim")
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video",      type=Path,  default=ROOT / "videos/0.mkv")
    parser.add_argument("--fps",        type=int,   default=20)
    parser.add_argument("--max-frames", type=int,   default=0,
                        help="Limiter à N frames pour test (0=toutes)")
    parser.add_argument("--save-optim", type=Path,  default=None,
                        help="Sauvegarder les frames optimisées (lossless MKV) avant boucle codec")
    parser.add_argument("--load-optim", type=Path,  default=None,
                        help="Charger frames optimisées depuis fichier (skip optimisation GPU)")
    args = parser.parse_args()

    device = torch.device(args.device)
    archive_dir = HERE / "archive"
    archive_dir.mkdir(exist_ok=True)

    # ── Charger SegNet + PoseNet (NE vont PAS dans l'archive) ─────────────
    print("Chargement SegNet + PoseNet...")
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(ROOT / "models/segnet.safetensors", device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(ROOT / "models/posenet.safetensors", device=str(device)))
    for p in (*segnet.parameters(), *posenet.parameters()):
        p.requires_grad_(False)

    # ── Charger les frames originales ─────────────────────────────────────
    print(f"Chargement vidéo : {args.video}")
    orig_frames = load_video_frames(args.video, device)  # (N, H, W, 3) uint8
    N = orig_frames.shape[0]
    if args.max_frames > 0:
        N = min(N, args.max_frames)
        # forcer N pair pour les paires
        N = N if N % 2 == 0 else N - 1
        orig_frames = orig_frames[:N]
    print(f"  {N} frames chargées")

    # ── Extraire poses GT et masks GT ─────────────────────────────────────
    print("Extraction cibles GT (SegNet + PoseNet)...")
    gt_masks = []
    gt_poses = []

    with torch.no_grad():
        for i in tqdm(range(N), desc="GT extraction"):
            f = orig_frames[i].float().permute(2,0,1).unsqueeze(0).to(device)
            seg_in = F.interpolate(f, size=(SEG_H, SEG_W), mode='bilinear')
            mask = segnet(seg_in).argmax(dim=1).squeeze(0).cpu()  # (SEG_H, SEG_W)
            gt_masks.append(mask)

        for i in tqdm(range(0, N-1, 2), desc="Pose GT extraction"):
            pair = torch.stack([
                orig_frames[i].float().permute(2,0,1),
                orig_frames[i+1].float().permute(2,0,1),
            ], dim=0).unsqueeze(0).to(device)  # (1, 2, 3, H, W)
            pose_in = posenet.preprocess_input(pair)
            pose = posenet(pose_in)["pose"][..., :6].cpu()
            gt_poses.append(pose)

    # Sauvegarder les poses (dans l'archive, comptent dans la taille)
    poses_np = torch.cat(gt_poses, dim=0).numpy()  # (N//2, 6)
    poses_buf = io.BytesIO()
    np.save(poses_buf, poses_np)
    poses_br = brotli.compress(poses_buf.getvalue(), quality=11)
    with open(archive_dir / "poses.npy.br", "wb") as f:
        f.write(poses_br)
    print(f"  Poses: {len(poses_br)/1024:.1f} KB")

    # ── Optimisation adversariale (skip si --load-optim fourni) ───────────
    if args.load_optim:
        print(f"\nChargement frames optimisées depuis {args.load_optim} (skip GPU optim)...")
        opt_np = decode_video_frames(args.load_optim)  # (N, H, W, 3) uint8
        N = opt_np.shape[0]
        print(f"  {N} frames chargées")
    else:
        print(f"\nOptimisation adversariale ({args.iters} iters/paire, CRF={args.crf})...")
        opt_frames = orig_frames.clone().float()  # (N, H, W, 3)

    if not args.load_optim:
        for i in tqdm(range(0, N-1, 2), desc="Optimisation"):
        pose_idx = i // 2
        gt_pose  = gt_poses[pose_idx].to(device)

        frame_t  = orig_frames[i]
        frame_t1 = orig_frames[i+1]

        xt, xt1 = optimize_pair(
            frame_t, frame_t1,
            segnet, posenet,
            gt_masks[i].to(device),
            gt_masks[i+1].to(device),
            gt_pose,
            args.iters, device,
        )

        opt_frames[i]   = xt.cpu()
        opt_frames[i+1] = xt1.cpu()

        # Dernière frame si N est impair
        if N % 2 == 1:
            opt_frames[N-1] = orig_frames[N-1].float()

    # ── Sauvegarder checkpoint lossless si demandé ────────────────────────
    if not args.load_optim:
        opt_np = opt_frames.clamp(0, 255).round().to(torch.uint8).numpy()  # (N,H,W,3)
    if args.save_optim:
        print(f"\nSauvegarde checkpoint lossless → {args.save_optim}")
        encode_video_lossless(opt_np, args.save_optim, args.fps)
        size_mb = args.save_optim.stat().st_size / 1024 / 1024
        print(f"  Checkpoint : {size_mb:.0f} MB")

    # ── Boucle codec : encode → decode → re-optim si nécessaire ──────────

    for codec_iter in range(args.codec_iters):
        print(f"\nBoucle codec {codec_iter+1}/{args.codec_iters}...")
        tmp_video = archive_dir / f"_tmp_codec_{codec_iter}.mkv"
        encode_video_av1(opt_np, tmp_video, args.fps, args.crf)
        decoded_np = decode_video_frames(tmp_video)
        tmp_video.unlink()

        # Vérifier la dégradation sur CPU (évite OOM sur 1200 frames)
        pixel_diff = np.abs(decoded_np.astype(np.float32) - opt_np.astype(np.float32)).mean()
        print(f"  Diff pixel moyenne après AV1: {pixel_diff:.2f}")

        # Re-optimiser depuis les frames décodées si dégradation > seuil
        if pixel_diff > 5.0 and codec_iter < args.codec_iters - 1:
            print("  Re-optimisation depuis frames décodées...")
            decoded_frames = torch.from_numpy(decoded_np)  # (N,H,W,3) uint8
            for i in tqdm(range(0, N-1, 2), desc=f"Re-optim codec {codec_iter+1}", leave=False):
                pose_idx = i // 2
                xt, xt1 = optimize_pair(
                    decoded_frames[i], decoded_frames[i+1],
                    segnet, posenet,
                    gt_masks[i].to(device),
                    gt_masks[i+1].to(device),
                    gt_poses[pose_idx].to(device),
                    args.iters // 2, device,  # moitié d'itérations
                )
                decoded_np[i]   = xt.cpu().clamp(0,255).round().byte().numpy()
                decoded_np[i+1] = xt1.cpu().clamp(0,255).round().byte().numpy()
            opt_np = decoded_np
        else:
            opt_np = decoded_np
            break

    # ── Encodage final ─────────────────────────────────────────────────────
    final_video = archive_dir / "0.mkv"
    print(f"\nEncodage final → {final_video}")
    encode_video_av1(opt_np, final_video, args.fps, args.crf)
    size_kb = final_video.stat().st_size / 1024
    poses_kb = (archive_dir / "poses.npy.br").stat().st_size / 1024
    total_kb = size_kb + poses_kb
    rate = total_kb * 1024 / (37_545_489)
    print(f"  Vidéo : {size_kb:.0f} KB")
    print(f"  Poses : {poses_kb:.1f} KB")
    print(f"  Total : {total_kb:.0f} KB  (rate={rate:.5f}, 25×rate={25*rate:.3f})")

    # ── Zipper l'archive ───────────────────────────────────────────────────
    import zipfile
    zip_path = HERE / "archive.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for f in archive_dir.iterdir():
            if not f.name.startswith("_"):
                zf.write(f, f.name)
    print(f"  archive.zip : {zip_path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
