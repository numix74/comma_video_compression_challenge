#!/usr/bin/env python
import argparse, av, torch
from pathlib import Path
import torch.nn.functional as F
from frame_utils import camera_size, yuv420_to_rgb


def decode_and_resize_to_file(video_path: str, dst: Path):
  target_w, target_h = camera_size
  fmt = 'hevc' if video_path.endswith('.hevc') else None
  container = av.open(video_path, format=fmt)
  stream = container.streams.video[0]
  n = 0
  with open(dst, 'wb') as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)  # (H, W, 3)
      H, W, _ = t.shape
      if H != target_h or W != target_w:
        x = t.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
        x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
      f.write(t.contiguous().numpy().tobytes())
      n += 1
  container.close()
  return n


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-dir", type=str, required=True, help="Directory with compressed video files")
  parser.add_argument("--output-dir", type=str, required=True, help="Directory to save raw tensor files")
  parser.add_argument("--file-list", type=str, required=True, help="Text file with video paths (one per line)")
  args = parser.parse_args()

  data_dir = Path(args.data_dir)
  output_dir = Path(args.output_dir)
  file_names = Path(args.file_list).read_text().splitlines()
  file_names = [str(Path(fn).with_suffix('.mkv')) for fn in file_names]

  for fn in file_names:
    src = data_dir / fn
    dst = output_dir / Path(fn).with_suffix('.raw')
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert src.exists(), f"ERROR: {src} not found"
    print(f"Decoding + resizing {fn} ...", end=" ", flush=True)
    n = decode_and_resize_to_file(str(src), dst)
    print(f"saved {n} frames")


if __name__ == "__main__":
  main()
