#!/usr/bin/env python
import av, torch, numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from frame_utils import camera_size, yuv420_to_rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL = (torch.outer(_r, _r) / (_r.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)
STRENGTH = 0.40


def segment_polygon(idx, w, h):
  segs = [
    (0,   299, [(0.14,0.52),(0.82,0.48),(0.98,1.00),(0.05,1.00)]),
    (300, 599, [(0.10,0.50),(0.76,0.47),(0.92,1.00),(0.00,1.00)]),
    (600, 899, [(0.18,0.50),(0.84,0.47),(0.98,1.00),(0.06,1.00)]),
    (900,1199, [(0.22,0.52),(0.90,0.49),(1.00,1.00),(0.10,1.00)]),
  ]
  for s, e, poly in segs:
    if s <= idx <= e:
      return [(x*w, y*h) for x,y in poly]
  return [(0.15*w,0.52*h),(0.85*w,0.48*h),(w,h),(0,h)]


def build_mask(idx, w, h, fr=24):
  img = Image.new("L", (w, h), 0)
  ImageDraw.Draw(img).polygon(segment_polygon(idx, w, h), fill=255)
  if fr > 0:
    img = img.filter(ImageFilter.GaussianBlur(radius=fr))
  m = torch.frombuffer(memoryview(img.tobytes()), dtype=torch.uint8).clone().view(h, w).float() / 255.0
  return m.unsqueeze(0).unsqueeze(0).to(DEVICE)


def decode_and_resize_to_file(video_path: str, dst: str):
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
        pil = Image.fromarray(t.numpy())
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), KERNEL, padding=0, groups=3)
        detail = x - blur
        mask = build_mask(n, target_w, target_h)
        x = x + STRENGTH * detail * mask
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
      f.write(t.contiguous().numpy().tobytes())
      n += 1
  container.close()
  return n


if __name__ == "__main__":
  import sys
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_and_resize_to_file(src, dst)
  print(f"saved {n} frames")
