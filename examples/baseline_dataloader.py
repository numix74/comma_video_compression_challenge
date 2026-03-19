import torch
import torch.nn.functional as F
from functools import partial
from frame_utils import AVVideoDataset, DaliVideoDataset, camera_size

class ResizingVideoDataset:
  """Mixin that resizes frames to camera_size using interpolation."""
  resize_mode = 'bicubic'

  def __iter__(self):
    target_w, target_h = camera_size
    mode = self.resize_mode
    for path, idx, batch in super().__iter__():
      # batch: (B, seq_len, H, W, 3) uint8
      B, S, H, W, C = batch.shape
      if H != target_h or W != target_w:
        x = batch.reshape(B * S, H, W, C).permute(0, 3, 1, 2).float()
        kwargs = {} if mode == 'nearest' else {'align_corners': False}
        x = F.interpolate(x, size=(target_h, target_w), mode=mode, **kwargs)
        batch = x.clamp(0, 255).permute(0, 2, 3, 1).reshape(B, S, target_h, target_w, C).to(torch.uint8)
      yield path, idx, batch

class ResizingDaliVideoDataset(ResizingVideoDataset, DaliVideoDataset): pass
class ResizingAVVideoDataset(ResizingVideoDataset, AVVideoDataset): pass

if torch.cuda.is_available():
  DatasetClass = partial(ResizingDaliVideoDataset, format='mkv')
else:
  DatasetClass = partial(ResizingAVVideoDataset, format='mkv')
