#!/usr/bin/env python
import os, torch, math, io, zipfile
from pathlib import Path
from tqdm import tqdm
from frame_utils_dali import DaliHevcDataset, camera_size, seq_len
from modules import DistortionNet, segnet_sd_path, posenet_sd_path
# import YourSubmissionDataset

num = 1
batch_size = 32
num_threads = 2
seed = 123
prefetch_queue_depth = 4
compressed_archive_path = Path('./comma2k19_submission.zip')
compressed_deflated_dir = Path('./deflated_comma2k19_submission/')
uncompressed_archive_path = Path('./test_videos.zip')
uncompressed_deflated_dir = Path('./deflated_test_videos/')

def get_zipped_size(deflated_dir: Path, compresslevel: int=1, file_names: None | list[str] = None) -> int:
  # this function zips the files in deflated_dir (or a subset if file_names is provided)
  # we recompute the size in case there are other files that are needed for the decompression and benefit from zip compression
  # this assumes those files have a different suffix than the provided file_names...
  if file_names is None:
    files = [f for f in deflated_dir.rglob("*") if f.is_file()]
  else:
    base_files = [deflated_dir / fn for fn in file_names]
    assert base_files
    assert all(p.suffix == base_files[0].suffix for p in base_files)
    other_files = [p for p in deflated_dir.rglob("*") if p.is_file() and p.suffix != base_files[0].suffix]
    if other_files: print(f"including {len(other_files)} additional files for size computation")
    files = base_files + other_files

  print(f"zipping {len(files)} files for size computation")
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
    _ = [zf.write(f) for f in files]
  return buf.tell()

def main():
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  rank = int(os.getenv("RANK", "0"))
  world_size = int(os.getenv("WORLD_SIZE", "1"))
  is_distributed = world_size > 1
  assert world_size == 1 or torch.distributed.is_available()
  assert torch.cuda.is_available()
  device = torch.device("cuda", local_rank)
  torch.cuda.set_device(device)

  if is_distributed and not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl", device_id=local_rank)

  distortion_net = DistortionNet().eval().to(device=device)
  distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

  with open("test_video_names.txt", "r") as file:
    test_video_names = [line.strip() for line in file.readlines()][:num]

  ds_gt = DaliHevcDataset(test_video_names, archive_path=uncompressed_archive_path, data_dir=uncompressed_deflated_dir, batch_size=batch_size, device_id=local_rank, num_threads=num_threads, seed=seed, prefetch_queue_depth=prefetch_queue_depth)
  ds_gt.prepare_data()
  dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)

  # replace with your YourSubmissionDataset implementation
  ds_comp = DaliHevcDataset(test_video_names, archive_path=compressed_archive_path, data_dir=compressed_deflated_dir, batch_size=batch_size, device_id=local_rank, num_threads=num_threads, seed=seed, prefetch_queue_depth=prefetch_queue_depth)
  ds_comp.prepare_data()
  dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)
  # end replace

  if rank == 0:
    compressed_size = get_zipped_size(compressed_deflated_dir, file_names=test_video_names)
    uncompressed_size = get_zipped_size(uncompressed_deflated_dir, file_names=test_video_names)
    rate = compressed_size / uncompressed_size

  dl = zip(dl_gt, dl_comp)
  posenet_dists = torch.zeros([], device=device)
  segnet_dists = torch.zeros([], device=device)
  steps = 0
  with torch.inference_mode():
    for (_,_,batch_gt), (_,_,batch_comp) in tqdm(dl):
      steps += 1
      assert batch_gt.shape == (batch_size, seq_len, camera_size[1], camera_size[0], 3), f"unexpected batch shape: {batch_gt.shape}"
      assert batch_comp.shape == (batch_size, seq_len, camera_size[1], camera_size[0], 3), f"unexpected batch shape: {batch_comp.shape}"
      posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
      posenet_dists += posenet_dist
      segnet_dists += segnet_dist

    if is_distributed and torch.distributed.is_initialized():
      torch.distributed.all_reduce(posenet_dists, op=torch.distributed.ReduceOp.AVG)
      torch.distributed.all_reduce(segnet_dists, op=torch.distributed.ReduceOp.AVG)

    if rank == 0:
      posenet_dist = (posenet_dists / steps).item()
      segnet_dist = (segnet_dists / steps).item()
      score = 100 * segnet_dist +  math.sqrt(posenet_dist * 10)  + 25 * rate
      print(f"Results over {steps*world_size*batch_size} samples")
      print(f"Average PoseNet Distortion: {posenet_dist:.8f}")
      print(f"Average SegNet Distortion: {segnet_dist:.8f}")
      print(f"Compression Rate (from deflated data): {rate:.8f}")
      print(f"Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = {score:.8f}")

  # Cleanup
  if is_distributed and torch.distributed.is_initialized():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
  main()
