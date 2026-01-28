# comma2k25 compression challenge 🤏

<p align="center">
<img height="300" alt="image" src="https://github.com/user-attachments/assets/f72fae51-96bd-47c6-b4ff-6e17e79220cb" />
</p>

[test_videos.zip](https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip) is a 2.4 GB archive of 64 driving videos from the comma2k19 dataset. Make it as small as possible while preserving semantic content (evaluated by a segmentation model) and temporal dynamics (evaluated by an egomotion relative pose model).

- distortion:
  - SegNet distortion: average class disagreements between the predictions of a SegNet evaluated on original vs. reconstructed frames
  - PoseNet distortion: MSE of the outputs of a PoseNet (x,y,z velocities and roll,pitch,yaw rates) evaluated on original vs. reconstructed seq_len frames
- rate
  - the size of the compressed archive devided by the size of the original archive
- score: a weighted average of the different components of the distortion and the rate

```
score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate
```

<p align="center">
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/4dc4d758-230f-4af7-9d5a-06230c48c274" />
</p>

## Quickstart
```
# clone the repo
git clone https://github.com/commaai/comma2k25_compression_challenge.git
cd comma2k25_compression_challenge

# git lfs
git lfs install
git lfs pull

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# gpu required... pick one: cu126/ cu128 / cu130
uv sync --group cu126

# activate or use "uv run python ..."
source .venv/bin/activate

# download videos
curl -fLO https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip

# test dataloaders
python frame_utils_dali.py

# test models
python modules.py

# naively recompress one video
bash examples/hevc_recompress.sh --crf 30 --scale 1 --in-dir deflated_test_videos/ --jobs 1 --video-names-file stage_1_test_video_names.txt

# evaluate the naive recompression strategy
torchrun --nproc-per-node 1 evaluate.py  # or just python evaluate.py
```

If everything worked as expected, this should producce a `report.txt` file with this content:

```
=== Evaluation config ===
  batch_size: 16
  compressed_archive_path: comma2k19_submission.zip
  compressed_deflated_dir: deflated_comma2k19_submission
  num_threads: 2
  prefetch_queue_depth: 4
  seed: 1234
  uncompressed_archive_path: test_videos.zip
  uncompressed_deflated_dir: deflated_test_videos
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.01504754
  Average SegNet Distortion: 0.00381063
  Submission file size (deflated): 14428760.00000000 bytes
  Original uncompressed size (deflated): 2399626528.00000000 bytes
  Compression Rate (deflated): 0.00601292
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.91929734
```

## submission format
```
WIP

```

`YourSubmissionDataset` should be a subclass of `torch.utils.data.IterableDataset` with the following signature and methods

```python

class YourSubmissionDataset(torch.utils.data.IterableDataset):
  def __init__(
      self,
      file_names: list[str],
      archive_path: Path,
      batch_size: int,
      device_id: int,
      **kwargs,
  ):
      self.file_names = file_names
      self.archive_path = archive_path

  def prepare_data(self):
    # called by all ranks
    # e.g. do something with comma2k19_submission.zip
    ...

  def __iter__(self):
    for file_name in self.file_names:
      # yield name, idx, batch

      # name: segment_id
      # idx: batch index within the segment
      # batch: torch.Tensor with shape (batch_size, seq_len, camera_size[1], camera_size[0], 3), dtype=uint8
      ...

```


