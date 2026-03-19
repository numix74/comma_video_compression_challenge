from functools import partial
from examples.baseline_dataloader import DatasetClass as BaselineDatasetClass

DatasetClass = partial(BaselineDatasetClass, format='hevc')