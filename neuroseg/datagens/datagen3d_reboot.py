from typing import Union, Tuple, List, Callable
import logging
import random

import numpy as np

from neuroseg.config import TrainConfig, PredictConfig
from batchgenerators.dataloading import SingleThreadedAugmenter, MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms import Compose
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform_2,
    Rot90Transform)
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessTransform,
    BrightnessMultiplicativeTransform,
    GammaTransform,
    ClipValueRange)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform)


# SingleStackCroppedDataLoader
# MultiStackCroppedDataLoader

# no single_images, no csv

class DataGen3D:
    def __init__(self,
                 config: Union[TrainConfig, PredictConfig],
                 partition: str = "train",
                 data_augmentation: bool = True,
                 verbose: bool = False,
                 normalize_inputs=True):

        self.config = config
        self.partition = partition
        self.data_path_dict = self.config.path_dict[partition]
        self.dataset_mode = self.config.dataset_mode
        self.ignore_last_channel = self.config.ignore_last_channel

        self.frames_paths, self.mask_paths = self._glob_subdirs()

        self.verbose = verbose

        self.window_size = self.config.window_size
        self.batch_size = self.config.batch_size

        self.data_augmentation = data_augmentation
        self.data_augmentation_transforms = self.config.da_transforms
        self.data_augmentation_transforms_config = self.config.da_transform_cfg

        self.single_thread_augmentation = self.config.da_single_thread
        self.data_augmentation_threads = 1 if config.da_single_thread is True else config.da_threads
        self.shuffle = self.config.da_shuffle
        self.seed = self.config.da_seed

        self.class_values = self.config.class_values
        self.normalize_inputs = normalize_inputs

        data_dict = {"img": self.frames_paths,
                     "label": self.maskpaths}

