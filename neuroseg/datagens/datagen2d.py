from typing import List, Tuple, Union, Callable
import logging
from pathlib import Path

import numpy as np
# from skimage import io as skio
import tifffile
from tqdm import tqdm

from neuroseg.config import TrainConfig, PredictConfig
from batchgenerators.dataloading.data_loader import DataLoader
# from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
# from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.color_transforms import (
    BrightnessTransform,
    BrightnessMultiplicativeTransform,
    GammaTransform,
    ClipValueRange)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform)
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform_2,
    Rot90Transform,
    MirrorTransform)
from batchgenerators.transforms.abstract_transforms import Compose

SUPPORTED_FORMATS = ["tiff", "tif"]


class DataGen2D:
    def __init__(self,
                 config: Union[TrainConfig, PredictConfig],
                 partition: str = "train",
                 data_augmentation: bool = True):
        self.config = config
        self.partition = partition
        self.data_augmentation = data_augmentation

        self.dataset_mode = self.config.dataset_mode
        assert self.dataset_mode in ["single_images", "stack", "multi_stack"], f"{self.dataset_mode} is not supported"
        self.dataset_path = self.config.dataset_path

        self.frames_paths, self.mask_paths = self._get_paths()

        self.window_size = self.config.window_size
        self.batch_size = self.config.batch_size
        self.binarize_labels = self.config.binarize_labels

        # data augmentation config
        self.data_augmentation_transforms = self.config.da_transforms
        self.data_augmentation_transforms_config = self.config.da_transform_cfg

        self.data_augmentation_single_thread = self.config.da_single_thread
        self.data_augmentation_threads = 1 if self.config.da_single_thread is True else self.config.da_threads
        self.data_augmentation_shuffle = self.config.da_shuffle
        self.data_augmentation_seed = self.config.da_seed

        self.data_dict = {"img": self.frames_paths,
                          "label": self.mask_paths}

        self.data_loader = self._get_dataloader(self.dataset_mode)
        self.steps_per_epoch = self.data_loader.steps_per_epoch
        self.composed_transform = self._get_transform_chain()

        if self.data_augmentation_single_thread:
            self.generator_bg = SingleThreadedAugmenter(self.data_loader, self.composed_transform)
        else:
            self.generator_bg = MultiThreadedAugmenter(self.data_loader, self.composed_transform,
                                                       self.data_augmentation_threads)

        # passing data to a keras-understandable format
        self.data = self._get_keras_gen(self.generator_bg, channel_last=True)
        self.iter = self.data.__iter__()

    @staticmethod
    def _get_keras_gen(batchgen_generator: Union[SingleThreadedAugmenter, MultiThreadedAugmenter],
                       channel_last: bool = True):

        while True:
            batch_dict = next(batchgen_generator)
            frames = batch_dict["data"]
            masks = batch_dict["seg"]

            if channel_last:
                frames = np.moveaxis(frames, source=1, destination=-1)
                masks = np.moveaxis(masks, source=1, destination=-1)

            yield frames.astype(np.float32), masks.astype(np.float32)

    def _get_dataloader(self, dataset_mode: str):
        if dataset_mode == "single_images":
            data_loader = SingleImagesDataLoader(
                data=self.data_dict,
                batch_size=self.batch_size,
                window_size=self.window_size,
                num_threads_in_multithreaded=self.data_augmentation_threads,
                shuffle=self.data_augmentation_shuffle,
                seed_for_shuffle=self.data_augmentation_seed
            )
        elif dataset_mode == "stack":
            raise NotImplementedError("stack mode is not implemented yet")
        elif dataset_mode == "multi_stack":
            data_loader = MultiStackDataLoader(
                data=self.data_dict,
                batch_size=self.batch_size,
                window_size=self.window_size,
                num_threads_in_multithreaded=self.data_augmentation_threads,
                shuffle=self.data_augmentation_shuffle,
                seed_for_shuffle=self.data_augmentation_seed,
                binarize_labels=self.binarize_labels
            )
            # raise NotImplementedError("multi_stack mode is not implemented yet")
        else:
            raise ValueError(f"{dataset_mode} is not supported")
        return data_loader

    def _get_paths(self) -> Tuple[List[Path], List[Path]]:
        """
        Returns a list of paths to the frames and a list of paths to the masks
        """
        frames_paths = []
        mask_paths = []
        dataset_format = "tif"
        data_path = self.dataset_path / self.partition
        # TODO: make this flexible
        for path in data_path.glob(f"*/*.{dataset_format}"):
            if self.partition in str(path) and "frames" in str(path):
                frames_paths.append(path)
            if self.partition in str(path) and "masks" in str(path):
                mask_paths.append(path)

        frames_paths.sort()
        mask_paths.sort()

        assert len(frames_paths) == len(mask_paths), f"{len(frames_paths)} != {len(mask_paths)}"

        # single frame datasets should contain only one frame
        if self.dataset_mode == "stack":
            assert len(frames_paths) == 1, f"{len(frames_paths)} != 1"

        return frames_paths, mask_paths

    def _get_transform_chain(self):
        transforms = []
        if self.data_augmentation:
            for transform_str in self.data_augmentation_transforms:
                transform_fn = self._get_transform_fn(transform_str)
                transform_cfg = self._get_transform_cfg(transform_str)

                try:
                    transforms.append(transform_fn(**transform_cfg))
                except TypeError:
                    raise TypeError(f"{transform_str} received a wrong config: {transform_cfg}")

        # clip data values to [0.,1.]
        transforms.append(ClipValueRange(min=0., max=1.))

        return Compose(transforms)

    @staticmethod
    def _get_transform_fn(transform_str: str) -> Callable:
        transform_fns = {
            "brightness_transform": BrightnessTransform,
            "brightness_multiplicative_transform": BrightnessMultiplicativeTransform,
            "gamma_transform": GammaTransform,
            "gaussian_noise_transform": GaussianNoiseTransform,
            "gaussian_blur_transform": GaussianBlurTransform,
            "mirror_transform": MirrorTransform,
            "rot90_transform": Rot90Transform,
            "spatial_transform": SpatialTransform_2
        }
        if transform_str not in transform_fns:
            raise NotImplementedError(f"{transform_str} is not implemented yet")

        return transform_fns[transform_str]

    def _get_transform_cfg(self, transform_str: str) -> dict:
        transform_cfgs = {
            "brightness_transform": self._brightness_transform_cfg,
            "brightness_multiplicative_transform": self._brightness_multiplicative_transform_cfg,
            "gamma_transform": self._gamma_transform_cfg,
            "gaussian_noise_transform": self._gaussian_noise_transform_cfg,
            "gaussian_blur_transform": self._gaussian_blur_transform_cfg,
            "mirror_transform": self._mirror_transform_cfg,
            "rot90_transform": self._rot90_transform_cfg,
            "spatial_transform": self._spatial_transform_cfg,
        }

        if transform_str not in transform_cfgs:
            raise NotImplementedError(f"{transform_str} is not implemented yet")

        return transform_cfgs[transform_str]() if transform_str in transform_cfgs else None

    def _brightness_transform_cfg(self) -> dict:
        """brightness additive transform configurator"""
        brightness_transform_cfg = {
            "p_per_sample": 0.15,
            "mu": 0,
            "sigma": 0.01,
        }

        brightness_transform_cfg.update(self.data_augmentation_transforms_config["brightness_transform"])
        return brightness_transform_cfg

    def _brightness_multiplicative_transform_cfg(self) -> dict:
        """brightness multiplicative transform configurator"""
        brightness_multiplicative_transform_cfg = {
            "multiplier_range": (0.9, 1.1),
            "per_channel": True,
            "p_per_sample": 0.15}

        brightness_multiplicative_transform_cfg.update(
            self.data_augmentation_transforms_config["brightness_multiplicative_transform"])
        return brightness_multiplicative_transform_cfg

    def _gamma_transform_cfg(self):
        """gamma trasnform configurator"""
        gamma_transform_cfg = {
            "p_per_sample": 0.15,
            "per_channel": True,
            "invert_image": False,
            "retain_stats": True,
            "gamma_range": (0.9, 1.1)
        }

        gamma_transform_cfg.update(self.data_augmentation_transforms_config["gamma_transform"])
        return gamma_transform_cfg

    def _gaussian_noise_transform_cfg(self) -> dict:
        """gaussian noise additive transform configurator"""
        gaussian_noise_transform_cfg = {
            "p_per_sample": 0.15,
            "noise_variance": [0, 0.0005]
        }

        gaussian_noise_transform_cfg.update(self.data_augmentation_transforms_config["gaussian_noise_transform"])
        return gaussian_noise_transform_cfg

    def _gaussian_blur_transform_cfg(self) -> dict:
        """gaussian filtering transform configurator"""
        gaussian_blur_transform_cfg = {
            "p_per_sample": 0.15,
            "blur_sigma": (1, 5),
            "different_sigma_per_channel": True,
            "p_per_channel": True
        }

        gaussian_blur_transform_cfg.update(self.data_augmentation_transforms_config["gaussian_blur_transform"])
        return gaussian_blur_transform_cfg

    def _mirror_transform_cfg(self) -> dict:
        """mirror transform configurator"""
        mirror_transform_cfg = {
            "axes": (0, 1)
        }

        mirror_transform_cfg.update(self.data_augmentation_transforms_config["mirror_transform"])
        return mirror_transform_cfg

    def _rot90_transform_cfg(self) -> dict:
        """rigid 90 degree rotation transform configurator"""
        rotation_axes = (0, 1)
        rot90_transform_cfg = {
            "num_rot": (1, 2),
            # "axes": (0,1,2),
            "axes": rotation_axes,
            "p_per_sample": 0.15
        }

        rot90_transform_cfg.update(self.data_augmentation_transforms_config["rot90_transform"])
        return rot90_transform_cfg

    def _spatial_transform_cfg(self) -> dict:
        """spatial transform"""
        # default options
        spatial_transform_cfg = {
            "patch_size": self.window_size,
            "patch_center_dist_from_border": np.array(self.window_size) // 2,
            "do_elastic_deform": False,
            "deformation_scale": (0.25, 0.25),
            "do_rotation": False,
            "angle_x": (-np.pi / 10, np.pi / 10),  # data is in z, y, x format
            "angle_y": (0, 2 * np.pi),
            "do_scale": True,
            "scale": (0.95, 1.05),
            "border_mode_data": "nearest",
            "random_crop": True,
            "p_el_per_sample": 1,
            "p_rot_per_sample": 1,
            "p_scale_per_sample": 1,
        }

        # user-defined options
        spatial_transform_cfg.update(self.data_augmentation_transforms_config["spatial_transform"])
        return spatial_transform_cfg


class SingleImagesDataLoader(DataLoader):
    def __init__(self,
                 data: dict,
                 batch_size: int,
                 window_size: Union[tuple, list],
                 num_threads_in_multithreaded: int = 1,
                 shuffle: bool = False,
                 seed_for_shuffle: int = 123,
                 infinite: bool = False):

        super().__init__(
            data=data,
            batch_size=batch_size,
            num_threads_in_multithreaded=num_threads_in_multithreaded,
            shuffle=shuffle,
            seed_for_shuffle=seed_for_shuffle,
            infinite=infinite
        )

        self.img_paths = self._data["img"]
        self.label_paths = self._data["label"]

        self.rng_seed = seed_for_shuffle
        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

        self.window_size = window_size

        if len(self.img_paths) != len(self.label_paths):
            raise ValueError(f"{len(self.img_paths)} != {len(self.label_paths)}")

        self.img_volume, self.label_volume = self._load_data(frame_paths=self.img_paths,
                                                             mask_paths=self.label_paths)
        self.steps_per_epoch = self._get_steps_per_epoch()

    def _load_data(self,
                   frame_paths: List[Path],
                   mask_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads data from disk
        """
        frame_list = [self._load_img(img_path=frame_path) for frame_path in frame_paths]
        mask_list = [self._load_img(img_path=mask_path) for mask_path in mask_paths]

        frame_volume = np.stack(frame_list, axis=0)
        mask_volume = np.stack(mask_list, axis=0)

        del frame_list, mask_list
        return frame_volume, mask_volume

    @staticmethod
    def _load_img(
            img_path: Path,
            normalize: bool = True) -> np.ndarray:
        img = tifffile.imread(str(img_path))

        if normalize:
            norm_constant = np.iinfo(img.dtype).max
            img = img / norm_constant

        # target shape is [y, x, channels]
        if len(img.shape) == 2:
            logging.debug("adding a channel dimension")
            img = np.expand_dims(img, axis=-1)
        elif len(img.shape) == 3:
            pass
        else:
            raise ValueError(f"{img_path} has shape {img.shape}")

        return img

    def _get_steps_per_epoch(self) -> int:
        """
        Returns the number of steps per epoch
        """
        tot_img_px = np.prod(self.img_volume.shape)
        tot_mask_px = np.prod(self.window_size)
        return tot_img_px // (tot_mask_px * self.batch_size)

    def generate_train_batch(self) -> dict:
        """
        generates a {"data": stack_img, "seg": stack_label} batch dictionary
        """
        img_batch = []
        label_batch = []

        for _ in range(self.batch_size):
            crop_img, crop_label = self.get_random_crop(
                frame_volume=self.img_volume,
                mask_volume=self.label_volume,
                window_size=self.window_size)

            img_batch.append(crop_img)
            label_batch.append(crop_label)
        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)

        # target shape is [batch, ch, z, y, x]
        # adding channel in channel_first format
        stack_img = np.moveaxis(stack_img, -1, 1)
        stack_label = np.moveaxis(stack_label, -1, 1)

        return {"data": stack_img, "seg": stack_label}

    @classmethod
    def get_random_crop(cls,
                        frame_volume: np.ndarray,
                        mask_volume: np.ndarray,
                        window_size: Union[list, tuple, np.ndarray]):
        img_plane = np.random.randint(0, frame_volume.shape[0])

        # assert that the window size is less than the image size
        assert np.all(list(window_size) < list(frame_volume.shape[1:])), "window size is larger than image size"

        crop_box = cls._get_bounding_box(frame_shape=frame_volume.shape[1:],
                                         window_size=window_size)

        frame_crop = frame_volume[img_plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        mask_crop = mask_volume[img_plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        if not all(np.array(frame_crop.shape)[0:2] == np.array(window_size)):
            raise ValueError("frame and window size are not equal")
        if not all(np.array(mask_crop.shape)[0:2] == np.array(window_size)):
            raise ValueError("mask and window size are not equal")

        return frame_crop, mask_crop

    @staticmethod
    def _get_bounding_box(frame_shape: Union[list, tuple],
                          window_size: Union[list, tuple, np.ndarray]) -> np.ndarray:
        lower_x = int(np.random.uniform(low=0, high=frame_shape[1] - window_size[1]))
        lower_y = int(np.random.uniform(low=0, high=frame_shape[0] - window_size[0]))

        upper_x = lower_x + window_size[1]
        upper_y = lower_y + window_size[0]

        crop_box = np.stack((lower_x, lower_y, upper_x, upper_y), axis=-1)
        return crop_box


class MultiStackDataLoader(DataLoader):
    def __init__(self,
                 data: dict,
                 batch_size: int,
                 window_size: Union[list, tuple],
                 num_threads_in_multithreaded: int = 1,
                 shuffle: bool = False,
                 seed_for_shuffle: int = 123,
                 infinite: bool = False,
                 binarize_labels: bool = False):

        super().__init__(
            data=data,
            batch_size=batch_size,
            num_threads_in_multithreaded=num_threads_in_multithreaded,
            shuffle=shuffle,
            seed_for_shuffle=seed_for_shuffle,
            infinite=infinite)

        self.img_paths = self._data["img"]
        self.label_paths = self._data["label"]
        self.binarize_labels = binarize_labels

        self.rng_seed = seed_for_shuffle
        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

        self.window_size = window_size

        if len(self.img_paths) != len(self.label_paths):
            raise ValueError("img and label paths must have the same length")

        self.dataset_list = self._load_data()
        self.steps_per_epoch = self._get_steps_per_epoch()

    def generate_train_batch(self):
        img_batch = []
        label_batch = []

        for _ in range(self.batch_size):
            random_idx = np.random.randint(0, len(self.dataset_list))

            crop_img, crop_label = SingleImagesDataLoader.get_random_crop(
                frame_volume=self.dataset_list[random_idx][0],
                mask_volume=self.dataset_list[random_idx][1],
                window_size=self.window_size)

            img_batch.append(crop_img)
            label_batch.append(crop_label)

        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)

        # target shape is [batch, ch, z, y, x]
        # adding channel in channel_first format
        stack_img = np.moveaxis(stack_img, -1, 1)
        stack_label = np.moveaxis(stack_label, -1, 1)

        return {"data": stack_img, "seg": stack_label}

    def _load_data(self):
        dataset_list = []
        print("Loading data from disk...")
        for img_path, label_path in tqdm(zip(self.img_paths, self.label_paths), total=len(self.img_paths)):
            img = self._load_img(img_path, normalize=True)
            label = self._load_img(label_path, binarize=self.binarize_labels)
            dataset_list.append((img, label, ))
        return dataset_list

    @staticmethod
    def _load_img(img_path: Path,
                  normalize: bool = True,
                  binarize: bool = False) -> np.ndarray:
        img = tifffile.imread(str(img_path))

        norm_constant = np.iinfo(img.dtype).max
        if binarize:
            img = np.where(img > 0, norm_constant, 0)
        if normalize:
            img = img / norm_constant

        # target shape is [z, y, x, channels]

        if len(img.shape) == 3:
            logging.debug("image shape is [z, y, x], adding channel dimension")
            img = np.expand_dims(img, axis=-1)
        elif len(img.shape) == 4:
            pass
        else:
            raise ValueError("image shape must be 3 or 4")

        return img

    def _get_steps_per_epoch(self):
        tot_img_px = 0
        for img, mask in self.dataset_list:
            tot_img_px += np.prod(img.shape)

        tot_window_px = np.prod(self.window_size)

        return tot_img_px // (tot_window_px * self.batch_size)