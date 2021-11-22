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


class DataGen2D:
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

        self.data_loader = BboxCroppedDataLoader(
            data=data_dict,
            batch_size=self.batch_size,
            window_size=self.window_size,
            num_threads_in_multithreaded=self.data_augmentation_threads,
            shuffle=self.shuffle,
            seed_for_shuffle=self.seed,
            class_values=self.class_values,
            ignore_last_channel=self.ignore_last_channel
        )

        self.steps_per_epoch = self.data_loader.steps_per_epoch
        self.composed_transform = self.get_transform_chain()

        if self.single_thread:
            self.gen = SingleThreadedAugmenter(self.dataLoader, self.composed_transform)
        else:
            self.gen = MultiThreadedAugmenter(self.dataLoader,
                                              self.composed_transform,
                                              self.threads)

        self.data = self.get_keras_gen(self.gen, channel_last=True)
        self.iter = self.data.__iter__()

    def _get_data_dict(self) -> dict:
        if self.dataset_mode == "stack":
            self.frames_path = self.frames_paths[0]
            self.masks_path = self.masks_path[0]
            data = {

            }
    def _path_sanity_check(self) -> None:
        if self.dataset_mode in ["single_images", "stack"]:
            if not (self.data_path_dict["frames"].is_dir()
                    and self.data_path_dict["masks"].is_dir()):
                raise ValueError("dataset paths are not actual dirs")

    def _glob_subdirs(self) -> Tuple[list, list]:
        assert self.dataset_mode in ["single_images", "stack",
                                     "multi_stack"], f"unexpected dataset mode {self.dataset_mode}"
        frames_paths = self._glob_dir("frames")
        mask_paths = self._glob_dir("masks")

        return frames_paths, mask_paths

    def _glob_dir(self, subdir: st) -> list:
        subdir_paths = [
            str(imgpath) for imgpath in sorted(self.data_path_dict[subdir].glob("*.*"))
            if self._is_supported_format(imgpath)]

        if self.verbose:
            logging.info("there are {} {} imgs".format(len(subdir_paths), subdir))
        return subdir_paths

    @staticmethod
    def _is_supported_format(fpath: Path) -> bool:
        extension = fpath.suffix.split(".")[1]
        return extension in SUPPORTED_IMG_FORMATS or extension in SUPPORTED_STACK_FORMATS

    @classmethod
    def get_keras_gen(cls,
                      batchgen: Union[SingleThreadedAugmenter, MultiThreadedAugmenter],
                      channel_last: bool = True):
        while True:
            batch_dict = next(batchgen)
            frames = batch_dict["data"]
            masks = batch_dict["seg"]

            if channel_last:
                frames = cls._to_channel_last(frames)
                masks = cls._to_channel_last(masks)

            yield frames.astype(np.float32), masks.astype(np.float32)

    @staticmethod
    def _to_channel_last(input_tensor: np.ndarray) -> np.ndarray:
        """convert (batch, ch, z, y, x) tensor to (batch, z, y, x, ch) tensor"""
        return np.moveaxis(input_tensor, source=1, destination=-1)

    def get_transform_chain(self) -> Compose:
        if self.data_augmentation:
            transforms = self.get_augment_transforms()
        else:
            transforms = []

        # clip data values to [0. 1.]
        clip_min = 0. if self.normalize_inputs else np.finfo(np.float32).min
        clip_max = 1.0 if self.normalize_inputs else np.finfo(np.float32).max

        transforms.append(ClipValueRange(min=clip_min, max=clip_max))

        return Compose(transforms)

    def get_augment_transforms(self):
        transforms = []
        for transform in self.transforms:
            transform_fn = self.get_transform_fn(transform)
            transform_cfg = self.get_transform_cfg(transform)

            transforms.append(transform_fn(**transform_cfg))
        return transforms

    @classmethod
    def get_transform_fn(cls, transform: str) -> Callable:
        transform_fns = {
            "brightness_transform": BrightnessTransform,
            "brightness_multiplicative_transform": BrightnessMultiplicativeTransform,
            "gamma_transform": GammaTransform,
            "gaussian_noise_transform": GaussianNoiseTransform,
            "gaussian_blur_transform": GaussianBlurTransform,
            "mirror_transform": MirrorTransform,
            "rot90_transform": Rot90Transform,
            "spatial_transform": SpatialTransform_2,
        }
        if transform not in transform_fns:
            raise NotImplementedError("transform {} not supported".format(transform))
        return transform_fns[transform] if transform in transform_fns else None

    def get_transform_cfg(self, transform: str) -> dict:
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
        if transform not in transform_cfgs:
            raise NotImplementedError("transform {} not supported".format(transform))

        return transform_cfgs[transform]() if transform in transform_cfgs else None

    def _brightness_transform_cfg(self) -> dict:
        """brightness additive transform configurator"""
        brightness_transform_cfg = {
            "p_per_sample": 0.15,
            "mu": 0,
            "sigma": 0.01,
        }

        brightness_transform_cfg.update(self.transform_cfg["brightness_transform"])
        return brightness_transform_cfg

    def _brightness_multiplicative_transform_cfg(self) -> dict:
        """brightness multiplicative transform configurator"""
        brightness_multiplicative_transform_cfg = {
            "multiplier_range": (0.9, 1.1),
            "per_channel": True,
            "p_per_sample": 0.15}

        brightness_multiplicative_transform_cfg.update(self.transform_cfg["brightness_multiplicative_transform"])
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

        gamma_transform_cfg.update(self.transform_cfg["gamma_transform"])
        return gamma_transform_cfg

    def _gaussian_noise_transform_cfg(self) -> dict:
        """gaussian noise additive transform configurator"""
        gaussian_noise_transform_cfg = {
            "p_per_sample": 0.15,
            "noise_variance": [0, 0.0005]
        }

        gaussian_noise_transform_cfg.update(self.transform_cfg["gaussian_noise_transform"])
        return gaussian_noise_transform_cfg

    def _gaussian_blur_transform_cfg(self) -> dict:
        """gaussian filtering transform configurator"""
        gaussian_blur_transform_cfg = {
            "p_per_sample": 0.15,
            "blur_sigma": (1, 5),
            "different_sigma_per_channel": True,
            "p_per_channel": True
        }

        gaussian_blur_transform_cfg.update(self.transform_cfg["gaussian_blur_transform"])
        return gaussian_blur_transform_cfg

    def _mirror_transform_cfg(self) -> dict:
        """mirror transform configurator"""
        mirror_transform_cfg = {
            "axes": (0, 1)
        }

        mirror_transform_cfg.update(self.transform_cfg["mirror_transform"])
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

        rot90_transform_cfg.update(self.transform_cfg["rot90_transform"])
        return rot90_transform_cfg

    def _spatial_transform_cfg(self) -> dict:
        """spatial transform"""
        # default options
        spatial_transform_cfg = {
            "patch_size": self.crop_shape,
            "patch_center_dist_from_border": np.array(self.crop_shape) // 2,
            "do_elastic_deform": False,
            "deformation_scale": (0.25, 0.25),
            "do_rotation": False,
            "angle_x": (-np.pi / 10, np.pi / 10),  # data is in z, y, x format
            "angle_y": (0, 2 * np.pi),
            "do_scale": True,
            "scale": (0.95, 1.05),
            "border_mode_data": "nearest",
            "random_crop": True,
        }

        # user-defined options
        spatial_transform_cfg.update(self.transform_cfg["spatial_transform"])
        return spatial_transform_cfg


# need different dataloaders for different modes

# BboxCroppedDataLoader for single-images_bbox
# SingleImagesCroppedDataLoader for single-images
# SingleStackCroppedDataLoader for stacks
# MultiStackCroppedDataLoader for multi-stacks

class SingleImagesDtaLoader(DataLoader):
    def __init__(self,
                 data: dict,
                 batch_size: int,
                 window_size: Union[tuple, list],
                 num_threads_in_multithreaded: int = 1,
                 shuffle: bool = False,
                 seed_for_shuffle: int = 123,
                 class_values: list = (0,),
                 ignore_last_channel=False):

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
        self.data_mode = data_mode
        self.window_size = window_size

        self.class_values = class_values
        self.ignore_last_channel = ignore_last_channel

        # setting random number generator
        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

        # asserting right number of images and labels
        if len(self.img_paths) != len(self.label_paths):
            raise ValueError("img and labels have different length, check dataset")

        self.img_volume, self.label_volume = self._load_single_images(
            frame_paths=self.img_paths,
            mask_paths=self.label_paths
        )

        self.steps_per_epoch = self._get_steps_per_epoch()

    def _get_steps_per_epoch(self) -> int:
        """
        returns the number of steps per epoch
        """
        img_px = np.prod(self.img_volume.shape)
        mask_px = np.prod(self.crop_shape)
        return img_px // (mask_px * self.batch_size)

    def _load_single_images(self,
                            frame_paths: list,
                            mask_paths: list) -> Tuple[np.ndarray, np.ndarray.np.ndarray]:
        frame_list = [self.load_img(img_to_load_path=fpath,
                                    normalize=self.normalize_inputs,
                                    ignore_last_channel=self.ignore_last_channel) for fpath in
                      frame_paths]
        frame_volume = np.stack(frame_list)
        del frame_list
        mask_list = [self.load_img(img_to_load_path=fpath,
                                   normalize=self.normalize_masks,
                                   ignore_last_channel=False,
                                   class_values=self.class_values) for fpath in mask_paths]

        mask_volume = np.stack(mask_list)
        del mask_list

        return frame_volume, mask_volume

    @staticmethod
    def load_img(
            img_to_load_path: str,
            normalize: bool = True,
            ignore_last_channel: bool = False,
            class_values: Union[list, tuple] = None,
    ):
        """
        load an image from path, apply basic transforms

        Parameters
        ----------
        img_to_load_path : str
            filepath to input image.
        normalize : bool, optional
            if True divides image values by np.iinfo(img.dtype).max. The default is True.
        ignore_last_channel : bool, optional
            If True doesn't consider the last channel of an image
            (RGB 2-channel images with empty third channel) . The default is False.
        class_values: list
            class values

        Returns
        -------
        img : np.array
            numpy array of loaded image.

        """
        try:
            img = skio.imread(img_to_load_path)

            if normalize and (class_values is None):
                norm_constant = np.iinfo(img.dtype).max
                img = img / norm_constant

            if class_values is not None:
                # image should be an indexmask
                # assert len(uniques) == n_classes + 1, "incorrect number of output classes in dataset"
                img_list = []
                for value in class_values:
                    val_img = np.where(img == value, 1, 0)
                    img_list.append(val_img)

                # we should have [y, x, ch]
                img = np.stack(img_list, axis=-1)

            # target shape = [y, x, ch]
            # adjust dimensions to target
            img_shape = img.shape
            if len(img_shape) == 2:
                logging.debug("expanding img shape")
                img = np.expand_dims(img, axis=-1)
            elif len(img_shape) in [3, 4]:
                pass

            # ignoring last channel if it's an RGB image with one empty ch
            if ignore_last_channel:
                img = img[..., :-1]
            return img
        except ValueError:
            raise ValueError(
                "This image failed: {}, check for anomalies".format(
                    str(img_to_load_path)
                )
            )

    @classmethod
    def get_random_crop(cls,
                        frame_volume: np.ndarray,
                        mask_volume: np.ndarray,
                        window_size: Union[list, tuple, np.ndarray]):

        plane = np.random.randint(low=0, high=frame_volume.shape[0])

        # outputs x_min, y_min, x_max, y_max
        crop_box = cls._get_bounding_box(frame_shape=frame_volume.shape[1:],
                                         window_size=window_size)
        # but stack ordering is [z,y,x]
        frame_crop = frame_volume[plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        mask_crop = mask_volume[plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        return frame_crop, mask_crop

    @staticmethod
    def _get_bounding_box(frame_shape: Union[list, tuple],
                          window_size: Union[list, tuple]) -> np.ndarray:

        lower_x = int(np.random.uniform(low=0, high=frame_shape[0] - window_size[0]))
        lower_y = int(np.random.uniform(low=0, high=frame_shape[1] - window_size[1]))

        upper_x = lower_x + window_size[0]
        upper_y = lower_y + window_size[1]

        crop_box = np.stack((lower_x, lower_y, upper_x, upper_y), axis=-1)

        return crop_box

    def generate_train_batch(self) -> dict:
        """generates a {"data": stack_img, "seg": stack_label} batch dictionary"""
        img_batch = []
        label_batch = []

        for _ in range(self.batch_size):
            crop_img, crop_label = self.get_random_crop(
                frame_volume=self.img_volume,
                mask_volume=self.label_volume,
                crop_shape=self.crop_shape,
            )
            img_batch.append(crop_img)
            label_batch.append(crop_label)

        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)

        # [batch, ch, z, y, x] shape is desired
        # adding channel in channel_first format
        stack_img = np.moveaxis(stack_img, source=-1, destination=1)
        stack_label = np.moveaxis(stack_label, source=-1, destination=1)

        return {"data": stack_img, "seg": stack_label}


class SingleStackCroppedDataLoader(DataLoader):
    def __init__(self,
                 data: dict,
                 batch_size: int,
                 window_size: Union[tuple, list],
                 num_threads_in_multithreaded: int = 1,
                 shuffle: bool = False,
                 seed_for_shuffle: int = 123,
                 class_values: list = (0,),
                 ignore_last_channel=False):

        super().__init__(
            data=data,
            batch_size=batch_size,
            num_threads_in_multithreaded=num_threads_in_multithreaded,
            shuffle=shuffle,
            seed_for_shuffle=seed_for_shuffle,
            infinite=infinite
        )

        self.imgs_path = self._data["img"]
        self.labels_paths = self._data["label"]

        self.rng_seed = seed_for_shuffle
        self.data_mode = data_mode
        self.window_size = window_size

        self.class_values = class_values
        self.ignore_last_channel = ignore_last_channel

        # setting random number generator
        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

        # asserting right number of images and labels
        if len(self.img_paths) != len(self.label_paths):
            raise ValueError("img and labels have different length, check dataset")

        self.img_volume, self.label_volume = self._load_single_stack(
            frames_path=self.imgs_path,
            masks_path=self.labels_paths
        )

        self.steps_per_epoch = self._get_steps_per_epoch()

    def _get_steps_per_epoch(self) -> int:
        """
        returns the number of steps per epoch
        """
        img_px = np.prod(self.img_volume.shape)
        mask_px = np.prod(self.crop_shape)
        return img_px // (mask_px * self.batch_size)

    def _load_single_stack(self,
                           frames_path: Path,
                           masks_path: Path) -> Tuple[np.ndarray, np.ndarray.np.ndarray]:
        frame_vol = self.load_img_stack(
            stack_to_load_path=frames_path,
            normalize=self.normalize_inputs,
            ignore_last_channel=self.ignore_last_channel
        )

        mask_vol = self.load_img_stack(
            stack_to_load_path=masks_path,
            normalize=self.normalize_inputs,
            ignore_last_channel=self.ignore_last_channel,
            class_values=self.class_valuess
        )

        return frame_vol, mask_vol

    @staticmethod
    def load_img_stack(
            stack_to_load_path: str,
            normalize: bool = True,
            ignore_last_channel: bool = True,
            class_values: Union[list, tuple] = None
    ):
        img = skio.imread(stack_to_load_path)

        if normalize and (class_values is None):
            norm_constant = np.iinfo(img.dtype).max
            img = img / norm_constant

        if class_values is None:
            img_list = []
            if class_values is not None:
                for value in class_values:
                    val_img = np.where(img == value, 1, 0)
                    img_list.append(val_img)

            img = np.stack(img_list, axis=-1)

        img_shape = img.shape
        if len(img_shape) == 3:
            logging.debug("expanding img shape")
            img = np.expand_dims(img, axis=-1)

        if ignore_last_channel:
            img = img[..., :-1]

        return img

    @classmethod
    def get_random_crop(cls,
                        frame_volume: np.ndarray,
                        mask_volume: np.ndarray,
                        window_size: Union[list, tuple, np.ndarray]):

        plane = np.random.randint(low=0, high=frame_volume.shape[0])

        # outputs x_min, y_min, x_max, y_max
        crop_box = cls._get_bounding_box(frame_shape=frame_volume.shape[1:],
                                         window_size=window_size)
        # but stack ordering is [z,y,x]
        frame_crop = frame_volume[plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        mask_crop = mask_volume[plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        return frame_crop, mask_crop

    @staticmethod
    def _get_bounding_box(frame_shape: Union[list, tuple],
                          window_size: Union[list, tuple]) -> np.ndarray:

        lower_x = int(np.random.uniform(low=0, high=frame_shape[0] - window_size[0]))
        lower_y = int(np.random.uniform(low=0, high=frame_shape[1] - window_size[1]))

        upper_x = lower_x + window_size[0]
        upper_y = lower_y + window_size[1]

        crop_box = np.stack((lower_x, lower_y, upper_x, upper_y), axis=-1)

        return crop_box

    def generate_train_batch(self) -> dict:
        """generates a {"data": stack_img, "seg": stack_label} batch dictionary"""
        img_batch = []
        label_batch = []

        for _ in range(self.batch_size):
            crop_img, crop_label = self.get_random_crop(
                frame_volume=self.img_volume,
                mask_volume=self.label_volume,
                crop_shape=self.crop_shape,
            )
            img_batch.append(crop_img)
            label_batch.append(crop_label)

        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)

        # [batch, ch, z, y, x] shape is desired
        # adding channel in channel_first format
        stack_img = np.moveaxis(stack_img, source=-1, destination=1)
        stack_label = np.moveaxis(stack_label, source=-1, destination=1)

        return {"data": stack_img, "seg": stack_label}


class MultiStackCroppedDataLoader(DataLoader):
    def __init__(self,
                 data: dict,
                 batch_size: int,
                 window_size: Union[tuple, list],
                 num_threads_in_multithreaded: int = 1,
                 shuffle: bool = False,
                 seed_for_shuffle: int = 123,
                 class_values: list = (0,),
                 ignore_last_channel: bool = False,
                 normalize_inputs: bool = True):

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
        self.batch_size = batch_size
        self.num_threads_in_multithreaded = num_threads_in_multithreaded
        self.shuffle = shuffle

        self.normalize_inputs = normalize_inputs
        self.rng_seed = seed_for_shuffle
        self.data_mode = data_mode
        self.window_size = window_size

        self.class_values = class_values
        self.ignore_last_channel = ignore_last_channel

        # setting random number generator
        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

        # asserting right number of images and labels
        if len(self.img_paths) != len(self.label_paths):
            raise ValueError("img and labels have different length, check dataset")

        self.volume_list = self._load_volumes()
        # self.single_stack_dataloaders = self._get_single_stack_dataloaders()

    def _load_volumes(self) -> list:
        volume_list = []
        for idx, img_path in enumerate(self.img_paths):
            img = SingleStackCroppedDataLoader.load_img_stack(stack_to_load_path=img_path,
                                                              normalize=self.normalize_inputs,
                                                              ignore_last_channel=self.ignore_last_channel,
                                                              class_values=None)

            label = SingleStackCroppedDataLoader.load_img_stack(stack_to_load_path=img_path,
                                                                normalize=self.normalize_inputs,
                                                                ignore_last_channel=False,
                                                                class_values=self.class_values)

            volume_list.append((img, label))
        return volume_list

    @property
    def generate_train_batch(self) -> dict:
        img_batch = []
        label_batch = []

        for _ in range(self.batch_size):
            vol_idx = np.randint(low=0, high=len(self.volume_list))

            frame_crop, mask_crop = SingleStackCroppedDataLoader.get_random_crop(
                frame_volume=self.volume_list[vol_idx][0],
                mask_volume=self.volume_list[vol_idx][1],
                window_size=self.window_size
            )
            img_batch.append(frame_crop)
            label_batch.append(mask_crop)
        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)

        stack_img = np.moveaxis(stack_img, source=-1, destination=1)
        stack_label = np.moveaxis(stack_label, source=-1, destination=1)

        return {"data": stack_img, "seg": stack_label}

    # def _get_single_stack_dataloaders(self) -> list:
    #     """return a list of dataloaders, one for each stack"""
    #     dl_list = []
    #     for idx, img_path in enumerate(self.img_paths):
    #         data_dict = {
    #             "img": img_path,
    #             "label": self.label_paths[idx]
    #         }
    #         dl = SingleStackCroppedDataLoader(data=data_dict,
    #                                           batch_size=self.batch_size,
    #                                           window_size=self.window_size,
    #                                           num_threads_in_multithreaded=self.num_threads_in_multithreaded,
    #                                           shuffle=self.shuffle,
    #                                           seed_for_shuffle=self.rng_seed,
    #                                           class_values=self.class_values,
    #                                           ignore_last_channel=self.ignore_last_channel)
    #         dl_list.append(dl)
    #
    #     return dl_list


class BboxCroppedDataLoader(DataLoader):

    def __init__(self,
                 data: dict,
                 batch_size: int,
                 window_size: Union[tuple, list],
                 num_threads_in_multithreaded: int = 1,
                 shuffle: bool = False,
                 seed_for_shuffle: int = 123,
                 class_values: list = (0,),
                 ignore_last_channel: bool = False,
                 normalize_inputs: bool = True):

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
        self.data_mode = data_mode
        self.window_size = window_size

        self.class_values = class_values
        self.ignore_last_channel = ignore_last_channel

        self.normalize_inputs = normalize_inputs

        # setting random number generator
        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

        # asserting right number of images and labels
        if len(self.img_paths) != len(self.label_paths):
            raise ValueError("img and labels have different length, check dataset")

        self.csv_paths = [Path(fpath).parent.joinpath(Path(fpath).name + ".csv") for fpath in self.label_paths]

        # load image volumes
        self.img_volume, self.label_volume, self.bboxes = self._load_single_images_csv(
            frame_paths=self.img_paths,
            mask_paths=self.label_paths,
            csv_paths=self.csv_paths
        )

        self.steps_per_epoch = self._get_steps_per_epoch()

    def _get_steps_per_epoch(self) -> int:
        """
        returns the number of steps per epoch based on the total annotated pixels
        calculated with the csv bboxes
        """
        annotated_pixels = 0
        for bbox in self.bboxes:
            widht = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            annotated_px_bbox = widht * height
            annotated_pixels += annotated_px_bbox
        mask_px = np.prod(self.crop_shape)
        return annotated_pixels // (mask_px * self.batch_size)

    @staticmethod
    def _parse_csv(csv_path: Path) -> list:
        """return the first row of a csv file"""
        out_list = []
        with csv_path.open(mode="r") as infile:
            reader = csv.reader(infile)
            for row in reader:
                row_ints = [int(elem) for elem in row]
                out_list.append(row_ints)
        return out_list[0]

    def _load_single_images_csv(self,
                                frame_paths: list,
                                mask_paths: list,
                                csv_paths: list) -> Tuple[np.ndarray, np.ndarray.np.ndarray]:
        csv_lists = []
        for fpath in csv_paths:
            if fpath.is_file():
                csv_lists.append(self._parse_csv(fpath))
            else:
                csv_lists.append([0, -1, 0, -1])
        frame_list = [SingleImagesDtaLoader.load_img(img_to_load_path=fpath,
                                                     normalize=self.normalize_inputs,
                                                     ignore_last_channel=self.ignore_last_channel) for fpath in
                      frame_paths]
        frame_volume = np.stack(frame_list)
        del frame_list
        mask_list = [SingleImagesDtaLoader.load_img(img_to_load_path=fpath,
                                                    normalize=self.normalize_masks,
                                                    ignore_last_channel=False,
                                                    class_values=self.class_values) for fpath in mask_paths]

        mask_volume = np.stack(mask_list)
        del mask_list
        b_boxes = np.array(csv_lists)
        return frame_volume, mask_volume, b_boxes

    @classmethod
    def get_random_crop(cls,
                        frame_volume: np.ndarray,
                        mask_volume: np.ndarray,
                        window_size: Union[list, tuple, np.ndarray],
                        b_boxes: Union[list, tuple, np.ndarray]):

        n_planes = frame_volume.shape[0]
        assert len(b_boxes) == (frame_volume.shape[0])
        plane_weights = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in b_boxes])
        plane_weights = plane_weights / np.sum(plane_weights)
        planes = list(range(n_planes))

        plane = np.random.choice(a=planes, p=plane_weights)
        # plane = np.random.randint(low=0, high=frame_volume.shape[0])

        # outputs x_min, y_min, x_max, y_max
        crop_box = cls._get_bounding_box(frame_shape=frame_volume.shape[1:],
                                         window_size=window_size,
                                         b_box=b_boxes[plane])
        # but stack ordering is [z,y,x]
        frame_crop = frame_volume[plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        mask_crop = mask_volume[plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        return frame_crop, mask_crop

    @staticmethod
    def _get_bounding_box(frame_shape: Union[list, tuple],
                          window_size: Union[list, tuple],
                          b_box: np.ndarray = None) -> np.ndarray:
        if b_box is None:
            b_box = [0, frame_shape[0], 0, frame_shape[1]]

        if not (np.array(crop_shape) > np.array(frame_shape)[1:2]).any():
            x_low = b_box[0]
            x_high = b_box[2] - window_size[0]

            y_low = b_box[1]
            y_high = b_box[3] - window_size[1]
        else:
            raise ValueError("crop_shape is larger than frame_shape")
        lower_x = int(np.random.uniform(low=x_low, high=x_high))
        lower_y = int(np.random.uniform(low=y_low, high=y_high))

        upper_x = lower_x + window_size[0]
        upper_y = lower_y + window_size[1]
        # crop_boxes = (lower_x, lower_y, upper_x, upper_y)
        crop_boxes = np.stack((lower_x, lower_y, upper_x, upper_y), axis=-1)
        # crop_boxes = np.column_stack((lower_x, lower_y, upper_x, upper_y))

        return crop_boxes

    def generate_train_batch(self) -> dict:
        """generates a {"data": stack_img, "seg": stack_label} batch dictionary"""
        img_batch = []
        label_batch = []

        for _ in range(self.batch_size):
            crop_img, crop_label = self.get_random_crop(
                frame_volume=self.img_volume,
                mask_volume=self.label_volume,
                crop_shape=self.crop_shape,
                b_boxes=self.bboxes
            )
            img_batch.append(crop_img)
            label_batch.append(crop_label)

        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)

        # [batch, ch, z, y, x] shape is desired
        # adding channel in channel_first format
        stack_img = np.moveaxis(stack_img, source=-1, destination=1)
        stack_label = np.moveaxis(stack_label, source=-1, destination=1)

        return {"data": stack_img, "seg": stack_label}
