import logging
import gc
from pathlib import Path
from typing import Union, Callable, Generator, Tuple

from batchgenerators.dataloading import (
    MultiThreadedAugmenter,
    SingleThreadedAugmenter)
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

import numpy as np

from neuroseg.datagens.datagenbase import DataGenBase
from neuroseg.config import TrainConfig, PredictConfig
from neuroseg.utils import load_volume


class DataGen3D(DataGenBase):
    """3D data pipeline, inherits from dataGenBase
    based on BatchGenerators

    Attributes
    ----------
    data: generator
        implements the self.data required method, gives access to generated data
        to be used with Keras model.fit()
    iter: iterator
        iterator to the generated data
    steps_per_epoch: int
        defines the number of steps to complete an epoch
        to be used in model.fit()"""

    def __init__(self,
                 config: Union[TrainConfig, PredictConfig],
                 partition: str = "train",
                 data_augmentation: bool = True,
                 verbose: bool = False):
        """
        3D data generator
        implementation based on BatchGenerators

        Parameters
        ----------
        config : neuroseg.TrainConfig or neuroseg.PredictConfig
            Neuroseg Config object.
        partition : str, optional
            Data partition, can be chosen in
            "train", "validation", "test".
            The default is "train".
        data_augmentation : bool, optional
            Enables data agumentation pipeline.
            You have to disable it for validation or test.
            The default is True.
        normalize_inputs : bool, optional
            if True divides image values by np.iinfo(img.dtype).max. The default is True.
        verbose : bool, optional
            Enable additional verbosity. The default is False.
            

        """

        super().__init__(config=config,
                         partition=partition,
                         data_augmentation=data_augmentation,
                         verbose=verbose)

        # self._parse_single_stack_paths()
        # self._load_volumes()

        self.shuffle = self.config.da_shuffle
        self.seed = self.config.da_seed
        # self.pre_crop_scales = self.config.da_pre_crop_scales

        self.data = self._setup_gen()
        self.iter = self.data.__iter__
        self.steps_per_epoch = self._get_steps_per_epoch(self.volume_shape, self.crop_shape, self.batch_size)

    @staticmethod
    def _normalize_stack(stack_arr: np.ndarray,
                         norm: Union[int, float] = 255) -> np.ndarray:
        """divide stack by normalization constant"""
        return (stack_arr / norm).astype(np.float32)

    def _get_data_dict(self) -> dict:
        if self.dataset_mode == "stack":
            self.frames_path = self.frames_paths[0]
            self.masks_path = self.masks_paths[0]

            data = {"img": self.frames_path,
                    "label": self.masks_path}
        elif self.dataset_mode == "multi_stack":
            data = {"img": self.frames_paths,
                    "label": self.masks_paths}
        elif self.dataset_mode == "single_images":
            data = {"img": self.frames_paths,
                    "label": self.masks_paths}
        else:
            raise ValueError(f"{self.dataset_mode} is not supported")
        return data

    def _spatial_transform_cfg(self) -> dict:
        """spatial transform"""
        # default options
        spatial_transform_cfg = {
            "patch_size": self.crop_shape,
            "patch_center_dist_from_border": np.array(self.crop_shape) // 2,
            "do_elastic_deform": False,
            "deformation_scale": (0, 0.25, 0.25),
            "do_rotation": False,
            "angle_x": (-np.pi / 10, np.pi / 10),  # data is in z, y, x format
            "angle_y": (0, 2 * np.pi),
            "angle_z": (0, 2 * np.pi),
            "do_scale": True,
            "scale": (0.95, 1.05),
            "border_mode_data": "nearest",
            "random_crop": True,
        }

        # user-defined options
        spatial_transform_cfg.update(self.transform_cfg["spatial_transform"])
        return spatial_transform_cfg

    def _brightness_multiplicative_transform_cfg(self) -> dict:
        """brightness multiplicative transform configurator"""
        brightness_multiplicative_transform_cfg = {
            "multiplier_range": (0.9, 1.1),
            "per_channel": True,
            "p_per_sample": 0.15}

        brightness_multiplicative_transform_cfg.update(self.transform_cfg["brightness_multiplicative_transform"])
        return brightness_multiplicative_transform_cfg

    def _brightness_transform_cfg(self) -> dict:
        """brightness additive transform configurator"""
        brightness_transform_cfg = {
            "p_per_sample": 0.15,
            "mu": 0,
            "sigma": 0.01,
        }

        brightness_transform_cfg.update(self.transform_cfg["brightness_transform"])
        return brightness_transform_cfg

    def _rot90_transform_cfg(self) -> dict:
        """rigid 90 degree rotation transform configurator"""

        if len(set(self.crop_shape)) == 1:
            rotation_axes = (0, 1, 2)
        else:
            rotation_axes = (1, 2)

        rot90_transform_cfg = {
            "num_rot": (1, 2, 3),
            # "axes": (0,1,2),
            "axes": rotation_axes,
            "p_per_sample": 0.15
        }

        rot90_transform_cfg.update(self.transform_cfg["rot90_transform"])
        return rot90_transform_cfg

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

    def _mirror_transform_cfg(self) -> dict:
        """mirror transform configurator"""
        mirror_transform_cfg = {
            "axes": (0, 1, 2)
        }

        mirror_transform_cfg.update(self.transform_cfg["mirror_transform"])
        return mirror_transform_cfg

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

    @classmethod
    def _get_transform_fn(cls, transform: str) -> Callable:
        """convert transform string to transform function"""
        TRANSFORM_FNS = {
            "brightness_transform": BrightnessTransform,
            "brightness_multiplicative_transform": BrightnessMultiplicativeTransform,
            "gamma_transform": GammaTransform,
            "gaussian_noise_transform": GaussianNoiseTransform,
            "gaussian_blur_transform": GaussianBlurTransform,
            "mirror_transform": MirrorTransform,
            "rot90_transform": Rot90Transform,
            "spatial_transform": SpatialTransform_2,
        }

        if transform not in TRANSFORM_FNS:
            raise NotImplementedError("transform {} not supported".format(transform))

        return TRANSFORM_FNS[transform] if transform in TRANSFORM_FNS else None

    def _get_transform_cfg(self, transform: str) -> dict:
        """convert transform string to transform configurator function"""
        TRANSFORM_CFGS = {
            "brightness_transform": self._brightness_transform_cfg(),
            "brightness_multiplicative_transform": self._brightness_multiplicative_transform_cfg(),
            "gamma_transform": self._gamma_transform_cfg(),
            "gaussian_noise_transform": self._gaussian_noise_transform_cfg(),
            "gaussian_blur_transform": self._gaussian_blur_transform_cfg(),
            "mirror_transform": self._mirror_transform_cfg(),
            "rot90_transform": self._rot90_transform_cfg(),
            "spatial_transform": self._spatial_transform_cfg(),
        }
        if transform not in TRANSFORM_CFGS:
            raise NotImplementedError("transform {} not supported".format(transform))

        return TRANSFORM_CFGS[transform] if transform in TRANSFORM_CFGS else None

    def _get_augment_transforms(self) -> list:
        """returns a list of pre-configured transform functions
        basically generates a pipeline of sequential transforms"""
        transforms = []
        for transform in self.transforms:
            transform_fn = self._get_transform_fn(transform)
            transform_cfg = self._get_transform_cfg(transform)

            transforms.append(transform_fn(**transform_cfg))

        return transforms

    def _get_transform_chain(self) -> Compose:
        """generate a Compose() global transform"""
        if self.data_augmentation:
            transforms = self._get_augment_transforms()
        else:
            transforms = [RandomCropTransform(crop_size=self.crop_shape)]

        # clip data values to [0. 1.]
        clip_min = 0. if self.normalize_inputs else np.finfo(np.float32).min
        clip_max = 1.01 if self.normalize_inputs else np.finfo(np.float32).max

        transforms.append(ClipValueRange(min=clip_min, max=clip_max))
        return Compose(transforms)

    def _setup_gen(self):
        """returns a python generator to be used with keras
        supports both SingleThread and MultiThread BatchGenerators augmenters"""

        self.data_dict = self._get_data_dict()

        if self.dataset_mode in ["single_images", "stack"]:

            self.dataLoader = CroppedDataLoaderBG(
                data=self.data_dict,
                batch_size=self.batch_size,
                crop_shape=self.crop_shape,
                data_mode=self.dataset_mode,
                num_threads_in_multithreaded=self.threads,
                shuffle=self.shuffle,
                seed_for_shuffle=self.seed,
                infinite=False,
                normalize_inputs=self.normalize_inputs,
                normalize_masks=self.normalize_masks,
                soft_labels=self.soft_labels,
                positive_class_value=self.positive_class_value
            )
        elif self.dataset_mode == "multi_stack":
            self.dataLoader = MultiCroppedDataLoaderBG(
                data=self.data_dict,
                batch_size=self.batch_size,
                crop_shape=self.crop_shape,
                normalize_inputs=self.normalize_inputs,
                num_threads_in_multithreaded=self.threads,
                shuffle=self.shuffle,
                seed_for_shuffle=self.seed,
                infinite=False)
        else:
            raise NotImplementedError(self.dataset_mode)

        self.volume_shape = self.dataLoader.volume_shape
        self.composed_transform = self._get_transform_chain()

        if self.single_thread:
            self.gen = SingleThreadedAugmenter(self.dataLoader, self.composed_transform)
        else:
            self.gen = MultiThreadedAugmenter(self.dataLoader,
                                              self.composed_transform,
                                              self.threads)

        return self._get_keras_gen(self.gen, channel_last=True)

    @staticmethod
    def _get_steps_per_epoch(frame_shape: Union[list, tuple],
                             crop_shape: Union[list, tuple],
                             batch_size: int) -> int:
        """calculate the number of steps per epoch"""
        frame_px = np.prod(frame_shape)
        crop_px = np.prod(crop_shape)
        return int(np.ceil((frame_px / crop_px) / float(batch_size)))

    @classmethod
    def _get_keras_gen(cls,
                       batchgen: Union[SingleThreadedAugmenter, MultiThreadedAugmenter],
                       channel_last: bool = True) -> Generator:
        """
        Adapter for BatchGenerator generators to be used with keras
        yields (frames, masks) batched pairs
        

        Parameters
        ----------
        batchgen : BatchGenerators generator
            generator of {"data": data_npy, "seg": seg_mask_npy} dicts.
        channel_last : bool, optional
            If True converts to channel_last data format. The default is True.

        Yields
        ------
        frames : npy array
            (batch, z, y, x, ch) frame npy tensor.
        masks : TYPE
            (batch, z, y, x) mask tensor.
            TODO: check if dims are correct
        """
        while True:
            batch_dict = next(batchgen)
            frames = batch_dict["data"]
            masks = batch_dict["seg"]

            if channel_last:
                frames = cls._to_channel_last(frames)
                masks = cls._to_channel_last(masks)

            yield frames, masks

    @staticmethod
    def _to_channel_last(input_tensor: np.ndarray) -> np.ndarray:
        """convert (batch, ch, z, y, x) tensor to (batch, z, y, x, ch) tensor"""
        return np.moveaxis(input_tensor, source=1, destination=-1)


class MultiCroppedDataLoaderBG(DataLoader):

    def __init__(
            self,
            data: dict,
            batch_size: int,
            crop_shape: Union[tuple, list],
            num_threads_in_multithreaded: int = 1,
            shuffle: bool = False,
            seed_for_shuffle: int = 12,
            infinite: bool = False,
            normalize_inputs: bool = True,
            normmalize_masks: bool = False,
            soft_labels: bool = False,
            positive_class_value: int = 255):

        # data = {"img": [img_paths], "label": [label_paths]}}
        super().__init__(
            data=data,
            batch_size=batch_size,
            num_threads_in_multithreaded=num_threads_in_multithreaded,
            shuffle=shuffle,
            seed_for_shuffle=seed_for_shuffle,
            infinite=infinite)

        self.crop_shape = crop_shape
        self.img_paths = self._data["img"]
        self.label_paths = self._data["label"]
        self.rng_seed = seed_for_shuffle
        self.normalize_inputs = normalize_inputs
        self.normalize_masks = normmalize_masks
        self.soft_labels = soft_labels
        self.positive_class_value = positive_class_value

        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

        if len(self.img_paths) != len(self.label_paths):
            raise ValueError("img and labels have different length, check dataset")

        self.img_path_dict = {Path(fpath).name: fpath for fpath in self.img_paths}
        self.label_path_dict = {Path(fpath).name: fpath for fpath in self.label_paths}

        if set(self.img_path_dict.keys()) != set(self.label_path_dict.keys()):
            raise ValueError("imag and labels have different names")

        self.path_dict = self._get_nested_path_dict()
        self.volume_dict = self._load_volumes()
        self.volume_names = list(self.volume_dict.keys())
        self.volume_shape_dict = {key: vol_dict_item["img"].shape[1:] for key, vol_dict_item in
                                  self.volume_dict.items()}
        self.volume_shape = self.volume_shape_dict[list(self.volume_shape_dict.keys())[0]][1:]

    def _get_nested_path_dict(self) -> dict:
        nested_dict = {}
        for key in self.img_path_dict.keys():
            key_dict = {
                "img": self.img_path_dict[key],
                "label": self.label_path_dict[key]}
            nested_dict[key] = key_dict
        return nested_dict

    def _load_volumes(self) -> dict:
        volume_dict = {}
        for key, vol_path_dict in self.path_dict.items():
            img_path = vol_path_dict["img"]
            label_path = vol_path_dict["label"]

            # NOTE LOAD_VOLUME_COUPLE IS NOW IN SINGLE CROPPED DATA LOADER

            img_volume, label_volume = CroppedDataLoaderBG._load_volume_couple(
                img_path=img_path,
                label_path=label_path,
                normalize_inputs=self.normalize_inputs,
                normalize_masks=self.normalize_masks,
                soft_labels=self.soft_labels,
                data_mode="stack",
                label_positive_class_value=self.positive_class_value
            )

            key_volume_dict = {
                "img": img_volume.astype(np.float32),
                "label": label_volume.astype(np.float32)}

            volume_dict[key] = key_volume_dict
        return volume_dict

    @staticmethod
    def _normalize_stack(stack_arr: np.ndarray,
                         norm: Union[int, float] = 255) -> np.ndarray:
        return (stack_arr / norm).astype(np.float32)

    def generate_train_batch(self) -> dict:
        img_batch = []
        label_batch = []
        for idx in range(self.batch_size):
            crop_img, crop_label = self._get_random_crop_multi()
            img_batch.append(crop_img)
            label_batch.append(crop_label)

        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)

        return {"data": stack_img, "seg": stack_label}

    def _get_random_crop_multi(self):
        vol_name = np.random.choice(self.volume_names)

        return CroppedDataLoaderBG._get_random_crop(
            frames_volume=self.volume_dict[vol_name]["img"],
            masks_volume=self.volume_dict[vol_name]["label"],
            volume_shape=self.volume_shape_dict[vol_name],
            crop_shape=self.crop_shape)


class CroppedDataLoaderBG(DataLoader):
    """expands DataLoader class
    used for loading pre-applied random cropping to data"""

    def __init__(
            self,
            data: dict,
            batch_size: int,
            crop_shape: Union[tuple, list],
            data_mode: str = "stack",
            num_threads_in_multithreaded: int = 1,
            shuffle: bool = False,
            seed_for_shuffle: int = 123,
            infinite: bool = False,
            normalize_inputs: bool = True,
            normalize_masks: bool = False,
            soft_labels: bool = False,
            positive_class_value: Union[int, float] = 255):
        """
        CroppedDataLoaderBG
        
        loads cropped data from npy volumes

        Parameters
        ----------
        data : dict
            {"img": img_paths, "label": label_paths} dict.
        batch_size : int
            batch size.
        crop_shape : tuple or list
            [z,y,x] crop shape.
        num_threads_in_multithreaded : int, optional
            assign a number of threads in mutlithreaded mode.
            The default for data fetching is 1, multithreading is implemented in augmentation.
        shuffle : bool, optional
            Enable shufffle. The default is False.
        seed_for_shuffle : int, optional
            self-explanatory. The default is 123.
        infinite : bool, optional
            create an infinite generator. The default is False.
        """

        super().__init__(
            data=data,
            batch_size=batch_size,
            num_threads_in_multithreaded=num_threads_in_multithreaded,
            shuffle=shuffle,
            seed_for_shuffle=seed_for_shuffle,
            infinite=infinite
        )

        self.img_path = data["img"]
        self.labels_path = data["label"]

        self.data_mode = data_mode
        self.normalize_inputs = normalize_inputs
        self.normalize_masks = normalize_masks
        self.soft_labels = soft_labels
        self.positive_class_value = positive_class_value

        if self.data_mode not in ["single_images", "stack"]:
            raise ValueError("CroppedDataLoaderBG does not suppport data mode {} \
                             maybe you wanted to use MultiVolumeDataLoaderBG?".format(self.data_mode))

        self.frames_volume, self.masks_volume = self._load_volume_couple(
            img_path=self.img_path,
            label_path=self.labels_path,
            normalize_inputs=self.normalize_inputs,
            normalize_masks=self.normalize_masks,
            soft_labels=self.soft_labels,
            data_mode=self.data_mode,
            label_positive_class_value=self.positive_class_value)

        self.volume_shape = self.frames_volume.shape[1:]
        self.crop_shape = crop_shape
        # self.pre_crop_scales = pre_crop_scales
        self.rng_seed = seed_for_shuffle

        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

    @classmethod
    def _load_volume_couple(cls,
                            img_path: Path,
                            label_path: Path,
                            normalize_inputs: bool,
                            normalize_masks: bool = False,
                            soft_labels: bool = False,
                            data_mode: str = "stack",
                            label_positive_class_value: Union[int, float] = 255) -> Tuple[np.ndarray, np.ndarray]:

        img_volume, img_norm = load_volume(img_path,
                                           data_mode=data_mode,
                                           ignore_last_channel=False,
                                           return_norm=True)

        label_volume, label_norm = load_volume(label_path,
                                               data_mode=data_mode,
                                               ignore_last_channel=False,
                                               return_norm=True)
        if normalize_inputs:
            img_volume = cls._normalize_stack(img_volume,
                                              norm=img_norm)
        if normalize_masks:
            label_volume = cls._normalize_stack(label_volume, label_norm)

        if not soft_labels:
            label_volume = np.where(label_volume >= label_positive_class_value, 1, 0).astype(img_volume.dtype)

        img_volume, label_volume = cls._adjust_stack_dims(img_volume, label_volume, to_channel_first=True)
        return img_volume, label_volume

    @classmethod
    def _adjust_stack_dims(cls,
                           img_volume: np.ndarray,
                           labels_volume: np.ndarray,
                           to_channel_first: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        # determine if third channel is empty
        img_volume_shape = img_volume.shape

        assert len(img_volume_shape) == 4, f"Image volume has shape {img_volume_shape}, it should be [z, y, x, ch]"

        if img_volume_shape[-1] == 3:
            if len(np.unique(img_volume[..., 2])) == 1:
                logging.warning(
                    "last channel of input volume is empty, you should use ignore_last_channel: true in config to avoid this warning")
                logging.warning("ignoring last channel")
                img_volume = img_volume[..., :-1]

        # img_volume = cls._expand_dims_if_needed(img_volume)
        # labels_volume = cls._expand_dims_if_needed(labels_volume)

        if to_channel_first:
            img_volume = np.moveaxis(img_volume, -1, 0)
            labels_volume = np.moveaxis(labels_volume, -1, 0)

        return img_volume, labels_volume

    # @staticmethod
    # def _expand_dims_if_needed(stack):
    #     stack_shape = stack.shape

    #     if len(stack_shape) == 4:
    #         pass
    #     elif len(stack_shape) == 3:
    #         stack = np.expand_dims(stack, axis=-1)
    #     else:
    #         raise ValueError("invalid stack_shape {}".format(stack_shape))

    #     return stack

    @staticmethod
    def _normalize_stack(stack_arr: np.ndarray, norm: Union[int, float] = 255) -> np.ndarray:
        return (stack_arr / norm).astype(np.float32)

    # TODO: consider if reviving pre-scaling is worth it or not

    #     self.scaled_crop_shape = self._get_scaled_crop_shape(self.crop_shape,
    #                                                          self.pre_crop_scales)

    #     assert (self.scaled_crop_shape < self.volume_shape), "pre_crop_scale is too large, cannot crop inside volume"

    # @staticmethod
    # def _get_scaled_crop_shape(crop_shape, pre_crop_scales):
    #     assert (np.array(pre_crop_scales) >= 1).all(), "pre_crop_scale must be >1"
    #     scaled_crop_shape = np.multiply(np.array(crop_shape), np.array(pre_crop_scales))
    #     scaled_crop_shape = np.ceil(scaled_crop_shape).astype(np.uint8)
    #     return tuple(scaled_crop_shape)

    # def get_len(self):
    #     img_pixels = np.prod(self.volume_shape)
    #     crop_pixels = np.prod(self.crop_shape)
    #     return int(np.ceil((img_pixels/crop_pixels)/float(self.batch_size)))

    def generate_train_batch(self) -> dict:
        """return a {"data": crooped_data, "seg": crooped_seg} dict"""

        img_batch = []
        label_batch = []

        for i in range(self.batch_size):
            crop_img, crop_label = self._get_random_crop(
                frames_volume=self.frames_volume,
                masks_volume=self.masks_volume,
                volume_shape=self.volume_shape,
                crop_shape=self.crop_shape)
            img_batch.append(crop_img)
            label_batch.append(crop_label)

        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)

        return {"data": stack_img, "seg": stack_label}

    @staticmethod
    def _get_random_crop(frames_volume: np.ndarray,
                         masks_volume: np.ndarray,
                         volume_shape: Union[tuple, list],
                         crop_shape: Union[tuple, list]) -> Tuple[np.ndarray, np.ndarray]:
        volume_shape = list(volume_shape)  # casting to list
        crop_shape = list(crop_shape)
        """get a random crop_img, crop_label tuple"""
        z_shape, y_shape, x_shape = volume_shape

        assert len(crop_shape) == len(
            volume_shape), f"crop_shape {crop_shape} and volume_shape {volume_shape} have different dimensionalities"
        # assert we're not trying to crop something larger than volume 
        assert volume_shape > crop_shape, f"crop_shape {crop_shape} > volume_shape {volume_shape}"

        z_0 = np.random.randint(low=0, high=z_shape - crop_shape[0])
        y_0 = np.random.randint(low=0, high=y_shape - crop_shape[1])
        x_0 = np.random.randint(low=0, high=x_shape - crop_shape[2])

        crop_img = frames_volume[
                   :,
                   z_0: z_0 + crop_shape[0],
                   y_0: y_0 + crop_shape[1],
                   x_0: x_0 + crop_shape[2]
                   ]

        crop_label = masks_volume[
                     :,
                     z_0: z_0 + crop_shape[0],
                     y_0: y_0 + crop_shape[1],
                     x_0: x_0 + crop_shape[2]
                     ]

        return crop_img, crop_label
