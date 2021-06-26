from pathlib import Path
from functools import partial
import csv
import logging
from typing import Union, Tuple, List, Callable

import numpy as np
from skimage import io as skio
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

from neuroseg.config import TrainConfig, PredictConfig
from neuroseg.datagens.datagenbase import DataGenBase


class DataGen2D(DataGenBase):
    def __init__(self,
                 config: Union[TrainConfig, PredictConfig],
                 partition: str = "train",
                 data_augmentation: bool = True,
                 verbose: bool = False):
        super().__init__(config=config,
                         partition=partition,
                         data_augmentation=data_augmentation,
                         verbose=verbose)
        self.shuffle = self.config.da_shuffle
        self.seed = self.config.da_seed

        self.ignore_last_channel = self.config.ignore_last_channel

        data_dict = {"img": self.frames_paths,
                     "label": self.masks_paths}

        self.dataLoader = BboxCroppedDataLoader(
            data=data_dict,
            batch_size=self.batch_size,
            crop_shape=self.crop_shape,
            data_mode=self.dataset_mode,
            num_threads_in_multithreaded=self.threads,
            shuffle=self.shuffle,
            seed_for_shuffle=self.seed,
            infinite=True,
            normalize_inputs=self.normalize_inputs,
            normalize_masks=self.normalize_masks,
            positive_class_value=self.positive_class_value,
            ignore_last_channel=self.ignore_last_channel
        )

        self.steps_per_epoch = self.dataLoader.steps_per_epoch
        self.composed_transform = self.get_transform_chain()

        if self.single_thread:
            self.gen = SingleThreadedAugmenter(self.dataLoader, self.composed_transform)
        else:
            self.gen = MultiThreadedAugmenter(self.dataLoader,
                                              self.composed_transform,
                                              self.threads)

        self.data = self.get_keras_gen(self.gen, channel_last=True)
        self.iter = self.data.__iter__()

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

            yield frames, masks

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

    def get_transform_cfg(self, transform: str) -> dict:
        TRANSFORM_CFGS = {
            "brightness_transform": self._brightness_transform_cfg,
            "brightness_multiplicative_transform": self._brightness_multiplicative_transform_cfg,
            "gamma_transform": self._gamma_transform_cfg,
            "gaussian_noise_transform": self._gaussian_noise_transform_cfg,
            "gaussian_blur_transform": self._gaussian_blur_transform_cfg,
            "mirror_transform": self._mirror_transform_cfg,
            "rot90_transform": self._rot90_transform_cfg,
            "spatial_transform": self._spatial_transform_cfg,
        }
        if transform not in TRANSFORM_CFGS:
            raise NotImplementedError("transform {} not supported".format(transform))

        return TRANSFORM_CFGS[transform]() if transform in TRANSFORM_CFGS else None

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


class BboxCroppedDataLoader(DataLoader):

    def __init__(self,
                 data: dict,
                 batch_size: int,
                 crop_shape: Union[tuple, list],
                 data_mode: str = "single_images",
                 use_bboxes: bool = False,
                 num_threads_in_multithreaded: int = 1,
                 shuffle: bool = False,
                 seed_for_shuffle: int = 123,
                 infinite: bool = False,
                 normalize_inputs: bool = True,
                 normalize_masks: bool = False,
                 # soft_labels: bool = False,
                 positive_class_value: Union[int, float] = 255,
                 ignore_last_channel: bool = False):

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
        self.normalize_inputs = normalize_inputs
        self.normalize_masks = normalize_masks
        # self.soft_labels = soft_labels
        self.positive_class_value = positive_class_value
        self.data_mode = data_mode
        self.crop_shape = crop_shape
        self.use_bboxes = use_bboxes
        self.ignore_last_channel = ignore_last_channel

        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)

        if len(self.img_paths) != len(self.label_paths):
            raise ValueError("img and labels have different length, check dataset")

        if use_bboxes is not None:
            self.csv_paths = [Path(fpath).parent.joinpath(Path(fpath).name + ".csv") for fpath in self.label_paths]
        else:
            self.csv_paths = None

        self.img_volume, self.label_volume, self.bboxes = self._single_images_load(
            frame_paths=self.img_paths,
            mask_paths=self.label_paths,
            csv_paths=self.csv_paths
        )

        annotated_pixels = 0
        for bbox in self.bboxes:
            widht = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            annotated_px_bbox = widht * height
            annotated_pixels += annotated_px_bbox
        mask_px = np.prod(self.crop_shape)
        self.steps_per_epoch = annotated_pixels // (mask_px * self.batch_size)

        # # apply normalization
        # NORMALIZATION IS ALREADY APPLIED IN LOAD_IMG
        # if self.normalize_inputs:
        #     norm = np.iinfo(self.img_volume.dtype).max
        #     self.img_volume = self.img_volume / norm
        # if self.normalize_masks:
        #     norm = np.iinfo(self.label_volume.dtype).max
        #     self.label_volume = self.label_volume / norm

        dataset_inspect_flag = False
        if dataset_inspect_flag:
            from neuroseg.utils import BatchInspector2D
            BatchInspector2D((self.img_volume, self.label_volume), bboxes=self.bboxes, title="Datagen2D Inspect")

        annotated_pixels = 0
        for b_box in self.bboxes:
            width = b_box[2] - b_box[0]
            height = b_box[3] - b_box[1]
            annotated_pixels_bbox = width * height
            annotated_pixels += annotated_pixels_bbox

        mask_px = np.prod(self.crop_shape)
        self.steps_per_epoch = annotated_pixels // (mask_px * self.batch_size)

    @classmethod
    def get_random_crop(cls,
                        frame_volume: np.ndarray,
                        mask_volume: np.ndarray,
                        crop_shape: Union[list, tuple, np.ndarray],
                        b_boxes: Union[list, tuple, np.ndarray]):

        n_planes = frame_volume.shape[0]
        assert len(b_boxes) == (frame_volume.shape[0])
        plane_weights = np.array([(box[2] - box[0])*(box[3]-box[1]) for box in b_boxes])
        plane_weights = plane_weights / np.sum(plane_weights)
        planes = list(range(n_planes))

        plane = np.random.choice(a=planes, p=plane_weights)
        #plane = np.random.randint(low=0, high=frame_volume.shape[0])

        # outputs x_min, y_min, x_max, y_max
        crop_box = cls._get_bounding_box(frame_shape=frame_volume.shape[1:],
                                         crop_shape=crop_shape,
                                         b_box=b_boxes[plane])
        # but stack ordering is [z,y,x]
        frame_crop = frame_volume[plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        mask_crop = mask_volume[plane, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        return frame_crop, mask_crop

    @staticmethod
    def _get_bounding_box(frame_shape: Union[list, tuple],
                          crop_shape: Union[list, tuple],
                          b_box: np.ndarray = None) -> np.ndarray:
        """get n_crops random bounding boxes in frame_shape"""
        if b_box is None:
            b_box = [0, frame_shape[0], 0, frame_shape[1]]

        if not (np.array(crop_shape) > np.array(frame_shape)[1:2]).any():
            x_low = b_box[0]
            x_high = b_box[2] - crop_shape[0]

            y_low = b_box[1]
            y_high = b_box[3] - crop_shape[1]
        else:
            raise ValueError("crop_shape is larger than frame_shape")
        lower_x = int(np.random.uniform(low=x_low, high=x_high))
        lower_y = int(np.random.uniform(low=y_low, high=y_high))

        upper_x = lower_x + crop_shape[0]
        upper_y = lower_y + crop_shape[1]
        # crop_boxes = (lower_x, lower_y, upper_x, upper_y)
        crop_boxes = np.stack((lower_x, lower_y, upper_x, upper_y), axis=-1)
        # crop_boxes = np.column_stack((lower_x, lower_y, upper_x, upper_y))

        return crop_boxes

    def _single_images_load(self,
                            frame_paths: list,
                            mask_paths: list,
                            csv_paths: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        csv_lists = []
        for fpath in csv_paths:
            if fpath.is_file():
                csv_lists.append(self._parse_csv(fpath))
            else:
                csv_lists.append([0, -1, 0, -1])
        frame_list = [self._load_img(img_to_load_path=fpath,
                                     normalize=self.normalize_inputs,
                                     is_binary_mask=False,
                                     ignore_last_channel=self.ignore_last_channel) for fpath in frame_paths]
        frame_volume = np.stack(frame_list)
        del frame_list
        mask_list = [self._load_img(img_to_load_path=fpath,
                                    is_binary_mask=True,
                                    normalize=self.normalize_masks,
                                    positive_class_value=self.positive_class_value) for fpath in mask_paths]
        mask_volume = np.stack(mask_list)
        del mask_list
        b_boxes = np.array(csv_lists)
        return frame_volume, mask_volume, b_boxes

    @staticmethod
    def _load_img(
            img_to_load_path: str,
            normalize: bool = True,
            ignore_last_channel: bool = False,
            is_binary_mask: bool = False,
            positive_class_value: int = 1,
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
        is_binary_mask : bool, optional
            self-explanatory. The default is False.
        positive_class_value : int, optional
            Value of positive class. The default is 1.

        Returns
        -------
        img : np.array
            numpy array of loaded image.

        """
        try:
            img = skio.imread(img_to_load_path)
            if normalize and (not is_binary_mask):
                norm_constant = np.iinfo(img.dtype).max
                img = img / norm_constant
            if is_binary_mask:
                values = np.unique(img)
                # assert len(values) in [1, 2], "mask is not binary {}\n there are {} values".format(str(
                # img_to_load_path), len(values))
                if len(values) not in [1, 2]:
                    logging.warning(
                        "Mask is not binary {}\nthere are {} values\nautomatically converting to binary mask".format(
                            str(img_to_load_path), len(values)
                        )
                    )
                img = np.where(img == positive_class_value, 1, 0)
                img = img.astype(np.float64)
                # img = np.expand_dims(img, axis=-1)

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

    def generate_train_batch(self) -> dict:

        img_batch = []
        label_batch = []

        for i in range(self.batch_size):
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
