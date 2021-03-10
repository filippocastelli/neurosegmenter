from functools import partial
import logging

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.data import Dataset

# from tensorflow.python import debug as tf_debug
# import tensorflow.keras.preprocessing.image as tfk_image

from skimage import io as skio
import numpy as np

# tf.compat.v1.keras.backend.set_session(
#     tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "shelob:6006"))

from datagens.datagenbase import dataGenBase
from utils import load_volume


TFA_INTERPOLATION_MODES = {"nearest": "NEAREST", "bilinear": "BILINEAR"}


class dataGen2D(dataGenBase):
    """inherits from dataGenBase template class"""
    def __init__(
        self,
        config,
        partition="train",
        data_augmentation=True,
        normalize_inputs=True,
        verbose=False
    ):
        """
        dataGen2D data generator
        
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

        Attributes
        ----------
        data: tf.Dataset
            implements the self.data required method, gives access to generated data
            to be used with Keras model.fit()
        iter: iterator
            iterator to the generated data

        """

        super().__init__(
            config=config,
            partition=partition,
            data_augmentation=data_augmentation,
            normalize_inputs=normalize_inputs,
            verbose=verbose,
        )

        self.steps_per_epoch = None
        self.ignore_last_channel = config.ignore_last_channel
        self.buffer_size = config.da_buffer_size
        self.debug_mode = config.da_debug_mode
        
        self.crop_shape = config.crop_shape

        # TODO: Multi_stack support
        # if self.prefetch_volume == True:
        if self.dataset_mode == "stack":
            self.frames_volume, self.masks_volume = self._load_single_stack()

        self.data = self.gen_dataset()
        self.iter = self.data.__iter__

    @classmethod
    def _get_transform(cls, transform):
        SUPPORTED_TRANSFORMS = {
            "brightness_transform": cls._brightness_transform,
            "gamma_transform": cls._gamma_transform,
            "gaussian_noise_transform": cls._gaussian_noise_transform,
            "mirror_transform": cls._mirror_transform,
            "rot90_transform": cls._rot90_transform,
            "brightness_multiplicative_transform": cls._brightness_multiplicative_transform,
            "spatial_transform": cls._spatial_transform,
        }
        return (
            SUPPORTED_TRANSFORMS[transform]
            if transform in SUPPORTED_TRANSFORMS
            else None
        )

    def _get_img_shape(self):
        
        if self.dataset_mode == "single_images":
            frame_path = self.frames_paths[0]
            mask_path = self.masks_paths[0]
            frame = self._load_img(frame_path, ignore_last_channel=self.ignore_last_channel)
            mask = self._load_img(mask_path, is_binary_mask=True)
            
            frame_shape = frame.shape
            mask_shape = mask.shape
            
        elif self.dataset_mode == "stack":
            frame_shape = self.frames_volume[0].shape
            mask_shape = self.masks_volume[0].shape
        else:
            raise NotImplementedError(self.dataset_mode)
            
        return frame_shape, mask_shape

    def _load_single_stack(self):
        assert (len(self.frames_paths) == 1) and (len(self.masks_paths) == 1), "More than 1 stack found"
        frame = load_volume(self.frames_paths[0],
                            drop_last_dim=self.ignore_last_channel,
                            expand_last_dim=True,
                            data_mode="stack")
        
        mask = load_volume(self.masks_paths[0],
                           drop_last_dim=False,
                           expand_last_dim=True,
                           data_mode="stack")
        
        mask = np.where(mask==self.positive_class_value, 1,0)
        
        if self.normalize_inputs:
            norm_constant_frame = np.iinfo(frame.dtype).max
            frame = frame / norm_constant_frame
            
            norm_constant_mask = np.iinfo(mask.dtype).max
            mask = mask / norm_constant_mask
            
        return frame,  mask
        
    # def _load_volumes(self):
    #     example_path_list = zip(self.frames_paths, self.masks_paths)
    #     frame_list = []
    #     mask_list = []
    #     for example_path in example_path_list:
    #         frame, mask = self._load_example(
    #             example_path[0],
    #             example_path[1],
    #             self.normalize_inputs,
    #             self.ignore_last_channel,
    #             self.positive_class_value,
    #         )
    #         frame_list.append(frame)
    #         mask_list.append(mask)

    #     self.frames_volume = np.array(frame_list)
    #     self.masks_volume = np.array(mask_list)

    @staticmethod
    def _load_img(
        img_to_load_path,
        normalize_inputs=True,
        ignore_last_channel=False,
        is_binary_mask=False,
        positive_class_value=1,
    ):
        """
        load an image from path, apply basic transforms

        Parameters
        ----------
        img_to_load_path : str
            filepath to input image.
        normalize_inputs : bool, optional
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
            if normalize_inputs and (not is_binary_mask):
                norm_constant = np.iinfo(img.dtype).max
                img = img / norm_constant
            if is_binary_mask:
                values = np.unique(img)
                # assert len(values) in [1, 2], "mask is not binary {}\n there are {} values".format(str(img_to_load_path), len(values))
                if len(values) not in [1, 2]:
                    logging.warning(
                        "Mask is not binary {}\nthere are {} values\nautomatically converting to binary mask".format(
                            str(img_to_load_path), len(values)
                        )
                    )
                img = np.where(img == positive_class_value, 1, 0)
                img = img.astype(np.float64)
                img = np.expand_dims(img, axis=-1)
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
    def _load_example(
        cls,
        frame_path,
        mask_path,
        normalize_inputs=True,
        ignore_last_channel=False,
        positive_class_value=1,
    ):
        """
        generate a (frame, mask) data point given frame_path and mask_path

        Parameters
        ----------
        frame_path, mask_path : str
            frame/mask paths.
        normalize_inputs : bool, optional
            if True divides image values by np.iinfo(img.dtype).max. The default is True.
        ignore_last_channel : bool, optional
            If True doesn't consider the last channel of an image
            (RGB 2-channel images with empty third channel) . The default is False.
        positive_class_value : int, optional
            Value of positive class. The default is 1.

        Returns
        -------
        (frame, mask): tuple
            numpy volumes of loaded data

        """

        if type(frame_path) is not str:
            frame_path_str = frame_path.numpy().decode("utf-8")
            mask_path_str = mask_path.numpy().decode("utf-8")
        else:
            frame_path_str = frame_path
            mask_path_str = mask_path

        frame = cls._load_img(
            frame_path_str,
            normalize_inputs=normalize_inputs,
            ignore_last_channel=ignore_last_channel,
        )
        mask = cls._load_img(
            mask_path_str,
            normalize_inputs=normalize_inputs,
            ignore_last_channel=False,
            is_binary_mask=True,
            positive_class_value=positive_class_value,
        )
        return frame, mask

    def _load_example_wrapper(self, frame_path, mask_path):
        """wrapper for _load_example function"""
        load_example_partial = partial(
            self._load_example,
            normalize_inputs=self.normalize_inputs,
            ignore_last_channel=self.ignore_last_channel,
            positive_class_value=self.positive_class_value,
        )
        file = tf.py_function(
            load_example_partial, [frame_path, mask_path], (tf.float64, tf.float64)
        )
        return file

    def gen_dataset(self):
        """generate a 2D pipeline tf.Dataset"""

        if self.dataset_mode == "stack":
            ds = Dataset.from_tensor_slices((self.frames_volume, self.masks_volume))
        else:
            ds = Dataset.from_tensor_slices((self.frames_paths, self.masks_paths))
            ds = ds.map(
                map_func=lambda frame_path, mask_path: self._load_example_wrapper(
                    frame_path, mask_path
                ),
                deterministic=True,
                num_parallel_calls=self.threads,
            )
        full_frame_shape, full_mask_shape = self._get_img_shape()
        # full_frame_shape = [768, 1024]
        # full_mask_shape = [768, 1024]
        ds = ds.map(
            map_func=lambda frame, mask: self._set_shapes(
                frame, mask, full_frame_shape, full_mask_shape
            )
        )
        ds = ds.map(
            map_func=lambda frame, mask: self._random_crop(
                frame, mask, crop_shape=self.crop_shape, batch_crops=True
            )
        )
        ds = ds.unbatch()
        if self.data_augmentation:
            if self.transform_cfg is None:
                raise ValueError(
                    "There are no configured data augmentation transforms in configuration YAML"
                )

            augment_partial = partial(self._augment, transform_cfg=self.transform_cfg)
            ds = ds.map(
                map_func=lambda frame, mask: tf.py_function(
                    func=augment_partial,
                    inp=[frame, mask],
                    Tout=[tf.float32, tf.float32],
                ),
                num_parallel_calls=self.threads,
            )

        frame_channels = full_frame_shape[-1]
        frame_shape = np.append(self.crop_shape, frame_channels)
        mask_shape = np.append(self.crop_shape, 1)
        ds = ds.map(
            map_func=lambda frame, mask: self._set_shapes(
                frame, mask, frame_shape, mask_shape
            )
        )

        ds = ds.batch(self.batch_size)
        # ds = ds.prefetch(self.buffer_size)
        return ds

    @staticmethod
    def _set_shapes(frame, mask, frame_shape, mask_shape):
        """apply tf.Tensor shapes to frame, mask tuple"""
        frame.set_shape(frame_shape)
        mask.set_shape(mask_shape)
        return frame, mask

    @classmethod
    def _random_crop(
        cls, frame, mask, crop_shape, batch_crops=True, keep_original_size=False
    ):
        """
        Get batches of random crops from (frame, mask) tensors
        
        Parameters
        ----------
        frame, mask : tf.Tensor 
            Input tensors.
        crop_shape : tuple
            shape of desired crop.
        batch_crops : bool, optional
            If True, calculates the number of random crops based on volumes of 
            frames and crop_shape, and returns a (n_crops, *crop_shape) wide batch.
            If False returns a (1, *crop_shape) batch.
            The default is True.
        keep_original_size : bool, optional
            If True, the output tensor shape will have the same shape as the frame Tensor.
            The default is False.

        Returns
        -------
        frame_crop_stack, mask_crop_shack: tf.Tensor
            (n_crops, *crop_shape) tensor batch

        """

        if len(frame.shape) < 4:
            frame_shape = frame.shape.as_list()[:-1]
        else:
            frame_shape = frame.shape.as_list()[1:-1]

        # print("frame.shape: {} \n mask.shape: {}".format(frame.shape, mask.shape))
        if batch_crops:
            # counting needed crops
            n_pix_frame = np.prod(frame_shape)
            n_pix_crop = np.prod(crop_shape)
            n_crops = n_pix_frame // n_pix_crop

            crop_boxes = cls._get_bound_boxes(frame_shape, crop_shape, n_crops)
        else:
            n_crops = 1
            crop_boxes = cls._get_bound_boxes(frame_shape, crop_shape, n_crops)

        # defining crop bounding boxes

        # concatenate frame and mask along channel
        # first expand dims for mask
        concat = tf.concat([frame, mask], axis=-1)

        # adding a batch dim if not already batched
        if len(concat.shape) < 4:
            concat = tf.expand_dims(concat, axis=0)

        # image cropping
        # cropped frames should be [n_crops, crop_height, crop_width, channels]
        crop_size = frame_shape if keep_original_size else crop_shape

        crops = tf.image.crop_and_resize(
            image=concat,
            boxes=crop_boxes,
            box_indices=np.zeros(n_crops),
            crop_size=crop_size,
            method="nearest",
            name="crop_stacked",
        )

        frame_crop_stack = crops[..., :-1]
        mask_crop_stack = crops[..., -1]

        mask_crop_stack = tf.expand_dims(mask_crop_stack, axis=-1)
        return frame_crop_stack, mask_crop_stack

    @staticmethod
    def _get_bound_boxes(frame_shape, crop_shape, n_crops):
        """get n_crops random bounding boxes in frame_shape"""
        if not (np.array(crop_shape) > np.array(frame_shape)).any():
            x_low = 0
            x_high = 1 - (crop_shape[0] / frame_shape[0])

            y_low = 0
            y_high = 1 - (crop_shape[1] / frame_shape[1])
        else:
            x_low = -(crop_shape[0] / (2 * frame_shape[0]))
            x_high = 1 - (crop_shape[0] / (2 * frame_shape[0]))

            y_low = -(crop_shape[1] / (2 * frame_shape[1]))
            y_high = 1 - (crop_shape[1] / (2 * frame_shape[1]))

        lower_x = np.random.uniform(low=x_low, high=x_high, size=(n_crops))
        lower_y = np.random.uniform(low=y_low, high=y_high, size=(n_crops))

        upper_x = lower_x + crop_shape[0] / frame_shape[0]
        upper_y = lower_y + crop_shape[1] / frame_shape[1]
        crop_boxes = np.column_stack((lower_x, lower_y, upper_x, upper_y))

        return crop_boxes

# OLD CODE FOR DATA AUGMENTATION

    # def _augment0(
    #     self,
    #     frame,
    #     mask,
    #     p_rotate=0.5,
    #     p_flip=0.5,
    #     p_brightness_scale=0.5,
    #     p_gaussian_noise=0.5,
    #     p_gamma_transform=0.5,
    #     brightness_scale_range=0.2,
    #     gaussian_noise_std_max=0.06,
    #     gamma_range=0.1,
    # ):
    #     # randomizing extractions

    #     extracted = np.random.uniform(low=0, high=1, size=(5))

    #     if extracted[0] < p_rotate:
    #         self._random_rotate(frame, mask)
    #     if extracted[1] < p_flip:
    #         self._random_flip(frame, mask)
    #     if extracted[2] < p_brightness_scale:
    #         self._random_brightness_scale(
    #             frame, mask, scale_range=brightness_scale_range
    #         )
    #     if extracted[3] < p_gaussian_noise:
    #         self._random_gaussian_noise(frame, mask, std_max=gaussian_noise_std_max)
    #     if extracted[4] < p_gamma_transform:
    #         self._random_gamma_transform(frame, mask, gamma_range=gamma_range)

    @classmethod
    def _augment(cls, frame, mask, transform_cfg):
        """data augmentation pipeline"""
        # transforms = list(transform_cfg.keys())
        # pudb.set_trace()
        transforms = list(transform_cfg.keys())
        # transforms = self.transforms[:]
        np.random.shuffle(transforms)
        extracted_probs = np.random.uniform(low=0, high=1, size=(len(transforms)))

        for idx, transform in enumerate(transforms):
            transform_func = cls._get_transform(transform)
            if transform_func is not None:
                transform_func = cls._get_transform(transform)
                # the dict() constructor creates a shallow copy
                transform_params = dict(transform_cfg[transform])

                if "p_per_sample" not in transform_params:
                    prob = 1
                else:
                    prob = transform_params["p_per_sample"]
                    transform_params.pop("p_per_sample")

                if "per_channel" in transform_params:
                    transform_params.pop("per_channel")

                if extracted_probs[idx] > prob:
                    frame, mask = transform_func(frame, mask, **transform_params)
            else:
                raise NotImplementedError(transform)

        return frame, mask

    @staticmethod
    def _rot90_transform(frame, mask, num_rot=None):
        """discrete 90 degree transform"""
        # num_rot is not used
        # maintained for consistency with batchgenerators syntax
        rot = tf.random.uniform(shape=[], minval=1, maxval=3, dtype=tf.int32)
        rot_frame = tf.image.rot90(image=frame, k=rot)
        rot_mask = tf.image.rot90(image=mask, k=rot)

        return rot_frame, rot_mask

    @staticmethod
    def _mirror_transform(frame, mask, axes=(0, 1)):
        """mirror transform"""
        flips = np.random.choice(a=[True, False], size=(2))
        if flips[0] and (0 in axes):
            frame = tf.image.flip_left_right(frame)
            mask = tf.image.flip_left_right(mask)
        if flips[1] and (1 in axes):
            frame = tf.image.flip_up_down(frame)
            mask = tf.image.flip_up_down(mask)
        return frame, mask

    @staticmethod
    def _brightness_transform(frame, mask, mu=0, sigma=0.1):
        """brightness additive transform"""
        brightness_shift = tf.random.normal(
            shape=[], mean=mu, stddev=sigma, dtype=tf.float32
        )
        frame += brightness_shift
        frame = tf.clip_by_value(frame, clip_value_min=0.0, clip_value_max=1.0)
        return frame, mask

    @staticmethod
    def _brightness_multiplicative_transform(frame, mask, multiplier_range=[0.8, 1.2]):
        """brightness multiplicative transform"""
        scale = tf.random.uniform(
            shape=[],
            minval=multiplier_range[0],
            maxval=multiplier_range[1],
            dtype=tf.float32,
        )

        frame = frame * scale
        frame = tf.clip_by_value(frame, clip_value_min=0.0, clip_value_max=1.0)
        return frame, mask

    @staticmethod
    def _gaussian_noise_transform(frame, mask, noise_variance=[0, 0.05]):
        """additive gaussian noise transform"""
        # noise_variance refers to the range of noise variance
        # maintained for consistency with batchgenerators syntax

        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]

        variance = tf.random.uniform(
            shape=[],
            minval=noise_variance[0],
            maxval=noise_variance[1],
            dtype=tf.float32,
        )
        noise = tf.random.normal(
            shape=tf.shape(frame), mean=0.0, stddev=np.sqrt(variance), dtype=tf.float32
        )

        frame += noise
        frame = tf.clip_by_value(frame, clip_value_min=0.0, clip_value_max=1.0)
        return frame, mask

    @staticmethod
    def _gamma_transform(frame, mask, gamma_range=[0.8, 1.2], invert_image=False):
        """gamma power transform"""
        # invert_image is not used
        # maintained for consistency with batchtgenerators syntax

        gamma = tf.random.uniform(
            shape=[], minval=gamma_range[0], maxval=gamma_range[1], dtype=tf.float32
        )
        frame = tf.pow(frame, gamma)
        frame = tf.clip_by_value(frame, clip_value_min=0.0, clip_value_max=1.0)
        return frame, mask

    @classmethod
    def _spatial_transform(
        cls,
        frame,
        mask,
        do_elastic_deform=False,
        deformation_scale=[0.25, 0.25],
        do_rotation=True,
        p_rot_per_sample=0.15,
        angle=[-0.314, 0.314],
        do_scale=True,
        scale=[0.95, 1.05],
        p_scale_per_sample=0.15,
        border_mode_data="nearest",
        random_crop=False,
    ):
        """generalized spatial transform"""

        if border_mode_data not in TFA_INTERPOLATION_MODES:
            raise NotImplementedError(border_mode_data)

        if do_rotation and np.random.uniform(0, 1) > p_rot_per_sample:
            rot_angle = np.random.uniform(low=angle[0], high=angle[1])
            frame = tfa.image.rotate(
                frame,
                rot_angle,
                interpolation=TFA_INTERPOLATION_MODES[border_mode_data],
            )
            mask = tfa.image.rotate(mask, rot_angle, interpolation="NEAREST")

        if do_scale and np.random.uniform(0, 1) > p_scale_per_sample:
            # determine crop shape
            crop_shape = cls._get_scaled_cropshape(frame.shape[1:-1], scale_range=scale)
            frame, mask = cls._random_crop(
                frame, mask, crop_shape, batch_crops=False, keep_original_size=True
            )
            # tfk_image.random_zoom(frame, zoom_range=scale,
            #                       row_axis=0, col_axis=1, channel_axis=2,
            #                       fill_mode=border_mode_data,
            #                       interpolation_order=1)
        if random_crop:
            pass
            # random cropping not implemented yet

        return frame, mask

    @staticmethod
    def _get_scaled_cropshape(frame_shape_trimmed, scale_range):
        """apply pre-scaling to a crop shape"""
        # frame_shape is a TensorShape
        crop_scale = tf.random.uniform(
            shape=[], minval=scale_range[0], maxval=scale_range[1], dtype=tf.float32
        )

        frame_shape_array = np.array(frame_shape_trimmed.as_list())
        crop_shape = np.array(frame_shape_array * crop_scale)

        return crop_shape

    # @staticmethod
    # def _clipped_zoom(img, zoom_factor, **kwargs):
    #     # shamelessly copypasted from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions/48097478

    #     # retrieving spatial dims
    #     height, width = img.shape[:2]

    #     # don't apply zoom to the channel dim
    #     zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    #     return
