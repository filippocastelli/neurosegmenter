from functools import partial

import tensorflow as tf
from tensorflow.data import Dataset
from skimage import io as skio
import numpy as np
import batchgenerators

SUPPORTED_FORMATS = ["png", "tif", "tiff"]

class dataGen2D:
    def __init__(self,
                 config,
                 partition="train",
                 data_augmentation=True,
                 normalize_inputs=True,
                 ignore_last_channel=False,
                 verbose=False):
        
        self.config = config
        self.partition = partition
        self.data_path_dict = self.config.path_dict[partition]
        
        self.verbose = verbose
        self._path_sanity_check()
        
        self.crop_shape = config.crop_shape
        self.batch_size = config.batch_size
        
        self.normalize_inputs = normalize_inputs
        self.ignore_last_channel = ignore_last_channel
        
        self.buffer_size = config.da_buffer_size
        self.debug_mode = config.da_debug_mode
        self.single_thread = config.da_single_thread
        self.threads = 1 if config.da_single_thread == True else config.da_threads
        
        self.transforms = config.da_transforms
        self.transform_cfg = config.da_transform_cfg
        
        # init sequence
        self._scan_dirs()
        self.data = self.gen_dataset()
        
        
    def _def_transforms(self):
        self.SUPPORTED_TRANSFORMS = {
            "brightness_transform": self._brightness_transform,
            "gamma_transform": self._gamma_transform,
            "gaussian_noise_transform": self._gaussian_noise_transform,
            "mirror_transform": self._mirror_transform,
            "rot90_transform": self._rot90_transform,
            "brightness_multiplicative_transform": self._brightness_multiplicative_transform
            }
        
    def _transform_supported(self, transform):
        return True if transform in self.SUPPORTED_TRANSFORMS else False
    
    def _scan_dirs(self):
        self.frames_paths = self._glob_subdirs("frames")
        self.masks_paths = self._glob_subdirs("masks")
        
    def _glob_subdirs(self, subdir):
        subdir_paths = [str(imgpath) for imgpath in
                        sorted(self.data_path_dict[subdir].glob("*.*"))
                        if self._is_supported_format(imgpath)]
        
        if self.verbose:
            print("there are {} {} imgs".format(len(subdir_paths), subdir))
            
        return subdir_paths
    
    @staticmethod
    def _is_supported_format(fpath):
        extension = fpath.suffix.split(".")[1]
        return extension in SUPPORTED_FORMATS
        
    def _path_sanity_check(self):
        if not (self.data_path_dict["frames"].is_dir()
                and self.data_path_dict["masks"].is_dir()):
            raise ValueError("dataset paths are not actual dirs")
            
    def _get_img_shape(self):
        frame_path = self.frames_paths[0]
        mask_path = self.masks_paths[0]
        frame = self._load_img(frame_path, ignore_last_channel=self.ignore_last_channel)
        mask = self._load_img(mask_path)
        return frame.shape, mask.shape
            
    def gen_dataset(self):
        ds = Dataset.from_tensor_slices((self.frames_paths, self.masks_paths))
        ds = ds.map(map_func=lambda frame_path, mask_path: self._load_example_wrapper(frame_path, mask_path),
                    deterministic=True,
                    num_parallel_calls=self.threads)
        frame_shape, mask_shape = self._get_img_shape()
        ds = ds.map(map_func=lambda frame, mask: self._set_shapes(frame, mask, frame_shape, mask_shape))
        ds = ds.map(map_func=lambda frame, mask: self._random_crop(frame, mask, self.crop_shape))
        ds = ds.unbatch()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.buffer_size)
        return ds
    
    @classmethod
    def _load_example(cls, frame_path, mask_path, normalize_inputs=True, ignore_last_channel=False):
        frame = cls._load_img(frame_path.numpy().decode("utf-8"), normalize_inputs=normalize_inputs, ignore_last_channel=ignore_last_channel)
        mask = cls._load_img(mask_path.numpy().decode("utf-8"), normalize_inputs=normalize_inputs, ignore_last_channel=False)
        return frame, mask
    
    @staticmethod
    def _load_img(img_to_load_path, normalize_inputs=True, ignore_last_channel=False):
        try:
            img = skio.imread(img_to_load_path)
            if normalize_inputs:
                norm_constant = np.iinfo(img.dtype).max
                img = img/norm_constant
            if ignore_last_channel:
                img = img[...,:-1]
            return img
        except ValueError:
            raise ValueError("This image failed: {}, check for anomalies".format(str(img_to_load_path)))
    
    def _load_example_wrapper(self, frame_path, mask_path):
        load_example_partial = partial(self._load_example, normalize_inputs=self.normalize_inputs, ignore_last_channel=self.ignore_last_channel)
        file = tf.py_function(load_example_partial, [frame_path, mask_path], (tf.float64, tf.float64))
        return file
    
    @staticmethod
    def _set_shapes(frame, mask, frame_shape, mask_shape):
        frame.set_shape(frame_shape)
        mask.set_shape(mask_shape)
        return frame, mask
    
    def _random_crop(self, frame, mask, crop_shape):
        
        # counting needed crops
        frame_shape = frame.shape.as_list()[:-1]
        n_pix_frame = np.prod(frame_shape)
        n_pix_crop = np.prod(crop_shape)
        
        n_crops = n_pix_frame // n_pix_crop
        
        #defining crop bounding boxes
        
        lower_x = np.random.uniform(
            low=0, high=1 - (crop_shape[0] / frame_shape[0]), size=(n_crops)
            )
        
        lower_y = np.random.uniform(
            low=0, high=1 - (crop_shape[1] / frame_shape[1]), size=(n_crops)
            )
        
        upper_x = lower_x + crop_shape[0] / frame_shape[0]
        upper_y = lower_y + crop_shape[1] / frame_shape[1]
        
        crop_boxes = np.column_stack((lower_x, lower_y, upper_x, upper_y))
        
        # concatenate frame and mask along channel
        # first expand dims for mask
        mask = tf.expand_dims(mask, axis=-1, name="expand_mask_channel")
        concat = tf.concat([frame, mask], axis=-1)
        
        # adding a batch dim
        concat = tf.expand_dims(concat, axis=0)
        
        # image cropping
        # cropped frames should be [n_crops, crop_height, crop_width, channels]
        
        crops = tf.image.crop_and_resize(
            image=concat,
            boxes=crop_boxes,
            box_indices=np.zeros(n_crops),
            crop_size=crop_shape,
            method="nearest",
            name="crop_stacked"
            )
        
        frame_crop_stack = crops[...,:-1]
        mask_crops_stack = crops[..., -1]
        
        return frame_crop_stack, mask_crops_stack
    
    def _augment0(self, frame, mask,
                 p_rotate=0.5,
                 p_flip=0.5,
                 p_brightness_scale=0.5,
                 p_gaussian_noise=0.5,
                 p_gamma_transform=0.5,
                 brightness_scale_range=0.2,
                 gaussian_noise_std_max=0.06,
                 gamma_range=0.1
                 ):
        #randomizing extractions
        
        extracted = np.random.uniform(low=0, high=1, size=(5))
        
        if extracted[0] < p_rotate:
            self._random_rotate(frame, mask)
        if extracted[1] < p_flip:
            self._random_flip(frame, mask)
        if extracted[2] < p_brightness_scale:
            self._random_brightness_scale(frame, mask, scale_range=brightness_scale_range)
        if extracted[3] < p_gaussian_noise:
            self._random_gaussian_noise(frame, mask, std_max=gaussian_noise_std_max)
        if extracted[4] < p_gamma_transform:
            self._random_gamma_transform(frame, mask, gamma_range=gamma_range)
        

    def _augment(self, frame, mask):
        # transforms = list(transform_cfg.keys())
        transforms = self.transforms[:]
        np.random.shuffle(transforms)
        extracted_probs = np.random.uniform(low=0, high=1, size=(len(transforms)))
        
        for idx, transform in enumerate(transforms):
            if self._transform_supported(transform):
                transform_func = self.SUPPORTED_TRANSFORMS[transform]
                # the dict() constructor creates a shallow copy
                transform_params = dict(self.transform_cfg[transform])
                
                if "p_per_sample" not in transform_params:
                    prob = 1
                else:
                    prob = transform_params["p_per_sample"]
                    transform_params.pop("p_per_sample") 
                    
                if extracted_probs[idx] > prob:
                    frame, mask = transform_func(frame, mask, **transform_params)
        
        
    @staticmethod
    def _rot90_transform(frame, mask):
        rot = tf.random.uniform(shape=[], minval=1, maxval=3, dtype=tf.int32)
        frame = tf.image.rot90(image=frame, k=rot)
        mask = tf.image.rot90(image=mask, k=rot)
        return frame, mask
    
    @staticmethod
    def _mirror_transform(frame, mask):
        flips = np.random.choice(a=[True, False], size=(2))
        if flips[0]:
            frame = tf.image.flip_left_right(frame)
            mask = tf.image.flip_left_right(mask)
        if flips[1]:
            frame = tf.image.flip_up_down(frame)
            mask = tf.image.flip_up_down(mask)
        return frame, mask
        
    @staticmethod
    def _brightness_multiplicative_transform(frame, mask, scale_range=0.2):
        half_range = scale_range / 2
        min_scale = 1.0 - half_range
        max_scale = 1.0 + half_range
        
        scale = tf.random.uniform(shape=[], minval=min_scale, maxval=max_scale, dtype=tf.float32)
        
        frame = frame*scale
        frame = tf.clip_by_value(frame, clip_value_min=0.0, clip_value_max=1.0)
        return frame, mask
        
    @staticmethod
    def _gaussian_noise_transform(frame, mask, std_min=1e-5, std_max=0.06):
        noise_std = tf.random.uniform(
            shape=[], minval=std_min, maxval=std_max, dtype=tf.float32)
        noise = tf.random.normal(
            shape=tf.shape(frame), mean=0.0, stddev=noise_std, dtype=tf.float32)
        
        frame += noise
        frame = tf.clip_by_value(frame, clip_value_min=0.0, clip_value_max=1.0)
        return frame, mask
    
    @staticmethod
    def _gamma_transform(frame, mask, gamma_range=0.2):
        half_range = gamma_range/2
        min_gamma = 1.0 - half_range
        max_gamma = 1.0 + half_range
        gamma = tf.random.uniform(
            shape=[], minval=min_gamma, maxval=max_gamma, dtype=tf.float32)
        frame = tf.pow(frame, gamma)
        frame = tf.clip_by_value(frame, clip_value_min=0.0, clip_value_max=1.0)
        return frame, mask
        
        
    
    
        
        
        
        