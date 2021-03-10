import logging
import gc

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
    GammaTransform)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform)

import numpy as np

from datagens.datagenbase import dataGenBase
from utils import load_volume


class datagen3DSingle(dataGenBase):
    """3D data pipeline, inherits from dataGenBase
    based on BatchGenerators"""
    def __init__(self,
                 config,
                 partition="train",
                 data_augmentation=True,
                 normalize_inputs=True,
                 verbose=False):
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
            
        Attributes
        ----------
        data: generator
            implements the self.data required method, gives access to generated data
            to be used with Keras model.fit()
        iter: iterator
            iterator to the generated data
        steps_per_epoch: int
            defines the number of steps to complete an epoch
            to be used in model.fit()
        """
        
        super().__init__(config=config,
                         partition=partition,
                         data_augmentation=data_augmentation,
                         normalize_inputs=normalize_inputs,
                         verbose=verbose)

        self._parse_single_stack_paths()
        self._load_volumes()
        
        self.shuffle = self.config.da_shuffle
        self.seed = self.config.da_seed
        # self.pre_crop_scales = self.config.da_pre_crop_scales
        
        self.data = self._setup_gen()
        self.iter = self.data.__iter__
        self.steps_per_epoch = self._get_steps_per_epoch(self.volume_shape, self.crop_shape, self.batch_size)

    def _parse_single_stack_paths(self):
        self.frames_path = self.frames_paths[0]
        self.masks_path = self.masks_paths[0]
    
    @staticmethod
    def _normalize_stack(stack_arr, norm=255):
        """divide stack by normalization constant"""
        return (stack_arr/norm).astype(np.float32)
    
    def _load_volumes(self):
        """load images from file,
        apply normalizations,
        fix stack dimensions,,
        create {"img: img_volume, "label": label_volume} data dict"""
        # for processing purposes we use "stack" instead of "multi_stack"
        data_mode = self.dataset_mode if self.dataset_mode != "multi_stack" else "stack"
        
        #TODO: check if need drop_last_dim or expand_last_dim
        self.frames_volume = load_volume(str(self.frames_path),
                                         data_mode=data_mode,
                                         drop_last_dim=False)
        self.masks_volume = load_volume(str(self.masks_path),
                                        data_mode=data_mode,
                                        drop_last_dim=False)
        
        # self.frames_volume = tifffile.imread(str(self.frames_path)).astype(np.float32)
        # self.masks_volume = tifffile.imread(str(self.masks_path)).astype(np.float32)
        
        logging.debug("reading from disk, dataset mode: {}...".format(self.dataset_mode))
        
        if self.normalize_inputs:
            #TODO check normalization constant
            self.frames_volume = self._normalize_stack(self.frames_volume,
                                                       norm=np.iinfo(self.frames_volume.dtype).max)
            self.masks_volume = self._normalize_stack(self.masks_volume,
                                                      norm=np.iinfo(self.masks_volume.dtype).max)
            
        self._adjust_stack_dims()
        
        self.data_dict = {
            "img" : self.frames_volume,
            "label": self.masks_volume }
        # self.frames_volume and self.masks_volume are redundant at this point
        # they might need to be deleted to free space
        
        del self.frames_volume
        del self.masks_volume
        gc.collect()
            
    
    def _adjust_stack_dims(self):
        """uniform stack dimensions
        target dimensions are [z, y, x, ch]"""
        frames_shape = self.frames_volume.shape
        masks_shape = self.masks_volume.shape
        
        # transform to [z, y, x, ch]
        
        if len(np.unique(self.frames_volume[...,2])) == 1:
            # if samples are two-colors, third rgb dim should not be included
            self.frames_volume = self.frames_volume[...,-1]
        
        if len(frames_shape) == 4 and len(masks_shape) == 4:
            # this should be ok except when labels are multichannel
            # check if labels are monochrome
            
            if masks_shape[-1] != 1:
                raise Exception(
                    "Labels must be monochrome for single-class semantic segmentation")
                
        elif len(frames_shape) == 4 and len(masks_shape) == 4:
            # frames are multichannel but labels are monochrome
            # expand dimensions for labels
            self.masks_volume = np.expand_dims(self.masks_volume, axis=-1)
        
        elif len(frames_shape) == 3 and len(masks_shape) == 3:
            # both stacks are monochrome
            # expand dimensions_for both
            
            self.frames_volume = np.expand_dims(self.frames_volume, axis=-1)
            self.masks_volume = np.expand_dims(self.masks_volume, axis=-1)
        
        self.volume_shape = self.frames_volume.shape[:-1]
        
        # convert to [ch, z, y, x]
        self.frames_volume = np.moveaxis(self.frames_volume, -1, 0)
        self.masks_volume = np.moveaxis(self.masks_volume, -1, 0)
        
    def _spatial_transform_cfg(self):
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
    
    def _brightness_multiplicative_transform_cfg(self):
        """brightness multiplicative transform configurator"""
        brightness_multiplicative_transform_cfg = {
            "multiplier_range": (0.9, 1.1),
            "per_channel" : True,
            "p_per_sample" : 0.15}
        
        brightness_multiplicative_transform_cfg.update(self.transform_cfg["brightness_multiplicative_transform"])    
        return brightness_multiplicative_transform_cfg
    
    def _brightness_transform_cfg(self):
        """brightness additive transform configurator"""
        brightness_transform_cfg = {
            "p_per_sample" : 0.15,
            "mu" : 0,
            "sigma" : 0.01,
            }
        
        brightness_transform_cfg.update(self.transform_cfg["brightness_transform"])
        return brightness_transform_cfg
    
    def _rot90_transform_cfg(self):
        """rigid 90 degree rotation transform configurator"""
        rot90_transform_cfg = {
            "num_rot" : (1,2,3),
            "axes": (0,1,2),
            "p_per_sample" : 0.15
            }
        
        rot90_transform_cfg.update(self.transform_cfg["rot90_transform"])
        return rot90_transform_cfg
    
    def _mirror_transform_cfg(self):
        """mirror transform configurator"""
        mirror_transform_cfg = {
            "axes" : (0, 1, 2)
            }
        
        mirror_transform_cfg.update(self.transform_cfg["mirror_transform"])
        return mirror_transform_cfg
        
    def _gamma_transform_cfg(self):
        """gamma trasnform configurator"""
        gamma_transform_cfg = {
            "p_per_sample" : 0.15,
            "per_channel" : True,
            "invert_image" : False,
            "retain_stats": True,
            "gamma_range": (0.9, 1.1)
            }
        
        gamma_transform_cfg.update(self.transform_cfg["gamma_transform"])
        return gamma_transform_cfg
    
    def _gaussian_noise_transform_cfg(self):
        """gaussian noise additive transform configurator"""
        gaussian_noise_transform_cfg = {
            "p_per_sample": 0.15,
            "noise_variance" : [0, 0.0005]
            }
        
        gaussian_noise_transform_cfg.update(self.transform_cfg["gaussian_noise_transform"])
        return gaussian_noise_transform_cfg
    
    def _gaussian_blur_transform_cfg(self):
        """gaussian filtering transform configurator"""
        gaussian_blur_transform_cfg = {
            "p_per_sample": 0.15,
            "blur_sigma": (1,5),
            "different_sigma_per_channel": True,
            "p_per_channel": True
            }
        
        gaussian_blur_transform_cfg.update(self.transform_cfg["gaussian_blur_transform"])
        return gaussian_blur_transform_cfg
    
    @classmethod
    def _get_transform_fn(cls, transform):
        """convert transform string to transform function"""
        TRANSFORM_FNS = {
            "brightness_transform" : BrightnessTransform,
            "brightness_multiplicative_transform" : BrightnessMultiplicativeTransform,
            "gamma_transform" : GammaTransform,
            "gaussian_noise_transform" : GaussianNoiseTransform,
            "gaussian_blur_transform" : GaussianBlurTransform,
            "mirror_transform" : MirrorTransform,
            "rot90_transform" : Rot90Transform,
            "spatial_transform" : SpatialTransform_2
            }
        
        if transform not in TRANSFORM_FNS:
            raise NotImplementedError("transform {} not supported".format(transform))
            
        return TRANSFORM_FNS[transform] if transform in TRANSFORM_FNS else None
    
    def _get_transform_cfg(self, transform):
        """convert transform string to transform configurator function"""
        TRANSFORM_CFGS = {
            "brightness_transform" : self._brightness_transform_cfg(),
            "brightness_multiplicative_transform" : self._brightness_multiplicative_transform_cfg(),
            "gamma_transform" : self._gamma_transform_cfg(), 
            "gaussian_noise_transform" : self._gaussian_noise_transform_cfg(),
            "gaussian_blur_transform" : self._gaussian_blur_transform_cfg(),
            "mirror_transform" : self._mirror_transform_cfg(),
            "rot90_transform" : self._rot90_transform_cfg(),
            "spatial_transform" : self._spatial_transform_cfg()
            }
        if transform not in TRANSFORM_CFGS:
            raise NotImplementedError("transform {} not supported".format(transform))
            
        return TRANSFORM_CFGS[transform] if transform in TRANSFORM_CFGS else None
        
    def _get_augment_transforms(self):
        """returns a list of pre-configured transform functions
        basically generates a pipeline of sequential transforms"""
        transforms = []
        for transform in self.transforms:
            transform_fn = self._get_transform_fn(transform)
            transform_cfg = self._get_transform_cfg(transform)
            
            transforms.append(transform_fn(**transform_cfg))
        
        return transforms
    
    def _get_transform_chain(self):
        """generate a Compose() global transform"""
        if self.data_augmentation:
            transforms = self._get_augment_transforms()
        else:
            transforms = [RandomCropTransform(crop_size=self.crop_shape)]
        
        return Compose(transforms)
    
    def _setup_gen(self):
        """returns a python generator to be used with keras
        supports both SingleThread and MultiThread BatchGenerators augmenters"""
        self.dataLoader = CroppedDataLoaderBG(
            data=self.data_dict,
            batch_size=self.batch_size,
            crop_shape=self.crop_shape,
            num_threads_in_multithreaded=self.threads,
            shuffle=self.shuffle,
            seed_for_shuffle=self.seed,
            infinite=False,
            # pre_crop_scales=self.pre_crop_scales
            )
        self.composed_transform = self._get_transform_chain()
        
        if self.single_thread:
            self.gen = SingleThreadedAugmenter(self.dataLoader, self.composed_transform)
        else:
            self.gen = MultiThreadedAugmenter(self.dataLoader,
                                         self.composed_transform,
                                         self.threads)
            
        return self._get_keras_gen(self.gen, channel_last=True)
        
    @staticmethod
    def _get_steps_per_epoch(frame_shape, crop_shape, batch_size):
        """calculate the number of steps per epoch"""
        frame_px = np.prod(frame_shape)
        crop_px = np.prod(crop_shape)
        return int( np.ceil( (frame_px / crop_px) / float(batch_size)))
    
    @classmethod
    def _get_keras_gen(cls, batchgen, channel_last=True):
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
    def _to_channel_last(input_tensor):
        """convert (batch, ch, z, y, x) tensor to (batch, z, y, x, ch) tensor"""
        return np.moveaxis(input_tensor, source=1, destination=-1)


class CroppedDataLoaderBG(DataLoader):
    """expands DataLoader class
    used for loading pre-applied random cropping to data"""
    
    def __init__(
            self,
            data,
            batch_size,
            crop_shape,
            num_threads_in_multithreaded=1,
            shuffle=False,
            seed_for_shuffle=123,
            infinite=False):
        """
        CroppedDataLoaderBG
        
        loads cropped data from npy volumes

        Parameters
        ----------
        data : dict
            {"img": img_npy, "label": label_npy} dict.
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
        
        self.volume_shape = data["img"].shape[1:]
        self.frames_volume = data["img"]
        self.masks_volume = data["label"]
        
        self.crop_shape = crop_shape
        # self.pre_crop_scales = pre_crop_scales
        self.rng_seed = seed_for_shuffle
        
        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)
        
        
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
    
    def generate_train_batch(self):
        """return a {"data": crooped_data, "seg": crooped_seg} dict"""
        
        img_batch = []
        label_batch = []
        
        for i in range(self.batch_size):
            crop_img, crop_label = self._get_random_crop()
            img_batch.append(crop_img)
            label_batch.append(crop_label)
        
        stack_img = np.stack(img_batch, axis=0)
        stack_label = np.stack(label_batch, axis=0)
        
        return {"data": stack_img, "seg": stack_label}
     
    def _get_random_crop(self):
        """get a random crop_img, crop_label tuple"""
        z_shape, y_shape, x_shape = self.volume_shape
        
        z_0 = np.random.randint(low=0, high=z_shape - self.crop_shape[0])
        y_0 = np.random.randint(low=0, high=y_shape - self.crop_shape[1])
        x_0 = np.random.randint(low=0, high=x_shape - self.crop_shape[2])
        
        crop_img = self.frames_volume[
            :,
            z_0 : z_0 + self.crop_shape[0],
            y_0 : y_0 + self.crop_shape[1],
            x_0 : x_0 + self.crop_shape[2]
            ]
        
        crop_label = self.masks_volume[
            :,
            z_0 : z_0 + self.crop_shape[0],
            y_0 : y_0 + self.crop_shape[1],
            x_0 : x_0 + self.crop_shape[2]
            ]
        
        return crop_img, crop_label
        
        
        