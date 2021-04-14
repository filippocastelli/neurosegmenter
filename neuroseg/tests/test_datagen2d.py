import unittest
from pathlib import Path
import os, sys, inspect
import pudb
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import numpy.testing as nptest
import pudb

import tensorflow as tf
import tensorflow.debugging as tfdebug

from datagens.datagen2d import dataGen2D
from utils import BatchInspector2D
from config import TrainConfig

class Datagen2DTest(unittest.TestCase):
    
    def setUp(self):
        self.currentdir = Path(currentdir)
        self.config_path = self.currentdir.joinpath("test_train_cfg.yml")
        self.config = TrainConfig(self.config_path)

        self.dataset_path = self.config.dataset_path
        
        
        self.train_paths = self.config.train_paths
        self.val_paths = self.config.val_paths
        self.test_paths = self.config.test_paths
        
        self.test_frames_dir_path = self.test_paths["frames"]
        self.test_masks_dir_path = self.test_paths["masks"]
        
        self.test_frames_paths = list(sorted(self.test_frames_dir_path.glob("*.png")))
        self.test_masks_paths = list(sorted(self.test_masks_dir_path.glob("*.png")))
        
        self.tmp_path = self.config.temp_path
        self.logs_path = self.config.logs_path
        
        self.transform_cfg = self.config.da_transform_cfg
        
    def tearDown(self):
        self._rm_tree(self.tmp_path)
        
    @classmethod
    def _rm_tree(cls, path):
        path = Path(path)
        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                cls._rm_tree(child)
        path.rmdir()
        
        
    def _setup_datagen_noaugment(self):
        # pudb.set_trace()
        datagen = dataGen2D(self.config, data_augmentation=False,
                            ignore_last_channel=True)
        return datagen
    
    def _setup_datagen_augment(self):
        datagen = dataGen2D(self.config, data_augmentation=True,
                            ignore_last_channel=True)
        return datagen
    
    # def test_setup_datagen_no_augmentation(self):
    #     self.datagen_noaugment = self._setup_datagen_noaugment()
    #     data_iterator = self.datagen_noaugment.data.as_numpy_iterator()
    #     ex_list = list(data_iterator)
        
    # def test_setup_datagen_augment(self):
    #     pudb.set_trace()
    #     self.datagen_augment = self._setup_datagen_augment()
    #     data_iterator = self.datagen_augment.data.as_numpy_iterator()
    #     ex_list = list(data_iterator)
        
    def test_load_img(self):
        frame, mask = self._load_frame_mask()
        nptest.assert_equal(frame.shape, (512, 512, 2),str(frame.shape))
        nptest.assert_equal(mask.shape, (512, 512, 1), str(mask.shape))
        nptest.assert_equal(np.unique(mask), (0,1))
    
    def test_random_crop(self):
        self._load_crop_stack()
        crop_shape = (64,64)
        frame_crop_stack, mask_crop_stack = self._load_crop_stack(crop_shape)
        frame_stack_shape = frame_crop_stack.shape.as_list()
        mask_stack_shape = mask_crop_stack.shape.as_list()
        
        # pudb.set_trace()
        # BatchInspector2D([frame_crop_stack, mask_crop_stack])
        
        nptest.assert_equal((64, 64, 64, 2), frame_stack_shape)
        nptest.assert_equal((64, 64, 64, 1), mask_stack_shape)
    
    def test_rot90_transform(self):
        crop_shape=(64,64)
        frame_crop, mask_crop = self._get_single_crop(crop_shape)
        rot_frame_crop, rot_mask_crop = dataGen2D._rot90_transform(frame_crop, mask_crop)
        self._assertshapes(rot_frame_crop, rot_mask_crop,
                           frame_shape=(1,64,64,2),
                           mask_shape=(1,64,64,1))
        
    def test_mirror_transform(self):
        crop_shape=(64,64)
        frame, mask = self._get_single_crop(crop_shape)
        frame, mask = dataGen2D._mirror_transform(frame, mask)
        self._assertshapes(frame, mask)
    
    def test_brightness_transform(self):
        crop_shape=(64,64)
        frame, mask = self._get_single_crop(crop_shape)
        frame, mask = dataGen2D._brightness_transform(frame, mask)
        self._assertshapes(frame, mask)
    
    def test_brightness_multiplicative_transform(self):
        crop_shape=(64,64)
        frame, mask = self._get_single_crop(crop_shape)
        frame, mask = dataGen2D._brightness_multiplicative_transform(frame, mask)
        self._assertshapes(frame, mask)
        
    def test_gaussian_noise_transform(self):
        crop_shape=(64,64)
        frame, mask = self._get_single_crop(crop_shape)
        frame, mask = dataGen2D._gaussian_noise_transform(frame, mask)
        self._assertshapes(frame, mask)
    
    def test_gamma_transform(self):
        crop_shape=(64,64)
        frame, mask = self._get_single_crop(crop_shape)
        frame, mask = dataGen2D._gamma_transform(frame, mask)
        self._assertshapes(frame, mask)
        
    def test_spatial_transform(self):
        crop_shape=(64,64)
        frame, mask = self._get_single_crop(crop_shape)
        # pudb.set_trace()
        frame, mask = dataGen2D._spatial_transform(frame, mask)
        self._assertshapes(frame, mask)
        
    def test_augment(self):
        crop_shape = (64,64)
        frame, mask = self._get_single_crop(crop_shape)
        # pudb.set_trace()
        frame, mask = dataGen2D._augment(frame, mask, self.transform_cfg)
        
    def _load_frame_mask(self):
        # pudb.set_trace()
        frame = dataGen2D._load_img(self.test_frames_paths[0],
                                    normalize_inputs=True,
                                    ignore_last_channel=True)
        
        mask = dataGen2D._load_img(self.test_masks_paths[0],
                                    normalize_inputs=False,
                                    is_binary_mask=True,
                                    ignore_last_channel=False)
        return frame, mask
        
    def _load_crop_stack(self, crop_shape=(64,64)):
        frame, mask = self._load_frame_mask()
        frame = tf.convert_to_tensor(frame)
        mask = tf.convert_to_tensor(mask)
        # pudb.set_trace()
        return dataGen2D._random_crop(frame, mask,
                                      crop_shape,
                                      batch_crops=True)
    
    def _get_single_crop(self, crop_shape=(64,64)):
        frame_crop_stack, mask_crop_stack = self._load_crop_stack(crop_shape)
        
        frame_crop = tf.expand_dims(frame_crop_stack[0], axis=0)
        mask_crop = tf.expand_dims(mask_crop_stack[0], axis=0)
        
        return frame_crop, mask_crop
    
    @staticmethod
    def _assertshapes(frame, mask, frame_shape=(1,64,64,2), mask_shape=(1,64,64,1)):
        tfdebug.assert_shapes(
            [
                (frame, frame_shape),
                (mask, mask_shape),
            ]
        )
        
    
    
    
    
        