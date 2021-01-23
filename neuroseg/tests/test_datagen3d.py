import unittest
from pathlib import Path
import os, sys, inspect
import pudb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import numpy.testing as nptest
import pudb

import tensorflow as tf
import tensorflow.debugging as tfdebug

from datagens.datagen3d import datagen3DSingle
from utils import BatchInspector3D
from config import TrainConfig

class Datagen3DSingleTest(unittest.TestCase):
    
    def setUp(self):
        self.currentdir = Path(currentdir)
        self.config_path = self.currentdir.joinpath("test_train_cfg_3d.yml")
        self.config = TrainConfig(self.config_path)
        
        self.tmp_path = self.config.temp_path
        self.logs_path = self.config.logs_path
        
        self.crop_shape = self.config.crop_shape
        self.batch_size = self.config.batch_size
        self.n_channels = self.config.n_channels
        
        self.transform_cfg = self.config.da_transform_cfg
        
        self.datagen_noaugment = datagen3DSingle(self.config, data_augmentation=False)
        self.datagen_augment =  datagen3DSingle(self.config, data_augmentation=True)
        
    def tearDown(self):
        self._rm_tree(self.tmp_path)
        
    @staticmethod
    def _get_frames_batch_shape(batch_size, crop_shape, n_channels):
        return tuple(np.concatenate([(batch_size,),crop_shape,(n_channels,)]))
    
    @staticmethod
    def _get_masks_batch_shape(batch_size, crop_shape):
       return tuple(np.concatenate([(batch_size,),crop_shape,(1,)]))
    
    @classmethod
    def _rm_tree(cls, path):
        path = Path(path)
        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                cls._rm_tree(child)
        path.rmdir()
        
    def test_augment_shapes(self):
        datagen_iter = self.datagen_augment.iter
        batch = next(datagen_iter())
        assert len(batch) == 2, "wrong batch length"
        frames = batch[0]
        masks = batch[1]
        expected_frames_batch_shape = self._get_frames_batch_shape(self.batch_size,
                                                               self.crop_shape,
                                                               self.n_channels)
        expected_masks_batch_shape = self._get_masks_batch_shape(self.batch_size,
                                                               self.crop_shape)
        assert frames.shape == expected_frames_batch_shape, "wrong frame batch shape"
        assert masks.shape == expected_masks_batch_shape, "wromg mask batch shape"
        
        
    def test_noaugment_shapes(self):
        datagen_iter = self.datagen_noaugment.iter
        batch = next(datagen_iter())
        assert len(batch) == 2, "wrong batch length"
        frames = batch[0]
        masks = batch[1]
        expected_frames_batch_shape = self._get_frames_batch_shape(self.batch_size,
                                                               self.crop_shape,
                                                               self.n_channels)
        expected_masks_batch_shape = self._get_masks_batch_shape(self.batch_size,
                                                               self.crop_shape)
        assert frames.shape == expected_frames_batch_shape, "wrong frame batch shape"
        assert masks.shape == expected_masks_batch_shape, "wromg mask batch shape"