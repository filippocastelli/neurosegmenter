import unittest
import pytest
from mock import patch

from pathlib import Path
import os, sys, inspect
import pudb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import numpy.testing as nptest
import yaml
# import pudb

import tensorflow as tf
import tensorflow.debugging as tfdebug

from datagens.datagen2d import dataGen2D
# from utils import BatchInspector2D
from config import TrainConfig

CURRENT_DIR_PATH = Path(currentdir)
YML_CONFIG_PATH = CURRENT_DIR_PATH.joinpath("test_cfg_datagen.yml")
TRAIN_CFG_PATH = CURRENT_DIR_PATH.joinpath("test_train_cfg.yml")

FRAME_SHAPES = [(512,512),(512,512,2)]
FRAME_SHAPES_IDS = ["frame512_512", "frame512_512_512_2"]

CROP_SHAPES = [(64,64), (128,128)]
CROP_SHAPES_IDS = ["crop64_64", "crop128_128"]

CHANNELS = [1,2]
CHANNELS_IDS = ["ch1", "ch2"]

BATCH_SIZES_AUGMENTATION = [1,]
BATCH_SIZES_IDS_AUGMENTATION = ["batch1",]

class YMLio:
    @staticmethod
    def read_yml(yml_path):
        with yml_path.open(mode="r") as yml_in:
            dict_yml = yaml.load(yml_in, Loader=yaml.FullLoader)
        return dict_yml
    
    @staticmethod
    def write_yml(yml_dict, yml_out_path):
        with yml_out_path.open(mode="w") as yml_out:
            yaml.dump(yml_dict, yml_out)
        return
    
    @classmethod
    def combine_ymls(cls, yml_dict_default_path, yml_dict_path, yml_out_path):
        yml_dict = cls.read_yml(yml_dict_path)
        yml_default_dict = cls.read_yml(yml_dict_default_path)
        yml_default_dict.update(yml_dict)
        cls.write_yml(yml_default_dict, yml_out_path)
        return 
    
    
@pytest.fixture(scope="module", params=["single_images", "stack", "multi_stack"])
def config_yml_path_fixture(request):
    dataset_yml_name = f"dataset_{request.param}.yml"
    dataset_yml_path = CURRENT_DIR_PATH.joinpath(dataset_yml_name)
    # print(dataset_yml_name)
    default_yml_path = TRAIN_CFG_PATH
    YMLio.combine_ymls(yml_dict_default_path=default_yml_path,
                       yml_dict_path=dataset_yml_path,
                       yml_out_path=YML_CONFIG_PATH)
        
    return YML_CONFIG_PATH, request.param

@pytest.fixture(scope="module", params=FRAME_SHAPES, ids=FRAME_SHAPES_IDS)
def frame_mask(request):
    frame = np.random.uniform(low=0.0, high=1.0, size=request.param)
    mask = np.where(frame < 0.5, 1., 0)
    if len(request.param) > 2:
        # single channel segmentation only
        mask = np.expand_dims(mask[...,0], axis=-1)
    return tf.convert_to_tensor(frame), tf.convert_to_tensor(mask)

@pytest.fixture(scope="module")
def get_crop_fixture():
    def _get_crop_fixture(batch_size, crop_shape, n_channels):
        full_shape_frames = [batch_size, *crop_shape, n_channels]
        full_shape_masks = [batch_size, *crop_shape, 1]
        
        frame_batch = np.random.uniform(low=0.0, high=1.0, size=full_shape_frames).astype(np.float32)
        mask_batch = np.random.choice([1,0, 0,0], size=full_shape_masks).astype(np.float32)
        
        return tf.convert_to_tensor(frame_batch), tf.convert_to_tensor(mask_batch)
    return _get_crop_fixture

# @pytest.fixture
# def get_crop_batch(request):
#     batch_size, crop_shape, n_channels = request.param
    
#     full_shape_frames = [batch_size, *crop_shape, n_channels]
#     full_shape_masks = [batch_size, *crop_shape, 1]
        
#     frame_batch = np.random.uniform(low=0.0, high=1.0, size=full_shape_frames)
#     mask_batch = np.random.choice([1,0, 0,0], size=full_shape_masks)
    
#     return frame_batch, mask_batch

class TestDatagen2D:
        
    @pytest.mark.ensemble_datagen2D
    @pytest.mark.parametrize("data_augmentation", [True, False], ids=["augment", "no_augment"])
    @pytest.mark.parametrize("partition", ["train", "test", "val"], ids=["train", "test", "val"])
    @pytest.mark.parametrize("normalize_inputs", [True, False], ids=["normalized", "not_normalized"])
    def test_setup_datagen(self, config_yml_path_fixture,
                            data_augmentation,
                            partition,
                            normalize_inputs):
        yml_cfg_path, dataset_mode = config_yml_path_fixture
        config = TrainConfig(yml_cfg_path)
        # print(dataset_mode)
        # print(config.dataset_mode)
        # if dataset_mode == "single_images":
        datagen = dataGen2D(config=config,
                            partition=partition,
                            data_augmentation=data_augmentation,
                            normalize_inputs=normalize_inputs)
        
        batch_frames, batch_masks = next(datagen.data.__iter__())
        batch_frames_shape = batch_frames.shape.as_list()
        batch_masks_shape = batch_masks.shape.as_list()
        
        batch_frames_npy = batch_frames.numpy().flatten()
        batch_mask_npy = batch_masks.numpy().flatten()
        
        if normalize_inputs:
            frame_max = batch_frames_npy.max()
            frame_min = batch_frames_npy.min()
            
            assert frame_max <= 1., "frame value > 1"
            assert frame_min >= 0., "frame value < 0"
        
        mask_max = batch_mask_npy.max()
        mask_min = batch_mask_npy.min()
        
        assert mask_max <= 1., "mask value > 1"
        assert mask_min >= 0., "mask value < 0"
        
        if not tf.is_tensor(batch_frames):
            raise TypeError("batch_frames is not a tf.Tensor")
        if not tf.is_tensor(batch_masks):
            raise TypeError("batch_masks is not a tf.Tensor")
            
        assert batch_frames_shape == batch_masks_shape, "different shapes"
        
        batch_n, size_z, size_y, size_c = batch_frames_shape
        assert batch_n == config.batch_size, "different batch size"
        assert [size_z, size_y] == config.crop_shape, "spatial dimensions different"
        assert size_c == config.n_channels, "different number of channels"
        
            
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    def test_random_crop(self, frame_mask, crop_shape):
        batch_crops = False
        frame, mask = frame_mask
        frame_shape = frame.shape.as_list()
        
        if len(frame_shape) == 2:
            frame_y, frame_x  = frame_shape
            frame_ch = 1
        elif len(frame_shape) == 3:
            frame_y, frame_x, frame_ch = frame_shape
        else:
            raise ValueError("unrecognized number of framme dimensions")

        frame_crop, mask_crop = dataGen2D._random_crop(frame, mask,
                                                       crop_shape,
                                                       batch_crops=batch_crops)
        
        frame_crop_shape = frame_crop.shape.as_list()
        mask_crop_shape = mask_crop.shape.as_list()
        
        if batch_crops:
            n_pix_frame = frame_x*frame_y
            n_pix_crop = np.prod(crop_shape)
            expected_n_crops = n_pix_frame // n_pix_crop
        else:
            expected_n_crops = 1
            
        full_crop_shape_frames = [expected_n_crops, *crop_shape, frame_ch]
        full_crop_shape_masks = [expected_n_crops, *crop_shape, 1]
        
        assert frame_crop_shape == full_crop_shape_frames, "frame crop has wrong shape"
        assert mask_crop_shape == full_crop_shape_masks, "mask crop has wrong shape"
        
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_rot90_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._rot90_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])
        
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_mirror_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._mirror_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])
        
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_brightness_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._brightness_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])     
        
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_brightness_multiplicative_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._brightness_multiplicative_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])    

    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_gaussian_noise_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._gaussian_noise_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])    
                
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_gamma_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._gamma_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])
                
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_spatial_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._spatial_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])
            
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_zoom_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._zoom_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])
        
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_rotation_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._rotation_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)])
    @pytest.mark.augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    def test_rotation_zoom_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = dataGen2D._rotation_zoom_transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)]) 
        
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
        # pudb.set_trace()
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
        
        pudb.set_trace()
        return frame_crop, mask_crop
    
    @staticmethod
    def _assertshapes(frame, mask, frame_shape=(1,64,64,2), mask_shape=(1,64,64,1)):
        tfdebug.assert_shapes(
            [
                (frame, frame_shape),
                (mask, mask_shape),
            ]
        )
        
    
    
    
    
        