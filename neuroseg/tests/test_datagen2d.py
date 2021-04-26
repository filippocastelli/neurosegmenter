import pytest
# from mock import patch
from pathlib import Path
import os, sys, inspect
# import pudb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import yaml

import tensorflow as tf
import tensorflow.debugging as tfdebug

from datagens.datagen2d import dataGen2D
from config import TrainConfig

CURRENT_DIR_PATH = Path(currentdir)
CONFIG_PATH = CURRENT_DIR_PATH.joinpath("test_configs")

DATASETS_PATH = CURRENT_DIR_PATH.joinpath("test_datasets")
DATASET_CONFIGS_PATH = DATASETS_PATH.joinpath("configs")

YML_CONFIG_PATH = CONFIG_PATH.joinpath("test_cfg_datagen.yml")
TRAIN_CFG_PATH = CONFIG_PATH.joinpath("test_train_cfg.yml")

FRAME_SHAPES = [(512,512),(512,512,2)]
FRAME_SHAPES_IDS = ["frame512_512", "frame512_512_512_2"]

CROP_SHAPES = [(64,64), (128,128)]
CROP_SHAPES_IDS = ["crop64_64", "crop128_128"]

CHANNELS = [1,2]
CHANNELS_IDS = ["ch1", "ch2"]

BATCH_SIZES_AUGMENTATION = [1,]
BATCH_SIZES_IDS_AUGMENTATION = ["batch1",]

AUGMENTATION_TRANSFORMS = [
    dataGen2D._rot90_transform,
    dataGen2D._mirror_transform,
    dataGen2D._brightness_transform,
    dataGen2D._brightness_multiplicative_transform,
    dataGen2D._gaussian_noise_transform,
    dataGen2D._gamma_transform,
    dataGen2D._zoom_transform,
    dataGen2D._rotation_transform,
    dataGen2D._rotation_zoom_transform,
    dataGen2D._spatial_transform,
    ]

AUGMENTATION_TRANSFORMS_IDS = [
    "transform_rot90",
    "transform_mirror",
    "transform_brightness",
    "transform_brightness_multiplicative",
    "transform_gaussian_noise",
    "transform_gamma",
    "transform_zoom",
    "transform_rotation",
    "transform_rotation_zoom",
    "transform_spatial"
    ]

DATASETS = [
    # "single_images_1ch",
    "single_images_2ch",
    # "stack_1ch",
    # "multi_stack_1ch"
    ]
    
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
        return yml_default_dict
    
@pytest.fixture(params=DATASETS)
def config_yml_path_fixture(request):
    # pudb.set_trace()
    dataset_yml_name = f"{request.param}.yml"
    dataset_yml_path = DATASET_CONFIGS_PATH.joinpath(dataset_yml_name)
    # print(dataset_yml_name)
    default_yml_path = TRAIN_CFG_PATH
    combined_dict = YMLio.combine_ymls(yml_dict_default_path=default_yml_path,
                       yml_dict_path=dataset_yml_path,
                       yml_out_path=YML_CONFIG_PATH)
    yield YML_CONFIG_PATH, combined_dict
    YML_CONFIG_PATH.unlink(missing_ok=False)
    

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


class TestDatagen2D:
    @pytest.mark.neuroseg_ensemble_datagen2D
    @pytest.mark.parametrize("data_augmentation", [True, False], ids=["augment", "no_augment"])
    @pytest.mark.parametrize("partition", ["train", "test", "val"], ids=["train", "test", "val"])
    @pytest.mark.parametrize("normalize_inputs", [True, False], ids=["normalized", "not_normalized"])
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    def test_setup_datagen(self, config_yml_path_fixture,
                            data_augmentation,
                            partition,
                            normalize_inputs,
                            crop_shape):
        yml_cfg_path, cfg_dict = config_yml_path_fixture
        # if cfg_dict["dataset_cfg"]["n_channels"] == 2:
        #     import pudb
        #     pudb.set_trace()
        cfg_dict["model_cfg"]["crop_shape"] = crop_shape
        config = TrainConfig(cfg_dict=cfg_dict)
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
            
        assert batch_frames_shape[1:-1] == batch_masks_shape[1:-1], "different spatial shapes between mask_batch and frame_batch"
        
        batch_n, size_z, size_y, size_c = batch_frames_shape
        assert batch_n == config.batch_size, "different batch size"
        assert [size_z, size_y] == list(config.crop_shape), "spatial dimensions different"
        assert size_c == config.n_channels, "different number of channels"
        
            
    @pytest.mark.neuroseg_augmentation2D
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


    @pytest.mark.neuroseg_augmentation2D
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("n_channels", CHANNELS, ids=CHANNELS_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES_AUGMENTATION, ids=BATCH_SIZES_IDS_AUGMENTATION)
    @pytest.mark.parametrize("transform", AUGMENTATION_TRANSFORMS, ids=AUGMENTATION_TRANSFORMS_IDS)
    def test_augmentation_transform(self, get_crop_fixture, crop_shape, n_channels, batch_size, transform):
        frame_batch, mask_batch = get_crop_fixture(batch_size, crop_shape, n_channels)
        transformed_frames, transformed_masks = transform(frame_batch, mask_batch)
        
        frame_expected_shape = [batch_size, *crop_shape, n_channels]
        mask_expected_shape = [batch_size, *crop_shape, 1]
        
        tfdebug.assert_shapes([(transformed_frames, frame_expected_shape),
                               (transformed_masks, mask_expected_shape)]) 