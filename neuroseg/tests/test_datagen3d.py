import pytest
from pathlib import Path
import os, sys, inspect
import pudb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import yaml

from neuroseg.datagens.datagen3d import DataGen3D
# from utils import BatchInspector3D
from neuroseg.config import TrainConfig

CURRENT_DIR_PATH = Path(currentdir)
CONFIG_PATH = CURRENT_DIR_PATH.joinpath("test_configs")

DATASETS_PATH = CURRENT_DIR_PATH.joinpath("test_datasets")
DATASET_CONFIGS_PATH = DATASETS_PATH.joinpath("configs")

YML_CONFIG_PATH = CONFIG_PATH.joinpath("test_cfg_datagen.yml")
TRAIN_CFG_PATH = CONFIG_PATH.joinpath("test_train_cfg_3d.yml")

DATASETS = [
    "single_images_1ch",
    # "single_images_2ch",
    "stack_1ch",
    "multi_stack_1ch"
]

CROP_SHAPES = [
    (64, 64, 64),
    (32, 64, 64),
    # (128,128,128)
]
CROP_SHAPES_IDS = [
    "crop64_64_64",
    "crop32_64_64",
    # "crop128_128_128"
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


class TestDatagen3D:
    @pytest.mark.neuroseg_ensemble_datagen3D
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
        cfg_dict["model_cfg"]["crop_shape"] = crop_shape
        # pudb.set_trace()
        config = TrainConfig(cfg_dict=cfg_dict)

        datagen = DataGen3D(config=config,
                            partition=partition,
                            data_augmentation=data_augmentation)

        # batch_frames and batch_masks are numpy arrays, not Tensors
        batch_frames, batch_masks = next(datagen.data.__iter__())

        batch_frames_shape = batch_frames.shape
        batch_masks_shape = batch_masks.shape

        frame_flatten = batch_frames.flatten()
        mask_flatten = batch_masks.flatten()

        if normalize_inputs:
            frame_max = frame_flatten.max()
            frame_min = frame_flatten.min()

            assert frame_max <= 1., "frame value > 1"
            assert frame_min >= 0., "frame value < 0"

        mask_max = mask_flatten.max()
        mask_min = mask_flatten.min()

        assert mask_max <= 1., "mask value > 1"
        assert mask_min >= 0., "mask value < 0"

        assert batch_frames_shape == batch_masks_shape, "different shapes"

        batch_n, size_z, size_y, size_x, size_c = batch_frames_shape
        assert batch_n == config.batch_size, "different batch size"
        assert [size_z, size_y, size_x] == list(config.crop_shape), "spatial dimensions different"
        assert size_c == config.n_channels, "different number of channels"
