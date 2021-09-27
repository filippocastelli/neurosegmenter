import pytest
from pathlib import Path
import os, sys, inspect
import pudb
from contextlib import nullcontext

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
# import yaml
# import pudb

from neuroseg.tiledpredict.tp3d import TiledPredictor3D


# from utils import BatchInspector3D
# from config import TrainConfig

def gen_ids(elems, name, separator="_"):
    str_elems = lambda x: [str(elem) for elem in x]
    if isinstance(elems[0], tuple) or isinstance(elems[0], list):
        elem_list = [str_elems(elem) if elem is not None else ["None", ] for elem in elems]
    else:
        elem_list = [str(elem) if elem is not None else ["None", ] for elem in elems]

    ids = [name + "_" + separator.join(str_elem) for str_elem in elem_list]
    return ids


CURRENT_DIR_PATH = Path(currentdir)
CONFIG_PATH = CURRENT_DIR_PATH.joinpath("test_configs")

DATASETS_PATH = CURRENT_DIR_PATH.joinpath("test_datasets")
DATASET_CONFIGS_PATH = DATASETS_PATH.joinpath("configs")

YML_CONFIG_PATH = CONFIG_PATH.joinpath("test_cfg_datagen.yml")
TRAIN_CFG_PATH = CONFIG_PATH.joinpath("test_train_cfg_3d.yml")

FRAME_SHAPES = [
    # (165, 327, 1024, 1),
    # (170, 334, 1024, 1),
    (512, 512, 512, 2),
    # (513,512,512,2),
    (512, 512, 512),
    # (513,512,512),
    (513, 513, 512)
]
FRAME_SHAPES_IDS = gen_ids(FRAME_SHAPES, "frame")

BATCH_SIZES = [
    1,
    # 2,
    # 3,
]
BATCH_SIZES_IDS = gen_ids(BATCH_SIZES, name="batch_size")

# CHUNK_SIZES = [1, 10, 21, None]
# CHUNK_SIZES_IDS = gen_ids(CHUNK_SIZES, name="chunk")

CROP_SHAPES = [
    np.array([32, 128, 128]),
    # (32, 128, 128),
    (64, 64, 64),
    (65, 64, 64),
    # (128,128,32)
]
CROP_SHAPES_IDS = gen_ids(CROP_SHAPES, "crop")

PAD_MODES = [
    "constant",
    "reflect"
]

PAD_WIDTHS = [
    ((0, 0), (0, 0), (0, 0)),
    ((0, 1), (0, 0), (0, 0)),
]

WINDOW_STEPS = [1, 2, 3, (1, 1, 1), (2, 2, 2)]
WINDOW_STEP_IDS = gen_ids(WINDOW_STEPS, name="window_step")

WINDOW_OVERLAP = [None, 10, 11, (5, 10, 10)]
WINDOW_OVERLAP_IDS = gen_ids(WINDOW_OVERLAP, name="window_overlap")

class mockModel:
    def __init__(
            self,
            pred_classes=1):
        self.pred_classes = pred_classes

    def predict(self, inputs):
        inputs_shape = inputs.shape
        if len(inputs_shape) == 5:
            batchsize, n_z, n_y, n_x, channels = inputs.shape
        elif len(inputs_shape) == 4:
            batchsize, n_z, n_y, n_x = inputs.shape
        else:
            raise ValueError("invalid input hsape for mockModel")
        return np.zeros(shape=(batchsize, n_z, n_y, n_x, self.pred_classes))


# pudb.set_trace()
@pytest.fixture(params=FRAME_SHAPES, ids=FRAME_SHAPES_IDS)
def input_volume_fixture(request):
    frame = np.random.uniform(low=0, high=255, size=request.param).astype(np.uint8)
    return frame


class TestDataPredictor3D:
    @pytest.mark.neuroseg_datapredictor3d_ensemble
    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_SIZES_IDS)
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    def test_init_tiledpredictor3D(self, input_volume_fixture, batch_size, crop_shape):
        fail_condition = not all((np.array(crop_shape) % 2) == 0)
        if fail_condition:
            context = pytest.raises(ValueError)
        else:
            context = nullcontext()
        with context:
            model = mockModel()
            tp = TiledPredictor3D(
                input_volume=input_volume_fixture,
                batch_size=batch_size,
                window_size=crop_shape,
                padding_mode="reflect",
                model=model
            )
            tp.predict()

    @pytest.mark.neuroseg_datapredictor3d
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("frame_shape", FRAME_SHAPES, ids=FRAME_SHAPES_IDS)
    def test_get_paddings(self, crop_shape, frame_shape):
        crop_shape_array = np.array(crop_shape)
        frame_shape_array = np.array(frame_shape)
        frame_shape_spatial = frame_shape_array[:3]

        fail_condition = not all((crop_shape_array % 2) == 0)
        context = pytest.raises(ValueError) if fail_condition else nullcontext()

        with context:
            paddings = TiledPredictor3D.get_paddings(frame_shape, crop_shape)

            assert len(paddings) == len(frame_shape_spatial), "Paddings ahve different length than expected"
            pad_lengths = np.array([len(padding) for padding in paddings])
            assert all(pad_lengths == 2), "Some paddings don't have length == 2"

            # total_pads_expected = (frame_shape_spatial - crop_shape_array) %
            # total_pads = np.array([sum(padding) for padding in paddings])

            # assert (total_pads_expected == total_pads).all(), "padding sums are different than expected"
            pad_elem_types = [type(elem) for elem in paddings]
            assert len(set(pad_elem_types)) == 1, "paddings have different types"

    @pytest.mark.neuroseg_datapredictor3d
    @pytest.mark.parametrize("pad_widths", PAD_WIDTHS)
    @pytest.mark.parametrize("pad_modes", PAD_MODES)
    def test_pad_image(self, input_volume_fixture, pad_widths, pad_modes):
        padded_img = TiledPredictor3D.pad_image(img=input_volume_fixture,
                                                pad_widths=pad_widths)
        vol_shape = input_volume_fixture.shape

        assert type(padded_img) == np.ndarray, "padded_img is not an image"

        total_pads = np.array([sum(padding) for padding in pad_widths])
        # pudb.set_trace()
        while len(total_pads) != len(vol_shape):
            total_pads = np.append(total_pads, 0)

        expected_total_shape = input_volume_fixture.shape + total_pads

        assert (np.array(padded_img.shape) == expected_total_shape).all(), "padded image has wrong shape"

    @pytest.mark.neuroseg_datapredictor3d
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("step", WINDOW_STEPS, ids=WINDOW_STEP_IDS)
    def test_patchify3D(self, input_volume_fixture, crop_shape, step):
        img_spatial_shape = input_volume_fixture.shape[:3]
        mod = (np.array(img_spatial_shape) - np.array(crop_shape)) % step
        fail_condition = not (mod == 0).all()

        context = pytest.raises(ValueError) if fail_condition else nullcontext()

        with context:
            windows = TiledPredictor3D.get_patch_windows(input_volume_fixture, crop_shape, step)
            # print(windows.shape)

            window_shape = windows.shape
            assert len(window_shape) in [6, 8], "window view should have 6 or 8 dims"
            if len(window_shape) == 8:
                assert window_shape[3] == 1, "there should be no sliding over channels"
            if len(window_shape) == 46:
                assert window_shape[-3:] == crop_shape, "crop_shapes dims"

    @pytest.mark.neuroseg_datapredictor3d
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_SIZES_IDS)
    @pytest.mark.parametrize("window_overlap", WINDOW_OVERLAP, ids=WINDOW_OVERLAP_IDS)
    def test_predict_tiles(self, input_volume_fixture, crop_shape, batch_size, window_overlap):
        img_spatial_shape = input_volume_fixture.shape[:3]
        if window_overlap is not None:
            step = np.array(crop_shape)[:3] - np.array(window_overlap)
        else:
            step = (np.array(crop_shape) // 2)[:3]
        mod = (np.array(img_spatial_shape) - np.array(crop_shape)) % step
        fail_condition = not (mod == 0).all()

        context = pytest.raises(ValueError) if fail_condition else nullcontext()

        with context:
            windows = TiledPredictor3D.get_patch_windows(input_volume_fixture, crop_shape, step)

            model = mockModel()

            predicted_tiles = TiledPredictor3D.predict_tiles(
                img_windows=windows,
                frame_shape=input_volume_fixture.shape,
                model=model,
                batch_size=batch_size,
            )
            # print(predicted_tiles.shape)
            # print(input_volume_fixture.shape)

            expected_output_shape = tuple([*input_volume_fixture.shape[:3], 1])

            assert expected_output_shape == predicted_tiles.shape, "Wrong output shape"

    @pytest.mark.neuroseg_datapredictor3d
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    def test_weighting_window(self, crop_shape):
        weight_window = TiledPredictor3D.get_weighting_window(window_size=crop_shape)
        epxected_window_shape = (*crop_shape, 1)
        assert weight_window.shape == epxected_window_shape, "weight window has different shape"
        assert np.max(weight_window) <= 1 and np.min(weight_window) >= 0, "weight window is not normalized"

    @pytest.mark.neuroseg_datapredictor3d
    @pytest.mark.parametrize("crop_shape", CROP_SHAPES, ids=CROP_SHAPES_IDS)
    def test_unpad_img(self, input_volume_fixture, crop_shape):

        fail_condition = not all((np.array(crop_shape) % 2) == 0)
        context = pytest.raises(ValueError) if fail_condition else nullcontext()

        with context:
            frame_shape = input_volume_fixture.shape
            paddings = TiledPredictor3D.get_paddings(frame_shape, crop_shape)
            padded_img = TiledPredictor3D.pad_image(input_volume_fixture, paddings)
            unpadded_img = TiledPredictor3D.unpad_image(padded_img, paddings)

            print(paddings)
            assert unpadded_img.shape == frame_shape, "unpadded and original image have different shapes"
