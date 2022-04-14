import numpy as np
from typing import Union
from pathlib import Path

from tensorflow.python.keras.models import load_model

from neuroseg.utils import load_volume, save_volume, glob_imgs
# from neuroseg.config.config import SUPPORTED_STACK_FORMATS
from skimage import io as skio
from neuroseg.config import TrainConfig, PredictConfig
from skimage.util import view_as_windows
from tqdm import tqdm
import tifffile


class SingleImagesDataPredictor:

    def __init__(self, config: Union[PredictConfig, TrainConfig], model=None, in_fpath=None):
        self.config = config
        self.mode = self.config.config_type
        self.to_8bit = config.to_8bit

        if in_fpath is None:
            if self.config.config_type == "training":
                self.data_path = self.config.test_paths["frames"]
            else:
                self.data_path = self.config.data_path
        else:
            self.data_path = in_fpath

        self.output_path = self.config.output_path

        self._supported_ext = [".png", ".tif", ".tiff"]

        self.image_list = sorted([fpath for fpath in self.data_path.glob("*") if fpath.suffix in self._supported_ext])
        self.normalize_data = self.config.normalize_data if self.mode == "predict" else self.config.normalize_inputs

        self.model = model
        self.keep_stack = True if self.mode == "training" else False
        self.predicted_data = self.predict(keep_stack=self.keep_stack)

    def predict(self, keep_stack=False):
        img_list = []
        for fpath in tqdm(self.image_list):
            img = self.load_img(fpath)
            pred = SingleImageTiledPredictor(input_img=img,
                                             batch_size=self.config.batch_size,
                                             window_size=self.config.window_size,
                                             # output_class_values=self.config.class_values,
                                             model=self.model,
                                             padding_mode=self.config.padding_mode,
                                             extra_padding_windows=self.config.extra_padding_windows,
                                             window_overlap=self.config.window_overlap)
            prediction = pred.predicted_data
            if self.keep_stack:
                img_list.append(prediction)
            out_fpath = self.output_path.joinpath(fpath.name)

            if self.to_8bit:
                save_arr = (pred.predicted_data * 255).astype(np.uint8)
            else:
                save_arr = pred.predicted_data

            save_arr = np.moveaxis(save_arr, source=-1, destination=0)
            save_arr = np.expand_dims(save_arr, axis=-1)
            tifffile.imsave(str(out_fpath), data=save_arr)

        if self.keep_stack:
            return np.stack(img_list, axis=0)
        else:
            return None

    @staticmethod
    def _get_norm_const(img):
        dtype = img.dtype
        if dtype in [float, np.float, np.float16, np.float32, np.float64]:
            norm = np.finfo(dtype).max
        elif dtype in [int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            norm = np.iinfo(dtype).max
        else:
            raise ValueError(f"unsupported dtype {dtype}")
        return norm

    def load_img(self, fpath: Path):
        if fpath.suffix in [".tif", ".tiff"]:
            img = tifffile.imread(str(fpath))
            if img.shape[0] == 3:
                img = np.moveaxis(img, 0, -1)
        else:
            img = skio.imread(str(fpath), plugin="pil")
        if self.config.ignore_last_channel:
            img = img[..., :2]
        norm = self._get_norm_const(img)
        if self.normalize_data:
            img = img / norm
        return img


class SingleImageTiledPredictor:
    def __init__(self,
                 input_img: np.ndarray,
                 batch_size: int = 5,
                 window_size: tuple = (128, 128),
                 # output_class_values: tuple = (0, 1, 2, 255),
                 model=None,
                 padding_mode: str = "reflect",
                 extra_padding_windows: int = 2,
                 tiling_mode: str = "average",
                 window_overlap: tuple = None,
                 ):
        self.input_img = input_img
        self.batch_size = batch_size
        self.window_size = np.array(window_size)
        # self.output_class_values = output_class_values
        self.model = model
        self.padding_mode = padding_mode
        self.extra_padding_windows = extra_padding_windows
        self.tiling_mode = tiling_mode
        self.window_overlap = window_overlap

        # asserting divisibility by 2
        for dim in range(len(self.window_size)):
            if not self.window_size[dim] % 2 == 0:
                raise ValueError("crop shape must be divisibile by 2 along all dims")

        # calculating step
        if self.window_overlap is not None:
            assert (np.array(self.window_overlap) % 2 == 0).all(), "window overlap must be divisible by 2"
            assert (np.array(self.window_overlap) - np.array(
                self.window_size) < 0).all(), "Window overlap must not be greater than crop_shape"

            self.step = np.array(self.window_size) - np.array(self.window_overlap)
        else:
            self.step = np.array(self.window_size) // 2

        self.padded_img, self.paddings = self.get_padded_img()
        self.patch_window_view = self.get_patch_windows()
        self.predicted_data = self.predict_tiles(tiling_mode=tiling_mode)
        self.predicted_data = self.unpad_image(self.predicted_data, self.paddings)

    def get_padded_img(self) -> (np.ndarray, list):
        """get padded image for window integer division"""
        image_shape = np.array(self.input_img.shape[:2])

        if self.step is None:
            step = self.window_size // 2
        else:
            step = np.array(self.step)

        # tot_paddings = [0, 0]
        # paddings = [(0, 0), (0, 0), (0, 0)]
        paddings = [(0, 0) for _ in range(len(self.input_img.shape))]

        paddings = [(0, 0), (0, 0)]
        a = image_shape + (self.extra_padding_windows - 1) * self.window_size

        tot_res_paddings = - (a % step)

        if any(tot_res_paddings < 0):
            # all paddings need to be positive
            for idx in range(len(tot_res_paddings)):
                if tot_res_paddings[idx] < 0:
                    tot_res_paddings[idx] = tot_res_paddings[idx] + step[idx]

        assert not any(tot_res_paddings < 0), "some paddings are negative"
        tot_paddings = (self.extra_padding_windows * self.window_size) + tot_res_paddings

        for idx in range(len(image_shape)):
            left_pad = tot_paddings[idx] // 2
            right_pad = tot_paddings[idx] - left_pad
            paddings[idx] = (left_pad, right_pad)

        if len(paddings) < len(image_shape):
            while len(paddings) != len(image_shape):
                paddings.append((0, 0))

        img = np.pad(self.input_img, paddings, mode=self.padding_mode)

        return img, paddings

    def get_patch_windows(self):
        if isinstance(self.step, int):
            step = [self.step, self.step]
        else:
            step = list(self.step)

        img_spatial_shape = self.padded_img.shape[:2]
        self.check_distortion_condition(img_spatial_shape, self.window_size, step)

        img_chans = None if len(self.padded_img.shape) == 2 else self.padded_img.shape[2]
        window_shape = list(self.window_size)
        if img_chans is not None:
            window_shape.append(img_chans)
            step.append(1)

        window_data_view = view_as_windows(arr_in=self.padded_img, window_shape=window_shape, step=step)
        return window_data_view

    @staticmethod
    def unpad_image(img, pad_widths):
        img_shape = img.shape

        unpadded = img[
                   pad_widths[0][0]: img_shape[0] - pad_widths[0][1],
                   pad_widths[1][0]: img_shape[1] - pad_widths[1][1]]

        return unpadded

    @staticmethod
    def check_distortion_condition(frame_shape, crop_shape, step):
        frame_shape = frame_shape[:2]
        crop_shape = crop_shape[:2]
        step = step[:2]
        mod = (np.array(frame_shape) - np.array(crop_shape)) % step
        if not (mod == 0).all():
            raise ValueError(
                "(img_shape - crop_shape) % step must be zeros to avoid reconstruction distorsions"
            )

    def predict_tiles(self,
                      tiling_mode: str = "weighted_average"):

        patch_view_shape = self.patch_window_view.shape

        if len(patch_view_shape) == 6:
            canvas_y, canvas_x, _, window_y, window_x, channels = patch_view_shape
            window_shape_spatial = np.array([window_y, window_x])
            window_shape = np.array([window_y, window_x, channels])

        elif len(patch_view_shape) == 4:
            canvas_y, canvas_x, window_y, window_x = patch_view_shape
            window_shape = np.array([window_y, window_x])
            window_shape_spatial = window_shape

        else:
            raise ValueError("unsupported window shape")

        if any((window_shape_spatial % 2) != 0):
            raise ValueError("the first two dimensions of window_shape should be divisible by 2")

        reshaped_windows = self.patch_window_view.reshape((-1, *window_shape))
        batched_inputs = self.divide_into_batches(reshaped_windows, self.batch_size)

        # out_img_shape = [*self.padded_img.shape[:2], len(self.output_class_values)]
        out_img_shape = [*self.padded_img.shape[:2], 1]
        output_img = np.zeros(out_img_shape, dtype=np.float32)
        weight_img = np.zeros(out_img_shape, dtype=np.float32)

        weight = self.get_weighting_window(window_shape_spatial) if tiling_mode == "weighted_average" else 1

        for batch_idx, batch in enumerate(batched_inputs):
            batch_global_index = int(batch_idx) * self.batch_size
            predicted_batch = self.model.predict(batch).astype(np.float32)

            for img_idx, pred_img in enumerate(predicted_batch):
                tile_idx = img_idx + batch_global_index
                canvas_index = np.array(np.unravel_index(tile_idx, self.patch_window_view.shape[:2]))
                pivot = canvas_index * self.step[:2]

                if tiling_mode in ["average", "weighted_average"]:
                    slice_y = slice(pivot[0], pivot[0] + window_shape[0])
                    slice_x = slice(pivot[1], pivot[1] + window_shape[1])

                    output_patch_shape = output_img[slice_y, slice_x].shape

                    if output_patch_shape != pred_img.shape:
                        raise ValueError("incorrect sliding window shape, check padding")
                    output_img[slice_y, slice_x] += pred_img
                    weight_img[slice_y, slice_x] += weight

                elif tiling_mode == "drop_borders":
                    assert all(np.array(self.window_overlap[
                                        :2]) % 2 == 0), "drop_borders mode needs window_overlap to be divisible by 2"
                    half_overlap = np.array(self.window_overlap) // 2

                    slice_y = slice(pivot[0] + half_overlap[0], pivot[0] + window_shape[0] - half_overlap[0])
                    slice_x = slice(pivot[1] + half_overlap[1], pivot[1] + window_shape[1] - half_overlap[1])

                    pred_img_dropped_borders = pred_img[
                                               half_overlap[0]: -half_overlap[0],
                                               half_overlap[1]: -half_overlap[1]]

                    output_patch_shape = output_img[slice_y, slice_x].shape
                    if output_patch_shape != pred_img_dropped_borders.shape:
                        raise ValueError("incorrect sliding window shape, check padding")

                    output_img[slice_y, slice_x] = pred_img_dropped_borders
                else:
                    raise ValueError(f"unsuppported tiling mode {tiling_mode}")

        final_img = output_img / weight_img
        return final_img

    @classmethod
    def get_weighting_window(cls, window_size, expand_dims=True):
        """Generate a 2D spline-based weighting window"""
        wind_y = cls.spline_window(window_size[1], power=2)
        wind_x = cls.spline_window(window_size[0], power=2)
        wind_y = np.expand_dims(wind_y, axis=-1)
        wind_x = np.expand_dims(wind_x, axis=-1)
        wind = wind_y.transpose() * wind_x
        wind = wind / wind.max()
        if expand_dims:
            wind = np.expand_dims(wind, axis=-1)
        return wind.astype("float32")

    @staticmethod
    def spline_window(window_linear_size, power=2):
        """ generates 1D spline window profile"""
        intersection = int(window_linear_size / 4)

        # noinspection PyUnresolvedReferences
        wind_outer = (abs(2 * (scipy.signal.triang(window_linear_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        # noinspection PyUnresolvedReferences
        wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_linear_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind

    @staticmethod
    def divide_into_batches(input_list, n):
        """divide a list in evenly sized batches of length n"""
        return [
            input_list[i * n: (i + 1) * n]
            for i in range((len(input_list) + n - 1) // n)
        ]
