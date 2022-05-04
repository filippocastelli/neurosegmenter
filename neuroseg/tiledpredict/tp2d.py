# from pathlib import Path
# import shutil
# from itertools import product
# import pickle
from contextlib import nullcontext
import numpy as np
import scipy
from skimage.util import view_as_windows
from tqdm import tqdm
import multiprocessing

from neuroseg.tiledpredict.datapredictorbase import DataPredictorBase
from neuroseg.utils import BatchInspector2D, toargmax
import tensorflow as tf


class DataPredictor2D(DataPredictorBase):
    def __init__(self, config, model=None, in_fpath=None):
        super().__init__(config, model, in_fpath=in_fpath)

    def predict(self):
        self.tiledpredictor = TiledPredictor2D(
            input_volume=self.input_data,
            batch_size=self.batch_size,
            n_output_classes=self.n_output_classes,
            window_size=self.window_size,
            model=self.prediction_model,
            padding_mode=self.padding_mode,
            extra_padding_windows=self.extra_padding_windows,
            tiling_mode=self.tiling_mode,
            window_overlap=self.window_overlap,
            debug=self.debug,
            multi_gpu=self.multi_gpu,
            n_tiling_threads=self.n_tiling_threads,
        )

        self.predicted_data = self.tiledpredictor.predict()

        if self.to_segmentation:
            self.predicted_data = toargmax(self.predicted_data, self.config.class_values, pos_value=1)

        return self.predicted_data


class MultiVolumeDataPredictor2D(DataPredictorBase):
    def __init__(self, config, model=None):
        super().__init__(config, model)

    def predict(self):
        tiled_predictors = {}
        for idx, volume in enumerate(tqdm(self.input_data)):
            volume_name = self.data_paths[idx].name

            tiled_predictor = TiledPredictor2D(
                input_volume=volume,
                batch_size=self.batch_size,
                window_size=self.window_size,
                model=self.prediction_model,
                padding_mode=self.padding_mode,
                n_output_classes=self.n_output_classes,
                extra_padding_windows=self.extra_padding_windows,
                tiling_mode=self.tiling_mode,
                window_overlap=self.window_overlap,
                debug=self.debug,
                multi_gpu=self.multi_gpu,
                n_tiling_threads=self.n_tiling_threads,
            )

            tiled_predictors[volume_name] = tiled_predictor

        self.predicted_data = {
            name: pred.predict() for name, pred in tiled_predictors.items()
        }

        # self.predicted_data = [tiledpredictor.output_volume for tiledpredictor in tiled_predictors]
        return self.predicted_data


class TiledPredictor2D:
    def __init__(
            self,
            input_volume,
            is_volume=True,
            batch_size=5,
            chunk_size=100,
            window_size=(128, 128),
            n_output_classes=1,
            model=None,
            padding_mode="reflect",
            extra_padding_windows=0,
            tiling_mode="average",
            window_overlap: tuple = None,
            debug: bool = False,
            multi_gpu: bool = False,
            n_tiling_threads: int = 1,
    ):
        self.input_volume = input_volume
        self.is_volume = is_volume
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.crop_shape = np.array(window_size)
        self.padding_mode = padding_mode
        self.model = model
        self.tiling_mode = tiling_mode
        self.extra_padding_windows = extra_padding_windows
        self.window_overlap = window_overlap
        self.debug = debug
        self.multi_gpu = multi_gpu
        self.n_tiling_threads = n_tiling_threads

        # self.tmp_folder = Path(tmp_folder)
        # self.keep_tmp = keep_tmp
        self.n_output_classes = n_output_classes
        # self.inference_volume = np.zeros()

        # asserting divisibility by 2
        for dim in range(len(self.crop_shape)):
            if not self.crop_shape[dim] % 2 == 0:
                raise ValueError("crop shape must be divisibile by 2 along all dims")
        # if not (self.crop_shape % 2 == 0).all():
        #     raise ValueError("crop shape must be divisible by 2 along all dims")
        # calculating steps

        if self.window_overlap is not None:
            assert (np.array(self.window_overlap) % 2 == 0).all(), "window overlap must be divisible by 2"
            assert (np.array(self.window_overlap) - np.array(
                self.crop_shape) < 0).all(), "Window overlap must not be greater than crop_shape"

            self.step = np.array(self.crop_shape) - np.array(self.window_overlap)
        else:
            self.step = np.array(self.crop_shape) // 2

        # self.check_distortion_condition(self.input_volume.shape, self.crop_shape, self.step)

    def predict(self):
        if self.is_volume:
            return self.predict_volume()
        else:
            return self.predict_image()

    def predict_volume(self):
        self.paddings = self.get_paddings(self.input_volume[0].shape,
                                          self.crop_shape,
                                          extra_windows=self.extra_padding_windows, )
        # not padding z
        self.paddings.insert(0, (0, 0))
        self.padded_volume = self.pad_image(self.input_volume, self.paddings, mode=self.padding_mode)
        self.padded_volume_shape = self.padded_volume.shape
        self.padded_img_shape = self.padded_volume[0].shape

        # self.prediction_volume = np.zeros_like(self.padded_volume)
        self.prediction_volume = np.zeros(shape=[*self.padded_volume.shape[:3], self.n_output_classes])

        # run self._pred_volume_slice() for each slice in volume using multiprocessing
        print("Making volume predictions...")

        img_windows = self.get_patch_windows(img=self.padded_volume, step=self.step, crop_shape=self.crop_shape,
                                             is_volume=True)
        self.prediction_volume = self.predict_tiles_volume(img_windows=img_windows,
                                                           frame_shape=self.padded_img_shape,
                                                           model=self.model,
                                                           batch_size=self.batch_size,
                                                           window_overlap=self.window_overlap,
                                                           tiling_mode=self.tiling_mode,
                                                           n_output_classes=self.n_output_classes,
                                                           multi_gpu=self.multi_gpu)
        """
        if self.n_tiling_threads > 1:
            with multiprocessing.Pool(processes=self.n_tiling_threads) as pool:
                pool_results = tqdm(pool.imap(self._pred_volume_slice, range(self.padded_volume.shape[0])),
                                    total=self.padded_volume.shape[0])
                res = tuple(pool_results)

            self.prediction_volume = np.array(res)
        else:
            for idx, img in enumerate(tqdm(self.padded_volume)):
                img_windows = self.get_patch_windows(img=img,
                                                     crop_shape=self.crop_shape,
                                                     step=self.step)

                predicted_tiles = self.predict_tiles(img_windows=img_windows,
                                                     frame_shape=self.padded_img_shape,
                                                     model=self.model,
                                                     batch_size=self.batch_size,
                                                     window_overlap=self.window_overlap,
                                                     tiling_mode=self.tiling_mode,
                                                     n_output_classes=self.n_output_classes,
                                                     multi_gpu=self.multi_gpu)
                self.prediction_volume[idx] = predicted_tiles
        """

        self.prediction_volume = self.unpad_volume(self.prediction_volume, self.paddings)
        return self.prediction_volume

    def _pred_volume_slice(self, vol_idx: bool):
        img = self.padded_volume[vol_idx]
        img_windows = self.get_patch_windows(img=img,
                                             crop_shape=self.crop_shape,
                                             step=self.step)
        predicted_tiles = self.predict_tiles(img_windows=img_windows,
                                             frame_shape=self.padded_img_shape,
                                             model=self.model,
                                             batch_size=self.batch_size,
                                             window_overlap=self.window_overlap,
                                             tiling_mode=self.tiling_mode,
                                             n_output_classes=self.n_output_classes,
                                             multi_gpu=self.multi_gpu)
        return predicted_tiles

    def predict_image(self):
        self.paddings = self.get_paddings(self.input_volume.shape, self.crop_shape)
        self.padded_volume = self.pad_image(
            self.input_volume, self.paddings, mode=self.padding_mode
        )
        self.padded_volume_shape = self.padded_volume.shape

        self.patch_window_view = self.get_patch_windows(
            img=self.padded_volume, crop_shape=self.crop_shape, step=self.step
        )
        self.prediction_volume_padded = self.predict_tiles(img_windows=self.patch_window_view,
                                                           frame_shape=self.padded_volume_shape,
                                                           model=self.model,
                                                           batch_size=self.batch_size,
                                                           tiling_mode=self.tiling_mode,
                                                           window_overlap=self.window_overlap,
                                                           multi_gpu=self.multi_gpu)

        self.prediction_volume = self.unpad_image(self.prediction_volume_padded, self.paddings)
        return self.prediction_volume

    @classmethod
    def get_paddings(cls,
                     image_shape,
                     crop_shape,
                     extra_windows=0,
                     step=None):
        """given image_shape and crop_shape get [(pad_left, pad_right)] paddings"""

        image_shape = np.array(image_shape)[:2]
        crop_shape = np.array(crop_shape)

        if step is None:
            step = crop_shape // 2
        step = np.array(step)

        if not (crop_shape % 2 == 0).all():
            raise ValueError("crop_shape should be divisible by 2")

        pad_list = [(0, 0) for _ in range(len(image_shape))]

        # non-distortion condition is (padded_shape - crop_shape) % step == 0

        # padded_shape = img_shape + paddings
        # paddings = res_paddings + extra_tiling_windows * crop_shape
        # padded_shape = img_shape + extra_tiling_windows * crop_shape + res_paddings
        # condition becomes
        # img_shape + (extra_tiling_windows - 1) * crop_shape + res_paddings % step = 0

        # which is in the form
        # a + x % b == 0
        # with a = img_shape + (extra_tiling_windows - 1) * crop_shape
        # b = step

        # im dumb so I bruteforce it.

        tot_paddings = [0, 0]
        tot_res_paddings = [0, 0]

        paddings = [(0, 0) for _ in tot_paddings]

        a = image_shape + (extra_windows - 1) * crop_shape
        b = step

        tot_res_paddings = - (a % b)

        if any(tot_res_paddings < 0):
            # making paddings positive
            for idx in range(len(tot_res_paddings)):
                if tot_res_paddings[idx] < 0:
                    tot_res_paddings[idx] = tot_res_paddings[idx] + step[idx]

        assert not any(tot_res_paddings < 0), "paddings must be positive"

        tot_paddings = (extra_windows * crop_shape) + tot_res_paddings

        for idx in range(len(image_shape)):
            left_pad = tot_paddings[idx] // 2
            right_pad = tot_paddings[idx] - left_pad
            paddings[idx] = (left_pad, right_pad)

        return paddings

    @staticmethod
    def pad_image(img, pad_widths, mode="constant"):
        img_shape = img.shape
        pad_width_list = list(pad_widths)
        """performing the padding"""
        # adding dims

        if len(pad_width_list) < len(img_shape):
            while len(pad_width_list) != len(img_shape):
                pad_width_list.append((0, 0))

        img = np.pad(img, pad_width_list, mode=mode)
        return img

    @classmethod
    def get_patch_windows(cls, img, crop_shape, step, is_volume=False):
        crop_shape = list(crop_shape)

        if isinstance(step, int):
            step = [step, step]
        else:
            step = list(step)

        if is_volume:
            img_spatial_shape = img.shape[1:3]
            img_chans = None if len(img.shape) == 3 else img.shape[3]
            cls.check_distortion_condition(img_spatial_shape, crop_shape, step)
            window_shape = [1] + crop_shape
            view_step = [1] + step
        else:
            img_spatial_shape = img.shape[:2]
            img_chans = None if len(img.shape) == 2 else img.shape[2]
            cls.check_distortion_condition(img_spatial_shape, crop_shape, step)
            window_shape = crop_shape
            view_step = step

        window_shape = crop_shape

        if img_chans is not None:
            window_shape.append(img_chans)
            view_step.append(1)

        view = view_as_windows(arr_in=img, window_shape=window_shape, step=view_step)
        return view

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

    @classmethod
    def predict_tiles_volume(cls,
                             img_windows,
                             frame_shape,
                             model,
                             batch_size,
                             n_output_classes=1,
                             window_overlap=None,
                             tiling_mode="average",
                             multi_gpu: bool = False):

        view_shape = img_windows.shape
        if len(view_shape) == 8:
            canvas_z, canvas_y, canvas_x, _, window_z, window_y, window_x, channels = view_shape
            window_shape_spatial = np.array([window_y, window_x])
            window_shape = np.array([1, window_y, window_x, channels])
        elif len(view_shape) == 6:
            canvas_z, canvas_y, canvas_x, _, window_y, window_x = view_shape
            window_shape = np.array([1, window_y, window_x])
            window_shape_spatial = window_shape
        else:
            raise ValueError("unrecognized img_windows shape")

        if all((window_shape_spatial % 2) != 0):
            raise ValueError("the first two dimensions of window_shape should be divisible by 2")

        if window_overlap is not None:
            step = np.array(window_shape_spatial) - np.array(window_overlap)
        else:
            step = np.array(window_shape_spatial) // 2
            window_overlap = step

        reshaped_windows = img_windows.reshape((-1, *window_shape))

        out_img_shape = [*frame_shape[:2], n_output_classes]
        output_img = np.zeros(out_img_shape, dtype=np.float)
        weight_img = np.ones_like(output_img)

        weight = cls.get_weighting_window(window_shape_spatial) if tiling_mode == "weighted_average" else 1

        ds = tf.data.Dataset.from_tensor_slices(reshaped_windows)
        ds = ds.batch(batch_size)

        if not multi_gpu:
            context = nullcontext
        else:
            gpus = tf.config.list_logical_devices('GPU')
            context = tf.distribute.MirroredStrategy(gpus).scope
            batch_options = tf.data.Options()
            batch_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            ds = ds.with_options(batch_options)

        with context():
            predictions = model.predict(ds).astype(np.float)

        for img_idx, pred_img in tqdm(enumerate(predictions)):
            canvas_index = np.array(np.unravel_index(img_idx, img_windows.shape[:2]))

            pivot = canvas_index * step[:2]

            if tiling_mode in ["average", "weighted_average"]:
                slice_y = slice(pivot[0], pivot[0] + window_shape[0])
                slice_x = slice(pivot[1], pivot[1] + window_shape[1])

                output_patch_shape = output_img[slice_y, slice_x].shape
                if output_patch_shape != pred_img.shape:
                    raise ValueError("incorrect sliding window shape, check padding")
                output_img[slice_y, slice_x] += pred_img
                weight_img[slice_y, slice_x] += weight

            elif tiling_mode == "drop_borders":
                assert all(
                    np.array(window_overlap[:2]) % 2 == 0), "drop_borders mode need window_overlap to be divisible by 2"
                half_overlap = np.array(window_overlap) // 2
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
            SAVE_DEBUG_TIFFS_FLAG = False
            if SAVE_DEBUG_TIFFS_FLAG:
                import tifffile
                tifffile.imsave("debug_output.tiff", output_img)
                tifffile.imsave("debug_weight.tiff", weight_img)
                tifffile.imsave("debug_final.tiff", final_img)
            return final_img

    @classmethod
    def predict_tiles(
            cls,
            img_windows,
            frame_shape,
            model,
            batch_size,
            n_output_classes=1,
            window_overlap=None,
            tiling_mode="average",
            debug: bool = False,
            multi_gpu: bool = False
    ):
        view_shape = img_windows.shape
        if len(view_shape) == 6:
            canvas_y, canvas_x, _, window_y, window_x, channels = view_shape
            window_shape_spatial = np.array([window_y, window_x])
            window_shape = np.array([window_y, window_x, channels])
        elif len(view_shape) == 4:
            canvas_y, canvas_x, window_y, window_x = view_shape
            window_shape = np.array([window_y, window_x])
            window_shape_spatial = window_shape
        else:
            # print(view_shape)
            raise ValueError("unrecognized img_windows shape")

        if all((window_shape_spatial % 2) != 0):
            raise ValueError("the first two dimensions of window_shape should be divisible by 2")

        if window_overlap is not None:
            if len(window_overlap) != len(window_shape):
                window_overlap = np.append(window_overlap, 0)
            step = np.array(window_shape) - np.array(window_overlap)
        else:
            step = np.array(window_shape) // 2
            window_overlap = step

        cls.check_distortion_condition(frame_shape, window_shape_spatial, step)
        reshaped_windows = img_windows.reshape((-1, *window_shape))

        batched_inputs = cls.divide_into_batches(reshaped_windows, batch_size)

        out_img_shape = [*frame_shape[:2], n_output_classes]
        output_img = np.zeros(out_img_shape, dtype=np.float)
        weight_img = np.ones_like(output_img)

        weight = cls.get_weighting_window(window_shape_spatial) if tiling_mode == "weighted_average" else 1
        # print(weight.shape)

        ds = tf.data.Dataset.from_tensor_slices(reshaped_windows)
        ds = ds.batch(batch_size)

        if not multi_gpu:
            context = nullcontext
        else:
            gpus = tf.config.list_logical_devices('GPU')
            context = tf.distribute.MirroredStrategy(gpus).scope
            batch_options = tf.data.Options()
            batch_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            ds = ds.with_options(batch_options)

        with context():
            predictions = model.predict(ds).astype(np.float)

        for img_idx, pred_img in tqdm(enumerate(predictions)):
            canvas_index = np.array(np.unravel_index(img_idx, img_windows.shape[:2]))

            pivot = canvas_index * step[:2]

            if tiling_mode in ["average", "weighted_average"]:
                slice_y = slice(pivot[0], pivot[0] + window_shape[0])
                slice_x = slice(pivot[1], pivot[1] + window_shape[1])

                output_patch_shape = output_img[slice_y, slice_x].shape
                if output_patch_shape != pred_img.shape:
                    raise ValueError("incorrect sliding window shape, check padding")
                output_img[slice_y, slice_x] += pred_img
                weight_img[slice_y, slice_x] += weight

            elif tiling_mode == "drop_borders":
                assert all(
                    np.array(window_overlap[:2]) % 2 == 0), "drop_borders mode need window_overlap to be divisible by 2"
                half_overlap = np.array(window_overlap) // 2
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
        SAVE_DEBUG_TIFFS_FLAG = False
        if SAVE_DEBUG_TIFFS_FLAG:
            import tifffile
            tifffile.imsave("debug_output.tiff", output_img)
            tifffile.imsave("debug_weight.tiff", weight_img)
            tifffile.imsave("debug_final.tiff", final_img)
        return final_img

    @staticmethod
    def divide_into_batches(input_list, n):
        """divide a list in evenly sized batches of length n"""
        return [
            input_list[i * n: (i + 1) * n]
            for i in range((len(input_list) + n - 1) // n)
        ]

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
    def unpad_image(img, pad_widths):
        img_shape = img.shape

        unpadded = img[
                   pad_widths[0][0]: img_shape[0] - pad_widths[0][1],
                   pad_widths[1][0]: img_shape[1] - pad_widths[1][1]]

        return unpadded

    @staticmethod
    def unpad_volume(vol, pad_widths):
        vol_shape = vol.shape
        unpadded = vol[
                   pad_widths[0][0]: vol_shape[0] - pad_widths[0][1],
                   pad_widths[1][0]: vol_shape[1] - pad_widths[1][1],
                   pad_widths[2][0]: vol_shape[2] - pad_widths[2][1]
                   ]
        return unpadded
