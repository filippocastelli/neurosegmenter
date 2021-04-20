# from pathlib import Path
# from itertools import product
# import pickle
# import shutil

import numpy as np
from tqdm import tqdm
import scipy.signal as signal
from skimage.util import view_as_windows, pad

from neuroseg.tiledpredict.datapredictorbase import DataPredictorBase

class TiledPredictor3D:
    def __init__(
        self,
        input_volume,
        batch_size=5,
        chunk_size=100,
        window_size=(64,64,64),
        n_output_classes=1,
        model=None,
        padding_mode="reflect",
        tmp_folder="tmp",
        keep_tmp=False,
    ):

        self.input_volume = input_volume
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.crop_shape = np.array(window_size)
        self.padding_mode = padding_mode
        self.model = model
        # self.tmp_folder = Path(tmp_folder)
        # self.keep_tmp = keep_tmp
        self.n_output_classes = n_output_classes
        # self.inference_volume = np.zeros()

        # asserting divisibility by 2
        if not (self.crop_shape % 2 == 0).all():
            raise ValueError("crop shape must be divisible by 2 along all dims")

        # calculating steps
        self.step = self.crop_shape // 2
        self.input_volume_shape = self.input_volume.shape
        
        # if len(self.input_volume_shape) == 4:
        #     import pudb
        #     pudb.set_trace()

        self.paddings = self.get_paddings(self.input_volume.shape, self.crop_shape)
        self.padded_volume = self.pad_image(
            self.input_volume, self.paddings, mode=self.padding_mode
        )
        self.padded_volume_shape = self.padded_volume.shape
        
        
        self.patch_window_view = self.get_patch_windows(
            img=self.padded_volume, crop_shape=self.crop_shape, step=self.step
        )

    def predict(self):
        self.prediction_volume_padded = self.predict_tiles(img_windows=self.patch_window_view,
                                       frame_shape=self.padded_volume_shape,
                                       model=self.model,
                                       batch_size=self.batch_size,
                                       n_output_classes=self.n_output_classes,
                                       chunk_size=None,
                                       weight_window=True)
        
        self.prediction_volume = self.unpad_image(self.prediction_volume_padded, self.paddings)
        return self.prediction_volume

    @staticmethod
    def get_paddings(image_shape, crop_shape):
        """given image_shape and crop_shape get [(pad_left, pad_right)] paddings"""

        image_shape = np.array(image_shape)
        crop_shape = np.array(crop_shape)
        
        if not (crop_shape % 2 == 0).all():
            raise ValueError("crop_shape should be divisible by 2")
        
        image_shape_spatial = image_shape[:3]
        pad_list = [(0, 0) for idx in range(len(image_shape_spatial))]

        # mod = np.array(image_shape_spatial) % np.array(crop_shape)
        step = crop_shape // 2
        mod = (image_shape_spatial - crop_shape) % step
        
        nonzero_idxs = np.nonzero(mod)[0]
        
        if len(nonzero_idxs) > 0:
            for nonzero_idx in nonzero_idxs:
                idx = int(nonzero_idx)
                total_pad = (step - mod)[idx]
                left_pad = total_pad % 2
                right_pad = total_pad - left_pad

                pad_list[idx] = (left_pad, right_pad)

        return pad_list

    @staticmethod
    def pad_image(img, pad_widths, mode="constant"):
        img_shape = img.shape
        pad_width_list = list(pad_widths)
        """performing the padding"""
        # adding dims
        if len(pad_width_list) < len(img_shape):
            while len(pad_width_list) != len(img_shape):
                pad_width_list.append((0, 0))
            
        img = pad(img, pad_width_list, mode=mode)
        return img

    @classmethod
    def get_patch_windows(cls, img, crop_shape, step):
        crop_shape = list(crop_shape)

        if isinstance(step, int):
            step = [step, step, step]
        else:
            step = list(step)

        img_spatial_shape = img.shape[:3]
        cls.check_distortion_condition(img_spatial_shape, crop_shape, step)

        img_chans = None if len(img.shape) == 3 else img.shape[3]
        window_shape = crop_shape

        if img_chans is not None:
            window_shape.append(img_chans)
            step.append(1)

        view = view_as_windows(arr_in=img, window_shape=window_shape, step=step)
        return view
    
    @staticmethod
    def check_distortion_condition(frame_shape, crop_shape, step):
        frame_shape = frame_shape[:3]
        crop_shape = crop_shape[:3]
        step = step[:3]
        mod = (np.array(frame_shape) - np.array(crop_shape)) % step
        if not (mod == 0).all():
            raise ValueError(
                "(img_shape - crop_shape) % step must be zeros to avoid reconstruction distorsions"
            )
            
    @classmethod
    def predict_tiles(
        cls,
        img_windows,
        frame_shape,
        model,
        batch_size,
        n_output_classes=1,
        chunk_size=None,
        weight_window=True
    ):
        view_shape = img_windows.shape
        if len(view_shape) == 8:
            canvas_z, canvas_y, canvas_x, _, window_z, window_y, window_x, channels = view_shape
            window_shape_spatial = np.array([window_z, window_y, window_x])
            window_shape = np.array([window_z, window_y, window_x, channels])
        elif len(view_shape) == 6:
            canvas_z, canvas_y, canvas_x, window_z, window_y, window_x = view_shape
            window_shape = np.array([window_z, window_y, window_x])
            window_shape_spatial = window_shape
        else:
            # print(view_shape)
            raise ValueError("unrecognized img_windows shape")
            
        if all((window_shape_spatial % 2) != 0):
            raise ValueError("the first two dimensions of window_shape should be divisible by 2")
            
        step = np.array(window_shape) // 2
        
        cls.check_distortion_condition(frame_shape, window_shape_spatial, step)
        reshaped_windows = img_windows.reshape((-1, *window_shape))
        
        batched_inputs = cls.divide_into_batches(reshaped_windows, batch_size)
        
        out_img_shape = [*frame_shape[:3],1]
        output_img = np.zeros(out_img_shape, dtype=np.float32)
        weight_img = np.zeros_like(output_img)
        
        weight = cls.get_weighting_window(window_shape_spatial) if weight_window else 1
        # print(weight.shape)
        for batch_idx, batch in enumerate(tqdm(batched_inputs)):
            batch_global_index = int(batch_idx)*batch_size
            predicted_batch = model.predict(batch)
            
            for img_idx, pred_img in enumerate(predicted_batch):
                tile_idx = img_idx + batch_global_index
                canvas_index = np.array(np.unravel_index(tile_idx, img_windows.shape[:3]))
                
                pivot = canvas_index * step[:3]

                output_img[
                    pivot[0]:pivot[0]+window_shape[0],
                    pivot[1]:pivot[1]+window_shape[1],
                    pivot[2]:pivot[2]+window_shape[2],
                    ] += pred_img

                weight_img[
                    pivot[0]:pivot[0]+window_shape[0],
                    pivot[1]:pivot[1]+window_shape[1],
                    pivot[2]:pivot[2]+window_shape[2]
                    ] += weight
            
        final_img = output_img / weight_img
        return final_img

    @staticmethod
    def divide_into_batches(input_list, n):
        """divide a list in evenly sized batches of length n"""
        return [
            input_list[i * n : (i + 1) * n]
            for i in range((len(input_list) + n - 1) // n)
        ]
    
    @classmethod
    def get_weighting_window(cls, window_size, power=2, expand_dims=True):
        """Generate a 3D spline-based weighting window"""
        wind_z = cls.spline_window(window_size[0], power=2)
        wind_y = cls.spline_window(window_size[1], power=2)
        wind_x = cls.spline_window(window_size[2], power=2)

        wind_z = np.expand_dims(wind_z, axis=-1)
        wind_y = np.expand_dims(wind_y, axis=-1)
        wind_x = np.expand_dims(wind_x, axis=-1)

        wind = wind_y.transpose() * wind_z
        wind = np.expand_dims(wind, axis=-1)
        wind = wind * wind_x.T

        wind = wind / wind.max()

        if expand_dims:
            wind = np.expand_dims(wind, axis=-1)

        return wind.astype("float32")

    @staticmethod
    def spline_window(window_linear_size, power=2):
        """ generates 1D spline window profile"""
        intersection = int(window_linear_size / 4)
        wind_outer = (abs(2 * (signal.triang(window_linear_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (signal.triang(window_linear_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind
    
    @staticmethod
    def unpad_image(img, pad_widths):

        unpadded = img
        img_shape = img.shape

        unpadded = img[
            pad_widths[0][0]: img_shape[0]-pad_widths[0][1],
            pad_widths[1][0]: img_shape[1]-pad_widths[1][1],
            pad_widths[2][0]: img_shape[2]-pad_widths[2][1],
            ]
        
        return unpadded
class DataPredictor3D(DataPredictorBase):
    def __init__(self, config, model=None):
        super().__init__(config, model)

    def predict(self):
        self.tiledpredictor = TiledPredictor3D(
            input_volume=self.input_data,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            window_size=self.window_size,
            n_output_classes=self.n_output_classes,
            model=self.prediction_model,
            padding_mode=self.padding_mode,
            tmp_folder=self.temp_path,
            keep_tmp=self.keep_tmp,

        )

        self.predicted_data = self.tiledpredictor.predict()
        return self.predicted_data


class MultiVolumeDataPredictor3D(DataPredictorBase):
    def __init__(self, config, model=None):
        super().__init__(config, model)

    def predict(self):
        tiled_predictors = {}
        for idx, volume in enumerate(self.input_data):
            volume_name = self.data_paths[idx].name
            tiledpredictor = TiledPredictor3D(
                input_volume=volume,
                batch_size=self.batch_size,
                chunk_size=self.chunk_size,
                n_output_classes=self.n_output_classes, 
                window_size=self.window_size,
                model=self.prediction_model,
                padding_mode=self.padding_mode,
                tmp_folder=self.temp_path,
                keep_tmp=self.keep_tmp,
            )

            tiled_predictors[volume_name] = tiledpredictor

        self.predicted_data = {
            name: pred.output_volume for name, pred in tiled_predictors.items()
        }
        return self.predicted_data


# class TiledPredictor3Dold:
#     def __init__(
#         self,
#         input_volume,
#         batch_size=5,
#         window_size=(64, 64, 64),
#         model=None,
#         padding_mode="reflect",
#         chunk_size=100,
#         tmp_folder="tmp",
#         keep_tmp=False,
#         n_output_classes=1,
#     ):

#         self.input_volume = input_volume
#         self._adjust_input_dims()

#         self.batch_size = batch_size
#         self.window_size = window_size
#         self.model = model
#         self.padding_mode = padding_mode
#         self.n_output_classes = n_output_classes

#         self.apply_padding(self.padding_mode)
#         self.pivots = self.get_pivots(self.padded_volume, self.window_size)
#         self.chunk_size = chunk_size
#         self.tmp_folder = Path(tmp_folder)
#         # TMP
#         # self.inference_volume = np.zeros_like(self.padded_volume).astype("float32")
#         self.inference_volume = np.zeros(shape=self._get_padded_output_dims()).astype(
#             "float32"
#         )
#         self.weighting_window = self.get_weighting_window(self.window_size)

#         self.generate_batches()
#         self.generate_chunks()
#         self.predict()
#         self.remove_padding()

#         if not keep_tmp:
#             self.empty_tmp()

#     def _adjust_input_dims(self):
#         volume_shape = self.input_volume.shape
#         if len(volume_shape) < 3:
#             # not enough dims
#             raise ValueError(volume_shape, "Incorrect input volume dims")
#         elif len(volume_shape) == 3:
#             # if 3 channels then all dims are spatial
#             # tensor must be expanded for color
#             self.input_volume = np.expand_dims(self.input_volume, axis=-1)

#         elif len(volume_shape) == 4:
#             # if 4 dims one is color
#             # should be channel_last
#             pass
#         else:
#             # too many dims, can't figure out what they are
#             raise ValueError(volume_shape, "Incorrect input volume dims")

#         assert len(self.input_volume.shape) == 4, "Incorrect volume adjust"

#     def _get_spatial_dims(self):
#         return self.input_volume.shape[:-1]

#     def _get_output_dims(self):
#         return np.append(self._get_spatial_dims(), (self.n_output_classes))

#     def _get_padded_output_dims(self):
#         padded_spatial_dims = self.padded_volume.shape[:-1]
#         return np.append(padded_spatial_dims, (self.n_output_classes))

#     @staticmethod
#     def get_paddings(input_volume, window_size, spatial_dims):
#         """
#         calculate the the padding vectors for the volume to be integer divisible
#         by the window size.
#         The padding is calculated not only to fit an entire number of windows
#         but also takes into consideration an extra window.
#         The padding is evenly distributed at the start and at the end of the 
#         volume dims. [jeez cant write in english today]
#         Parameters
#         ----------
#         input_volume : np.ndarray
#             Input volume.
#         window_size : list or tuple
#             shape of the window.
#         spatial_dims : list or tuple or np.ndarray
#             spatial dims of the Input Volume
#         Returns
#         -------
#         paddings : np.ndarray
#             padding matrix compatible with np.pad().
            
#             [left_z, right_z]
#             [left_y, right_y]
#             [left_x, right_x]
#             [0,0]
#         """
#         window_size_array = np.array(window_size)
#         spatial_dims = spatial_dims

#         tot_paddings = np.zeros_like(spatial_dims)
#         window_mod = window_size_array - (spatial_dims % window_size_array)

#         tot_paddings = window_mod + window_size_array

#         half_paddings = tot_paddings // 2

#         mod_half_paddings = tot_paddings % 2

#         paddings_left = half_paddings
#         paddings_right = half_paddings + mod_half_paddings
#         spatial_paddings = np.stack([paddings_left, paddings_right]).T
#         zeros = np.expand_dims(np.array([0, 0]), axis=-1).T
#         paddings = np.concatenate([spatial_paddings, zeros], axis=0)

#         return paddings

#     def apply_padding(self, padding_mode="reflect"):
#         """ apply the paddings"""
#         self.paddings = self.get_paddings(
#             self.input_volume, self.window_size, self._get_spatial_dims()
#         )
#         self.padded_volume = np.pad(self.input_volume, self.paddings, mode="reflect")

#     def remove_padding(self):
#         """removes padding from the image to restore initial size"""
#         paddings = self.paddings
#         self.output_volume = self.inference_volume[
#             paddings[0][0] : -paddings[0][1],
#             paddings[1][0] : -paddings[1][1],
#             paddings[2][0] : -paddings[2][1],
#             :,
#         ]

#     @staticmethod
#     def list_chunks(input_list, n):
#         """divide a list in evenly sized chunks of length n"""
#         return [
#             input_list[i * n : (i + 1) * n]
#             for i in range((len(input_list) + n - 1) // n)
#         ]

#     @staticmethod
#     def get_pivots(padded_volume, window_size):
#         """
#         calculate pivot points
#         Parameters
#         ----------
#         padded_volume : np.ndarray
#             input volume, already padded.
#         window_size : tuple or list
#             shape of the window.
#         Returns
#         -------
#         pivots : list
#             list of [z,y,x] pivot coordinatess.
#         """
#         padded_shape = padded_volume.shape

#         window_size_array = np.array(window_size)
#         spatial_dimensions_array = np.array(padded_shape[:-1])

#         steps_z, steps_y, steps_x = spatial_dimensions_array // (window_size_array // 2)

#         z_grid = np.array(range(steps_z - 1)) * window_size_array[0] // 2
#         y_grid = np.array(range(steps_y - 1)) * window_size_array[1] // 2
#         x_grid = np.array(range(steps_x - 1)) * window_size_array[2] // 2

#         pivots = np.array(list(product(z_grid, y_grid, x_grid)))

#         return pivots

#     @classmethod
#     def batch_generator(cls, padded_volume, pivot_list, window_size, batch_size):
#         """
#         generator for model inputs
#         returns batches batch_size long of (pivot, window) tuples
#         Parameters
#         ----------
#         padded_volume : ndarray
#             input volume, needs to be already padded.
#         pivot_list : list
#             list of pivot indices.
#         window_size : tuple
#             shape of input window in (z,y,x) format.
#         batch_size : int
#             batchsize.
#         Yields
#         ------
#         out_batch : list
#             list of (pivot, window) tuples.
#         """
#         pivot_batches = cls.list_chunks(pivot_list, batch_size)

#         for idx, pivot_batch in enumerate(pivot_batches):
#             batch_list = []
#             for idx, pivot in enumerate(pivot_batch):
#                 z, y, x = pivot
#                 extracted_window = padded_volume[
#                     z : z + window_size[0],
#                     y : y + window_size[1],
#                     x : x + window_size[2],
#                     :,
#                 ]
#                 batch_list.append((pivot, extracted_window))
#             yield batch_list

#     def generate_batches(self):
#         """generate input_windows"""

#         self.batchgen = self.batch_generator(
#             self.padded_volume, self.pivots, self.window_size, self.batch_size
#         )

#     def generate_chunks(self):
#         self.batch_iterator = self.batchgen.__iter__()

#         self.batch_list_pivots = self.list_chunks(self.pivots, self.batch_size)
#         self.chunk_list_pivots = self.list_chunks(
#             self.batch_list_pivots, self.chunk_size
#         )

#         self.tmp_folder.mkdir(exist_ok=True, parents=True)

#         chunk_fpaths = []

#         print("Saving chunks...")
#         for chunk_idx, chunk_pivots in enumerate(tqdm(self.chunk_list_pivots)):
#             batch_list = []
#             for batch_idx, batch_pivots in enumerate(chunk_pivots):
#                 batch_list.append(next(self.batch_iterator))

#             chunk_fpath = self.tmp_folder.joinpath("chunk_{}.pickle".format(chunk_idx))
#             chunk_fpaths.append(chunk_fpath)

#             with chunk_fpath.open(mode="wb") as wfile:
#                 pickle.dump(batch_list, wfile)

#         self.chunk_fpath_list = chunk_fpaths

#     def predict(self):
#         for idx, chunk_fpath in enumerate(self.chunk_fpath_list):
#             print("inferencing chunk {} / {}".format(idx, len(self.chunk_fpath_list)))
#             chunk_pivots, chunk_inputs = self.chunk_path_to_inputs(chunk_fpath)
#             chunk_predictions = self.model.predict(chunk_inputs)

#             weighted_predictions = self.weight_chunk(chunk_predictions)

#             for i, patch in enumerate(weighted_predictions):
#                 pivot = chunk_pivots[i]
#                 z, y, x = pivot
#                 try:
#                     self.inference_volume[
#                         z : z + self.window_size[0],
#                         y : y + self.window_size[1],
#                         x : x + self.window_size[2],
#                         :,
#                     ] += patch
#                 except ValueError:
#                     raise ValueError(
#                         "Incompatible shapes between sliced inference volume and patch, check the padding routines for errors."
#                     )

#             # distribute to predictions

#     def weight_chunk(self, prediction_chunk):
#         return prediction_chunk * self.weighting_window

#     @staticmethod
#     def chunk_path_to_inputs(chunk_fpath):
#         """
#         Load a chunk of data from disk, return the list of its pivots
#         and the corresponding [batch, z, y, x, c] np.array tensor
#         Parameters
#         ----------
#         chunk_fpath : pathlib Path
#             path of the chunk.
#         Returns
#         -------
#         tuple
#             (pivot_list, chunk_tensor)
#         """
#         with chunk_fpath.open(mode="rb") as readfile:
#             chunk = pickle.load(readfile)
#         list_pivots = []
#         list_inputs = []

#         for batch in chunk:
#             for input_point in batch:
#                 pivot, input_window = input_point
#                 list_pivots.append(pivot)
#                 list_inputs.append(input_window)

#         return list_pivots, np.array(list_inputs)

#     @classmethod
#     def get_weighting_window(cls, window_size, power=2, expand_dims=True):
#         """Generate a 2D spline-based weighting window"""
#         wind_z = cls.spline_window(window_size[0], power=2)
#         wind_y = cls.spline_window(window_size[1], power=2)
#         wind_x = cls.spline_window(window_size[2], power=2)

#         wind_z = np.expand_dims(wind_z, axis=-1)
#         wind_y = np.expand_dims(wind_y, axis=-1)
#         wind_x = np.expand_dims(wind_x, axis=-1)

#         wind = wind_y.transpose() * wind_z
#         wind = np.expand_dims(wind, axis=-1)
#         wind = wind * wind_x.T

#         wind = wind / wind.max()

#         if expand_dims:
#             wind = np.expand_dims(wind, axis=-1)

#         return wind.astype("float32")

#     @staticmethod
#     def spline_window(window_linear_size, power=2):
#         """ generates 1D spline window profile"""
#         intersection = int(window_linear_size / 4)
#         wind_outer = (abs(2 * (signal.triang(window_linear_size))) ** power) / 2
#         wind_outer[intersection:-intersection] = 0

#         wind_inner = 1 - (abs(2 * (signal.triang(window_linear_size) - 1)) ** power) / 2
#         wind_inner[:intersection] = 0
#         wind_inner[-intersection:] = 0

#         wind = wind_inner + wind_outer
#         wind = wind / np.average(wind)
#         return wind

#     def empty_tmp(self):
#         path = self.tmp_folder
#         shutil.rmtree(path)
