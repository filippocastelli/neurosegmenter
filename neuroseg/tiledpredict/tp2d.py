from pathlib import Path
import shutil
from itertools import product
import pickle

import numpy as np
import scipy.signal as signal
from tqdm import tqdm

from tiledpredict.datapredictorbase import DataPredictorBase


class DataPredictor2D(DataPredictorBase):
    def __init__(self, config, model=None):
        super().__init__(config, model)

    def predict(self):
        self.tiledpredictor = TiledPredictor2D(
            input_volume=self.input_data,
            batch_size=self.batch_size,
            window_size=self.window_size,
            model=self.prediction_model,
            padding_mode=self.padding_mode,
            chunk_size=self.chunk_size,
            tmp_folder=self.temp_path,
            keep_tmp=self.keep_tmp,
            n_output_classes=self.n_output_classes,
        )

        self.predicted_data = self.tiledpredictor.output_volume
        return self.predicted_data


class MultiVolumeDataPredictor2D(DataPredictorBase):
    def __init__(self, config, model=None):
        super().__init__(config, model)

    def predict(self):
        tiled_predictors = {}
        for idx, volume in enumerate(self.input_data):
            
            volume_name = self.data_paths[idx].name
            
            tiled_predictor = TiledPredictor2D(
                    input_volume=volume,
                    batch_size=self.batch_size,
                    window_size=self.window_size,
                    model=self.prediction_model,
                    padding_mode=self.padding_mode,
                    chunk_size=self.chunk_size,
                    tmp_folder=self.temp_path,
                    keep_tmp=self.keep_tmp,
                    n_output_classes=self.n_output_classes,
                )
            
            tiled_predictors[volume_name] = tiled_predictor
            
        self.predicted_data = {name: pred.output_volume for name, pred in tiled_predictors.items()}
        # self.predicted_data = [tiledpredictor.output_volume for tiledpredictor in tiled_predictors]
        return self.predicted_data


class TiledPredictor2D:
    def __init__(
        self,
        input_volume,
        batch_size=5,
        window_size=(128, 128),
        model=None,
        padding_mode="reflect",
        chunk_size=100,
        tmp_folder="tmp",
        keep_tmp=False,
        n_output_classes=1,
    ):

        self.input_volume = input_volume
        self.batch_size = batch_size
        self.window_size = window_size
        self.model = model
        self.padding_mode = padding_mode
        self.n_output_classes = n_output_classes

        self.apply_padding(self.padding_mode)
        self.pivots = self.get_pivots(self.padded_volume, self.window_size)
        self.chunk_size = chunk_size
        self.tmp_folder = Path(tmp_folder)
        # TMP
        # self.inference_volume = np.zeros_like(self.padded_volume).astype("float32")
        self.inference_volume = np.zeros(shape=self._get_spatial_dims()).astype(
            "float32"
        )
        self.weighting_window = self.get_weighting_window(self.window_size)
        self.generate_batches()
        self.generate_chunks()
        self.predict()
        self.remove_padding()

        if not keep_tmp:
            self.empty_tmp()

    def _get_spatial_dims(self):
        return np.append(self.padded_volume.shape[:-1], (self.n_output_classes))

    @staticmethod
    def get_paddings(input_volume, window_size):
        """
        calculate the the padding vectors for the volume to be integer divisible
        by the window size.
        The padding is calculated not only to fit an entire number of windows
        but also takes into consideration an extra window.
        The padding is evenly distributed at the start and at the end of the 
        volume dims. [jeez cant write in english today]

        Parameters
        ----------
        input_volume : np.ndarray
            Input volume.
        window_size : list or tuple
            shape of the window.

        Returns
        -------
        paddings : np.ndarray
            padding matrix compatible with np.pad().
            
            [0,0]
            [left_y, right_y]
            [left_x, right_x]
            [0,0]

        """
        window_size_array = np.array(window_size)
        spatial_dims = input_volume.shape[1:-1]

        tot_paddings = np.zeros_like(spatial_dims)
        window_mod = window_size_array - (spatial_dims % window_size_array)

        tot_paddings = window_mod + window_size_array

        half_paddings = tot_paddings // 2

        mod_half_paddings = tot_paddings % 2

        paddings_left = half_paddings
        paddings_right = half_paddings + mod_half_paddings
        spatial_paddings = np.stack([paddings_left, paddings_right]).T
        zeros = np.expand_dims(np.array([0, 0]), axis=-1).T
        paddings = np.concatenate([zeros, spatial_paddings, zeros], axis=0)

        return paddings

    def apply_padding(self, padding_mode="reflect"):
        """ apply the paddings"""
        self.paddings = self.get_paddings(self.input_volume, self.window_size)
        self.padded_volume = np.pad(self.input_volume, self.paddings, mode="reflect")

    def remove_padding(self):
        """removes padding from the image to restore initial size"""
        paddings = self.paddings
        self.output_volume = self.inference_volume[
            :, paddings[1][0] : -paddings[1][1], paddings[2][0] : -paddings[2][1], :
        ]

    @staticmethod
    def list_chunks(input_list, n):
        """divide a list in evenly sized chunks of length n"""
        return [
            input_list[i * n : (i + 1) * n]
            for i in range((len(input_list) + n - 1) // n)
        ]

    @staticmethod
    def get_pivots(padded_volume, window_size):
        """
        calculate pivot points

        Parameters
        ----------
        padded_volume : np.ndarray
            input volume, already padded.
        window_size : tuple or list
            shape of the window.

        Returns
        -------
        pivots : list
            list of [z,y,x] pivot coordinatess.

        """
        padded_shape = padded_volume.shape

        window_size_array = np.array(window_size)
        spatial_dimensions_array = np.array(padded_shape[1:-1])

        steps_y, steps_x = spatial_dimensions_array // (window_size_array // 2)

        z_grid = np.array(range(padded_shape[0]))
        y_grid = np.array(range(steps_y - 1)) * window_size_array[0] // 2
        x_grid = np.array(range(steps_x - 1)) * window_size_array[1] // 2

        pivots = np.array(list(product(z_grid, y_grid, x_grid)))

        return pivots

    @classmethod
    def batch_generator(cls, padded_volume, pivot_list, window_size, batch_size):
        """
        generator for model inputs
        returns batches batch_size long of (pivot, window) tuples

        Parameters
        ----------
        padded_volume : ndarray
            input volume, needs to be already padded.
        pivot_list : list
            list of pivot indices.
        window_size : tuple
            shape of input window in (y,x) format.
        batch_size : int
            batchsize.

        Yields
        ------
        out_batch : list
            list of (pivot, window) tuples.

        """
        pivot_batches = cls.list_chunks(pivot_list, batch_size)

        for idx, pivot_batch in enumerate(pivot_batches):
            batch_list = []
            for idx, pivot in enumerate(pivot_batch):
                z, y, x = pivot
                extracted_window = padded_volume[
                    z, y : y + window_size[0], x : x + window_size[1], ...
                ]
                batch_list.append((pivot, extracted_window))
            yield batch_list

    def generate_batches(self):
        """generate input_windows"""

        self.batchgen = self.batch_generator(
            self.padded_volume, self.pivots, self.window_size, self.batch_size
        )

    def generate_chunks(self):
        self.batch_iterator = self.batchgen.__iter__()

        self.batch_list_pivots = self.list_chunks(self.pivots, self.batch_size)
        self.chunk_list_pivots = self.list_chunks(
            self.batch_list_pivots, self.chunk_size
        )

        self.tmp_folder.mkdir(exist_ok=True, parents=True)

        chunk_fpaths = []

        print("Saving chunks...")
        for chunk_idx, chunk_pivots in enumerate(tqdm(self.chunk_list_pivots)):
            batch_list = []
            for batch_idx, batch_pivots in enumerate(chunk_pivots):
                batch_list.append(next(self.batch_iterator))

            chunk_fpath = self.tmp_folder.joinpath("chunk_{}.pickle".format(chunk_idx))
            chunk_fpaths.append(chunk_fpath)

            with chunk_fpath.open(mode="wb") as wfile:
                pickle.dump(batch_list, wfile)

        self.chunk_fpath_list = chunk_fpaths

    def predict(self):
        for idx, chunk_fpath in enumerate(self.chunk_fpath_list):
            print(
                "inferencing chunk {} / {}".format(idx + 1, len(self.chunk_fpath_list))
            )
            chunk_pivots, chunk_inputs = self.chunk_path_to_inputs(chunk_fpath)
            chunk_predictions = self.model.predict(chunk_inputs)

            weighted_predictions = self.weight_chunk(chunk_predictions)

            for i, patch in enumerate(weighted_predictions):
                pivot = chunk_pivots[i]
                z, y, x = pivot
                self.inference_volume[
                    z, y : y + self.window_size[0], x : x + self.window_size[1], ...
                ] += patch
            # distribute to predictions

    def weight_chunk(self, prediction_chunk):
        return prediction_chunk * self.weighting_window

    @staticmethod
    def chunk_path_to_inputs(chunk_fpath):
        """
        Load a chunk of data from disk, return the list of its pivots
        and the corresponding [batch, z, y, x, c] np.array tensor

        Parameters
        ----------
        chunk_fpath : pathlib Path
            path of the chunk.

        Returns
        -------
        tuple
            (pivot_list, chunk_tensor)

        """
        with chunk_fpath.open(mode="rb") as readfile:
            chunk = pickle.load(readfile)
        list_pivots = []
        list_inputs = []

        for batch in chunk:
            for input_point in batch:
                pivot, input_window = input_point
                list_pivots.append(pivot)
                list_inputs.append(input_window)

        return list_pivots, np.array(list_inputs)

    @classmethod
    def get_weighting_window(cls, window_size, power=2, expand_dims=True):
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
        wind_outer = (abs(2 * (signal.triang(window_linear_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (signal.triang(window_linear_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind

    def empty_tmp(self):
        path = self.tmp_folder
        shutil.rmtree(path)


# if __name__ == "__main__":
#     test_path = Path("tests/res/test_stack.tif")
#     input_volume = load_volume(test_path, expand_dims=True)
#     input_volume = input_volume[:, :-1, :-1, :]

#     window_size = (64, 64)

#     class fakeModel:
#         def predict(self, inputs):
#             return inputs

#     mockmodel = fakeModel()

#     tpred = TiledPredictor2D(
#         input_volume, window_size=window_size, batch_size=5, model=mockmodel
#     )
