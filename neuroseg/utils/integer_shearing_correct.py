import numpy as np

class IntegerShearingCorrect:
    def __init__(self,
        direction: str = 'x',
        delta: int = 1,
        inverse: bool = False):

        self.direction = direction
        self.delta = delta
        self.inverse = inverse
        if self.direction not in ['x', 'y']:
            raise ValueError('Direction must be either x or y')
        
        self.inverse = inverse
        self.delta = delta
        assert type(self.delta) is int, 'Delta must be an integer'

        
    def run(self):
        if self.inverse:
            return self.inverse_correct()
        else:
            return self.forward_correct()

    def _get_corrected_shape(self, arr_shape: tuple, inverse=False):
        # the input image shape is (z, y, x)
        inverse_factor = -1 if inverse else 1
        if self.direction == 'x':
            corrected_shape = (arr_shape[0],
                               arr_shape[1],
                               arr_shape[2] + inverse_factor * arr_shape[0] * np.abs(self.delta))

        elif self.direction == 'y':
            corrected_shape = (arr_shape[0],
                               arr_shape[1] + inverse_factor * arr_shape[0] * np.abs(self.delta),
                               arr_shape[2])
        else:
            raise ValueError('Direction of shearing must be specified')

        return corrected_shape

    def forward_correct(self, arr:np.ndarray):
        arr_shape = arr.shape
        self.corrected_shape = self._get_corrected_shape(arr_shape, inverse=False)
        corrected_arr = np.zeros(self.corrected_shape, dtype=arr.dtype)
        mask = np.zeros(self.corrected_shape, dtype=bool)
        for z in range(arr_shape[0]):
            if self.direction == 'x':
                if self.delta > 0:
                    x_start = z * self.delta
                    x_end = x_start + arr_shape[2]
                else:
                    x_end = self.corrected_shape[2] + z * self.delta
                    x_start = x_end - arr_shape[2]

                corrected_arr[z, :, x_start:x_end] = arr[z, :, :]
                mask[z, :, x_start:x_end] = True

            elif self.direction == 'y':
                if self.delta > 0:
                    y_start = z * self.delta
                    y_end = y_start + arr_shape[1]
                else:
                    y_end = self.corrected_shape[1] + z * self.delta
                    y_start = y_end - arr_shape[1]

                corrected_arr[z, y_start:y_end, :] = arr[z, :, :]
                mask[z, y_start:y_end, :] = True
            else:
                raise ValueError('Direction of shearing must be specified')
        return corrected_arr, mask

    def inverse_correct(self, arr=None):
        arr_shape = arr.shape
        self.corrected_shape = self._get_corrected_shape(arr_shape, inverse=True)
        corrected_arr = np.zeros(self.corrected_shape, dtype=arr.dtype)
        mask = np.zeros(self.corrected_shape, dtype=bool)
        for z in range(arr_shape[0]):
            if self.direction == "x":
                if self.delta > 0:
                    x_start = z * self.delta
                    x_end = x_start + self.corrected_shape[2]
                else:
                    x_end = arr_shape[2] + z * self.delta
                    x_start = x_end - self.corrected_shape[2]
                corrected_arr[z, :, :] = arr[z, :, x_start:x_end]
                mask[z, :, :] = True
            elif self.direction == "y":
                if self.delta > 0:
                    y_start = z * self.delta
                    y_end = y_start + self.corrected_shape[1]
                else:
                    y_end = arr_shape[1] + z * self.delta
                    y_start = y_end - self.corrected_shape[1]
                self.corrected_arr[z, :, :] = arr[z, y_start:y_end, :]
                mask[z, :, :] = True
            else:
                raise ValueError('Direction of shearing must be specified')
        return corrected_arr
