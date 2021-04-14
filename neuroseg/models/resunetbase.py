class ResUNETBase:
    def __init__(self, config):
        self.config = config
        self.crop_shape = config.crop_shape
        self.depth = self.config.unet_depth
        self.base_filters = self.config.base_filters
        self.batch_normalization = self.config.batch_normalization
        self.pre_activation = self.config.residual_preactivation
        self.transposed_convolution = self.config.transposed_convolution
        self.n_channels = self.config.n_channels

        self.input_shape = self._get_input_shape()
        self.model = self._get_model(
            input_shape=self.input_shape,
            base_filters=self.base_filters,
            depth=self.depth,
            batch_normalization=self.batch_normalization,
            pre_activation=self.pre_activation,
            transposed_convolution=self.transposed_convolution,
        )

    def _get_input_shape(self):
        input_shape = self.crop_shape.copy()
        input_shape.append(self.n_channels)
        return input_shape
