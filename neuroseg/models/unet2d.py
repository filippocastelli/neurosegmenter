from typing import Union, Callable
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    add,
    concatenate,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.activations import get as get_activation

from neuroseg.config import TrainConfig, PredictConfig


class Unet2D:
    def __init__(self, config: Union[TrainConfig, PredictConfig]):
        self.config = config
        self.crop_shape = config.crop_shape
        self.config = config
        self.crop_shape = config.crop_shape
        self.depth = self.config.unet_depth
        self.base_filters = self.config.base_filters
        self.batch_normalization = self.config.batch_normalization
        self.transposed_convolution = self.config.transposed_convolution
        self.n_channels = self.config.n_channels

        self.input_shape = self._get_input_shape()

        self.model = self._get_model(
            input_shape=self.input_shape,
            base_filters=self.base_filters,
            depth=self.depth,
            batch_normalization=self.batch_normalization,
            transposed_convolution=self.transposed_convolution,
        )

    def _get_input_shape(self):
        input_shape = self.crop_shape.copy()
        input_shape.append(self.n_channels)
        return input_shape

    @classmethod
    def _get_model(cls,
                   input_shape: Union[tuple, list] = (128, 128, 1),
                   base_filters: int = 16,
                   depth: int = 2,
                   batch_normalization: bool = True,
                   transposed_convolution: bool = False,
                   ):
        """Segmentation model based on UNET

        Parameters
        ----------
        input_shape : tuple
            shape of the inputs, Default (128, 128, 1)
        base_filters : int
            base filters, Default 16
        depth : int
            unet depth, Default 2
        batch_normalization : bool
            enables batchnorm, defaults to True
        transposed_convolution : bool
            use transposed convolution instead of resize-convolution, defaults to False.

        Returns
        -------
        model : keras Model
            UNCOMPILED model
        """
        # Large-view model scheme :

        #             +-----------------+        concatenation            +-----------------+      +--------------+
        # INPUTS+---->+ ENCODER BLOCK 0 +-------------------------------->+ DECODER BLOCK 0 +----->+  FINAL CONV  |
        #             +--------+--------+                                 +-----+-----------+      +------+-------+
        #                      | maxpool2d                                      ^                         |
        #                      v                                                | transposed conv         v
        #               +------+----------+      concatenation         +--------+--------+         +------+-------+
        #               | ENCODER BLOCK 1 +--------------------------->+ DECODER BLOCK 1 |         |   SIGMOID    |
        #               +--------+--------+                            +------+----------+         +------+-------+
        #                        | maxpool2d                                  ^                           |
        #                        v                                            | transposed conv           v
        #                  +-----+-----------+    concatenation      +--------+--------+                 OUT
        #                  | ENCODER BLOCK 2 +---------------------->+ DECODER BLOCK 2 |
        #                  +--------+--------+                       +--------+--------+
        #                           | maxpool2d                               ^
        #                           |            +--------------+             | transposed conv
        #                           +------------+ BRIDGE BLOCK +-------------+
        #                                        +--------------+

        # can edit this on http://asciiflow.com/

        inputs = Input(input_shape)

        # Encoder List
        encoders = []
        for d in range(depth + 1):

            # determine inputs
            if d == 0:
                enc_inputs = inputs
            else:
                enc_inputs = encoders[d - 1][0]
            pooled_enc, enc = cls._get_encoder_block(
                input_layer=enc_inputs,
                conv_filters_depths=cls._get_filter_depths(
                    base_filters=base_filters, block_depth=d
                ),
                batch_normalization=batch_normalization,
            )
            encoders.append((pooled_enc, enc))

        # Bridge
        _, bridge = cls._get_encoder_block(
            input_layer=encoders[-1][0],
            conv_filters_depths=cls._get_filter_depths(
                base_filters=base_filters, block_depth=depth + 1
            ),
            batch_normalization=batch_normalization,
        )

        # Decoder List
        decoders = [None] * (depth + 1)

        for d in reversed(range(depth + 1)):
            if d == depth:
                input_layer = bridge
            else:
                input_layer = decoders[d + 1]
            concat_layer = encoders[d][1]

            dec = cls._get_decoder_block(
                input_layer=input_layer,
                to_concatenate_layer=concat_layer,
                conv_filters_depths=cls._get_filter_depths(
                    base_filters=base_filters, block_depth=d
                ),
                transposed_convolution=transposed_convolution,
                batch_normalization=batch_normalization,
            )
            decoders[d] = dec

        # Final Conv layer
        final_conv = Conv2D(filters=1, kernel_size=(1, 1), data_format="channels_last")(
            decoders[0]
        )

        final_activation = Activation("sigmoid")(final_conv)

        model = Model(inputs=inputs, outputs=final_activation)

        return model

    @classmethod
    def _get_encoder_block(
            cls,
            input_layer,
            conv_filters_depths: Union[tuple, list],
            batch_normalization: bool = True,
            pre_activation: bool = True,
            is_first_block: bool = False,
    ):
        """
        Parameters
        ----------
        input_layer : keras layer
        conv_filters_depths : tuple or list
            number of convolutional filters to use in the two convolutional steps
        batch_normalization : bool
            enables batchnorms
        pre_activation : bool
            activates pre-activation ordering scheme
        is_first_block : bool
            if it's the first block of the net disables pre-activation on the first conv block

        Returns
        -------
        (pool_enc, enc) : tuple
            pool_enc is the (2,2,2) maxpooled version of enc

        """
        # Layer ordering is based on
        # He. et al. Deep Residual Learning for Image Recognition
        # https://arxiv.org/abs/1512.03385

        # The basic encoder block is composed as
        #       INPUT
        #         +
        #         |
        #         v
        # +-------+--------+
        # |  CONV BLOCK 0  |
        # +-------+--------+
        #         |
        #         v
        # +-------+--------+
        # |  CONV BLOCK 1  |
        # +-------+--------+
        #         |
        #         v
        # +-------+--------+
        # |  CONV BLOCK 2  |
        # +-------+--------+
        #         |
        #         v
        #        OUT1
        #         +
        #         |
        #         v
        # +-------+--------+
        # |   2D MAXPOOL   +---------> OUT2
        # +----------------+



        # First convblock
        if is_first_block:
            # If it's the first block we don't want to create a full pre-activated and batchnormalized block
            # we just need a simple convolution of the inputs.
            first_conv = Conv2D(
                conv_filters_depths[0],
                (3, 3),
                padding="same",
                strides=(1, 1),
                data_format="channels_last",
            )(input_layer)
        else:
            first_conv = cls._get_convolution_block(
                input_layer=input_layer,
                n_filters=conv_filters_depths[0],
                batch_normalization=batch_normalization,
                pre_activation=pre_activation,
            )

        # Second convblock
        out = cls._get_convolution_block(
            input_layer=first_conv,
            n_filters=conv_filters_depths[1],
            batch_normalization=batch_normalization,
            pre_activation=pre_activation,
        )

        # Residual shortcut

        # Input and output dimensionalities are not supposed to match: we do a
        # conv_filter_depts[1] 1x1x1 convolutions to adjust input and output sizes
        shortcut = Conv2D(
            conv_filters_depths[1],
            kernel_size=(1, 1),
            strides=(1, 1),
            data_format="channels_last",
        )(input_layer)
        shortcut = BatchNormalization(axis=-1)(
            shortcut
        )  # Batchnormalization of the transformed inputs
        out = add([out, shortcut])  # Adding inputs and outputs

        out_maxpooled = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(out)

        return out_maxpooled, out
