# =============================================================================
# MODEL
#
# 3D Residual UNET model definition
#
# Filippo Maria Castelli
# castelli@lens.unifi.it
# =============================================================================


from tensorflow.keras.layers import (
    Input,
    Conv3D,
    Conv3DTranspose,
    UpSampling3D,
    MaxPooling3D,
    BatchNormalization,
    Activation,
    add,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.activations import get as get_activation

from neuroseg.models import ResUNETBase


class ResUNET3D(ResUNETBase):
    @classmethod
    def _get_model(
        cls,
        input_shape=(64, 64, 64, 1),
        base_filters=16,
        depth=2,
        batch_normalization=True,
        transposed_convolution=False,
        pre_activation=True,
    ):
        """
    resUnet3D
      Residual UNET3D Model
    
        Parameters
        ----------
        input_shape : tuple
            shape of the inputs, Default (64, 64, 64, 1)
        base_filters : int
            base filters, Default 16
        batch_normalization : bool
            enables batchnorm
        pre_activation : bool
            enables pre-activation ordering scheme
    
        Returns
        -------
        model : keras Model
            UNCOMPILED model
    
        """
        # The model is loosely based on
        # Cicek et al. "3D U-Net: Learning Dense Volumetric Segmentation for Sparse Annotation"
        # https://arxiv.org/abs/1606.06650
        #
        # Large-view model scheme :

        #             +-----------------+        concatenation            +-----------------+      +--------------+
        # INPUTS+---->+ ENCODER BLOCK 0 +-------------------------------->+ DECODER BLOCK 0 +----->+  FINAL CONV  |
        #             +--------+--------+                                 +-----+-----------+      +------+-------+
        #                      | maxpool3d                                      ^                         |
        #                      v                                                | transposed conv         v
        #               +------+----------+      concatenation         +--------+--------+         +------+-------+
        #               | ENCODER BLOCK 1 +--------------------------->+ DECODER BLOCK 1 |         |   SIGMOID    |
        #               +--------+--------+                            +------+----------+         +------+-------+
        #                        | maxpool3d                                  ^                           |
        #                        v                                            | transposed conv           v
        #                  +-----+-----------+    concatenation      +--------+--------+                 OUT
        #                  | ENCODER BLOCK 2 +---------------------->+ DECODER BLOCK 2 |
        #                  +--------+--------+                       +--------+--------+
        #                           | maxpool3d                               ^
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
                pre_activation=pre_activation,
            )
            encoders.append((pooled_enc, enc))

        # Bridge
        _, bridge = cls._get_encoder_block(
            input_layer=encoders[-1][0],
            conv_filters_depths=cls._get_filter_depths(
                base_filters=base_filters, block_depth=depth + 1
            ),
            batch_normalization=batch_normalization,
            pre_activation=pre_activation,
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
                pre_activation=pre_activation,
            )
            decoders[d] = dec

        # Final Conv layer
        final_conv = Conv3D(
            filters=1, kernel_size=(1, 1, 1), data_format="channels_last"
        )(decoders[0])

        final_activation = Activation("sigmoid")(final_conv)

        model = Model(inputs=inputs, outputs=final_activation)

        return model

    # Layer block definitions
    @classmethod
    def _get_convolution_block(
        cls,
        input_layer,
        n_filters,
        activation="relu",
        batch_normalization=True,
        pre_activation=True,
    ):
        """
        create_convolution_block
    
        Parameters
        ----------
        input_layer :  keras_layer
        n_filters : int
            number of convolutional filters
        activation : string or callable
            activation function, can be the name of a standard Keras activation (see https://keras.io/activations/)
            or a custom callable
        batch_normalization : bool
            enables batch normalization
        pre_activation : bool
            enables pre-activation ordering scheme
    
        Returns
        -------
        convolution_block
        """
        # Layer definitions
        conv = Conv3D(
            filters=n_filters,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            data_format="channels_last",  # Make sure data is in "channel_last" format
        )
        batchnorm_layer = BatchNormalization(axis=-1)
        if activation:
            # tf.keras.activations.get (here aliased as get_activation() ), is an undocumented Keras method:
            # it basically accepts an identifier and returns an activation method
            # if the identifier is None it returns a linear activation function,
            # if it's a string it returns the corresponding Keras activation function,
            # if it's a callable it just returns the callable itself
            # allowing for both standard activation method use and custom activation function definition)
            #
            # See original implementation
            # https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/keras/activations.py#L310-L325

            activation_layer = Activation(get_activation(activation))

        # Assembling layers together

        # Layer ordering follows a residual "full pre-activation" scheme
        # see He et. al. "Identity Mappings in Deep Residual Networks" pag. 5
        # https://arxiv.org/pdf/1603.05027.pdf

        # Base convolution unit is composed as

        #               +-----------------------+
        # INPUTS +----->+  BATCH NORMALIZATION  |
        #               +-----------+-----------+
        #                           |
        #                           v
        #               +-----------+-----------+
        #               |  ACTIVATION FUNCTION  |
        #               +-----------+-----------+
        #                           |
        #                           v
        #               +-----------+-----------+
        #               |        CONV3D         +------> OUT
        #               +-----------------------+

        layer_list = []

        if batch_normalization:
            layer_list.append(batchnorm_layer)

        if activation:
            layer_list.append(activation_layer)

        if pre_activation:
            layer_list.append(conv)
        else:
            # This configuration corresponds to the "standard" non-residual activation scheme, its use is generally
            # discouraged in residual block because of representational problems, but the option is there.
            # (ref. He. et al.)
            layer_list.insert(0, conv)

        layer_out = cls._combine_layers(input_layer, layer_list)

        return layer_out

    @classmethod
    def _get_encoder_block(
        cls,
        input_layer,
        conv_filters_depths,
        batch_normalization=True,
        pre_activation=True,
        is_first_block=False,
    ):
        """
    encoder_block
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

        #       INPUT  +----------------+
        #         +                     |
        #         |                     |
        #         v                     v
        # +-------+--------+   +--------+----------+
        # |  CONV BLOCK 0  |   | 1X1X1 CONVOLUTION |
        # +-------+--------+   +--------+----------+
        #         |                     |
        #         v                     v
        # +-------+--------+   +--------+----------+
        # |  CONV BLOCK 1  |   |    BATCHNORM      |
        # +-------+--------+   +---------+---------+
        #         |                      |
        #         v                      |
        # +-------+--------+             |
        # |      SUM       +<------------+
        # +-------+--------+
        #         |
        #         v
        #        OUT1
        #         +
        #         |
        #         v
        # +-------+--------+
        # |   3D MAXPOOL   +---------> OUT2
        # +----------------+

        # First convblock
        if is_first_block:
            # If it's the first block we don't want to create a full pre-activated and batchnormalized block
            # we just need a simple convolution of the inputs.
            first_conv = Conv3D(
                conv_filters_depths[0],
                (3, 3, 3),
                padding="same",
                strides=(1, 1, 1),
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

        # Input and output dimensionalities are not supposed to match: we do a conv_filter_depts[1] 1x1x1 convolutions
        # to adjust input and output sizes
        shortcut = Conv3D(
            conv_filters_depths[1],
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            data_format="channels_last",
        )(input_layer)
        shortcut = BatchNormalization(axis=-1)(
            shortcut
        )  # Batchnormalization of the transformed inputs
        out = add([out, shortcut])  # Adding inputs and outputs

        out_maxpooled = MaxPooling3D(pool_size=(2, 2, 2), data_format="channels_last")(
            out
        )

        return out_maxpooled, out

    @classmethod
    def _get_decoder_block(
        cls,
        input_layer,
        to_concatenate_layer,
        conv_filters_depths,
        batch_normalization,
        transposed_convolution=False,
        pre_activation=True,
    ):
        """
    
        Parameters
        ----------
        input_layer : keras layer
            primary input layer
        to_concatenate_layer : keras layer
            layer to be concatenated to inputs
        conv_filters_depths : tuple or list
            number of convolutional filters to use in the two convolutional steps
        batch_normalization : bool
            enables batchnorms
        pre_activation : bool
            enables pre-activation ordering scheme
        transposed_convolution : bool
            enables transposed convolution instead of resize-convolution
    
        Returns
        -------
        out : keras_layer
    
        """

        # Decoder block structure is similar to encoder block, with exception made for the initial concatenation.
        # The inputs are upsampled using a Conv3DTranspose layer.

        # Transposed Convolution layers are known to possibly cause checkerboard artifacts, if that's the case
        # one could switch to a deterministic Upsampling + Conv3D alternative, sacrificing some representational capability
        # https://distill.pub/2016/deconv-checkerboard/

        # The general structure of a Decoder block is:

        #       INPUT1                  INPUT2
        #         +                        +
        #         |                        |
        #         v                        v
        # +-------+-----------+    +-------+-------+
        # | TRANSPOSED CONV3D +--->+ CONCATENATION |
        # +-------------------+    +-------+-------+
        #                                  |
        #         +------------------------+
        #         |                        |
        #         v                        v
        # +-------+--------+    +----------+------+
        # |  CONV BLOCK 0  |    |1X1X1 CONVOLUTION|
        # +-------+--------+    +--------+--------+
        #         |                      |
        #         v                      v
        # +-------+--------+    +--------+--------+
        # |  CONV BLOCK 1  |    |   BATCHNORM     |
        # +-------+--------+    +---------+-------+
        #         |                       |
        #         v                       |
        # +-------+--------+              |
        # |      SUM       +<-------------+
        # +-------+--------+
        #         |
        #         v
        #        OUT
        if not transposed_convolution:
            upsampled = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(
                input_layer
            )

            up_conv = Conv3D(
                filters=conv_filters_depths[0],
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                data_format="channels_last",
            )(upsampled)
        else:
            up_conv = Conv3DTranspose(
                filters=conv_filters_depths[0],
                kernel_size=(2, 2, 2),
                strides=(2, 2, 2),
                data_format="channels_last",
            )(input_layer)

        # Concatenation with corresponding element in contracting path
        concatenation_layer = concatenate([up_conv, to_concatenate_layer], axis=-1)

        # First conv block
        first_conv = cls._get_convolution_block(
            input_layer=concatenation_layer,
            n_filters=conv_filters_depths[1],
            batch_normalization=batch_normalization,
            pre_activation=pre_activation,
        )

        # Second conv block
        out = cls._get_convolution_block(
            input_layer=first_conv,
            n_filters=conv_filters_depths[1],
            batch_normalization=batch_normalization,
            pre_activation=pre_activation,
        )

        # Residual shortcut
        shortcut = Conv3D(
            filters=conv_filters_depths[1],
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            data_format="channels_last",
        )(concatenation_layer)
        shortcut = BatchNormalization(axis=-1)(shortcut)
        out = add([out, shortcut])

        return out

    @staticmethod
    def _combine_layers(input_layer, layerlist):
        """
        combine_layers
    
        combines a list of sequential layers into a single one
    
        Parameters:
            input_layer : keras layer
                input_layer
            layerlist : list
                list of layers to stack
    
        Returns:
            combined_layer : keras_layer
                stacked layers
        """

        layer_in = input_layer
        for layerfunc in layerlist:
            layer_in = layerfunc(layer_in)

        return layer_in

    @staticmethod
    def _get_filter_depths(base_filters, block_depth, decoder=False):
        """
    get_filter_depths
    
        Calculates the number of filters for each convolutional layer.
    
        Encoding filters are calculated as
    
        f = 2^d * base
    
        (f, 2f)
    
        where d is a depth index and base is user-defined
    
        Decoding filters are calculated as
    
        f1 = 2^(d+1) * base
    
        f2 = 2^d * base
    
        (2f1, f2)
    
        to match input filter layers.
    
        Parameters
        ----------
        base_filters : int
            base number of filters
        block_depth : int
            block depth index in UNET structure
        decoder : bool
            to be enabled for decoding blocks
    
        Returns
        -------
        (filters0, filters1) : tuple
            number of conv filters for first and second convolutional block
    
        """
        if decoder:
            filters0 = (
                2 ** (block_depth + 2)
            ) * base_filters  # Decoder 2 gets 2*filters3, filters2
            filters1 = (2 ** block_depth) * base_filters
            return filters0, filters1
        else:
            filters = (
                2 ** block_depth
            ) * base_filters  # Encoder 2 gets filters2, 2*filters2
            return filters, 2 * filters
