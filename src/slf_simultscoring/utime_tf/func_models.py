import keras
import tensorflow as tf

from collections import namedtuple
from typing import Any


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

BlockArgs = namedtuple('BlockArgs',
                       ('kernel_size', 'output_filters', 'pool_size', 'dilation', 'strides', 'se_ratio'),
                       defaults=(3, 8, 1, 1, 1, 0.25))

OutputArgs = namedtuple('OutputArgs',(
    'output_name', 'n_classes', 'samples_per_segment', 'segment_ksize',
    'segment_activation', 'dense_ksize', 'dense_activation'),
    defaults=(None, 2, None, 1, 'softmax', 1, 'tanh'))


def squeeze_and_excite_1d(input_tensor, se_ratio=0.25, channel_axis=-1, activation=keras.activations.relu):
    filters = input_tensor.shape[channel_axis]
    num_reduced_filters = max(1, int(filters*se_ratio))
    se_shape = (1, filters)
    x = keras.layers.GlobalAvgPool1D()(input_tensor)
    x = keras.layers.Reshape(se_shape)(x)
    x = keras.layers.Conv1D(
        num_reduced_filters, 1,
        activation=activation,
        strides=1,
        padding='same',
        kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = keras.layers.Conv1D(
        filters, 1,
        activation='sigmoid',
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding='same')(x)
    
    return keras.layers.multiply([x, input_tensor])


def ASPP_1D(inputs, depth=128, activation=keras.activations.relu, atrous_rates=[6, 12, 18],
            ksizes=[5, 5, 5], conv_cls=keras.layers.Conv1D):
    """Atrous spatial pyramid pooling
    
    https://arxiv.org/pdf/1706.05587.pdf
    https://github.com/rishizek/tensorflow-deeplab-v3/blob/master/deeplab_model.py#L21
    https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/_deeplab.py
    """
    conv_1 = conv_cls(depth, 1, strides=1, padding='same', use_bias=False)(inputs)
    conv_1 = keras.layers.BatchNormalization()(conv_1)
    conv_1 = activation(conv_1)
    
    dilated_1 = conv_cls(depth, ksizes[0], dilation_rate=atrous_rates[0],
                                   padding='same', use_bias=False)(inputs)
    dilated_1 = keras.layers.BatchNormalization()(dilated_1)
    dilated_1 = activation(dilated_1)
    
    dilated_2 = conv_cls(depth, ksizes[1], dilation_rate=atrous_rates[1],
                                   padding='same', use_bias=False)(inputs)
    dilated_2 = keras.layers.BatchNormalization()(dilated_2)
    dilated_2 = activation(dilated_2)
    
    dilated_3 = conv_cls(depth, ksizes[2], dilation_rate=atrous_rates[2],
                                   padding='same', use_bias=False)(inputs)
    dilated_3 = keras.layers.BatchNormalization()(dilated_3)
    dilated_3 = activation(dilated_3)
    
    pooled = keras.ops.mean(inputs, axis=1, keepdims=True)
    pooled = keras.layers.Conv1D(depth, 1, padding='same', use_bias=False)(pooled)
    pooled = keras.layers.BatchNormalization()(pooled)
    pooled = activation(pooled)
    pooled = keras.ops.ones_like(conv_1) * pooled
    
    concatenated = keras.layers.Concatenate(axis=-1)([conv_1, dilated_1, dilated_2, dilated_3, pooled])
    
    projected = keras.layers.Conv1D(depth, 1, padding='same', use_bias=False)(concatenated)
    projected = keras.layers.BatchNormalization()(projected)
    projected = activation(projected)
    return projected


def ASPP(inputs, depth=128, activation=keras.activations.relu, atrous_rates=[6, 12, 18],
    ksizes=[5, 5, 5], conv_cls=keras.layers.Conv1D):
    """Create a Keras Model from the ASPP_1D function for
    prettier summary.
    """
    inps = keras.layers.Input(shape=inputs.shape[1:], name='aspp_input')
    outputs = ASPP_1D(
        inps, depth=depth, activation=activation,
        atrous_rates=atrous_rates, ksizes=ksizes,
        conv_cls=conv_cls
    )
    aspp_m = tf.keras.Model(inputs=inps, outputs=outputs, name='ASPP')
    return aspp_m(inputs)


class PadNodesToMatch(keras.Layer):
    """Pad to same shape if needed.
    
    This may be necessary when pooling leads to fractionated output shape.
    E.g. if pool_size is 4, and input length is 10, pooling leads to
    10/4 = 2.5. In this case, Keras pooling layer implicitly crops the output to
    length 2. This needs to be taken into account by padding the up path output
    in upsampling part.
    
    NOTE: node1 should be the potentially larger one.
    
    Credit: https://github.com/perslev/U-Time/blob/master/utime/models/utime.py
    """

    def call(self, node1, node2):
        s1 = tf.shape(node1)
        s2 = tf.shape(node2)
        diffs = s1[1] - s2[1]
        left_pad = diffs // 2
        right_pad = diffs // 2
        right_pad = right_pad + (diffs % 2)
        pads = tf.convert_to_tensor([[0, 0], [left_pad, right_pad], [0, 0]])
        return tf.pad(node2, pads, 'CONSTANT')


def Conv1DBlock(
        inputs: tf.Tensor,
        output_filters: int,
        kernel_size: int,
        dilation_rate: int,
        strides: int,
        se_ratio: float,
        pool_size: int,
        activation=keras.activations.relu) -> tf.Tensor:
    x = inputs

    x = keras.layers.Conv1D(
        output_filters, kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding='same',
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        use_bias=False    
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv1D(
        output_filters, kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding='same',
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        use_bias=False
    )(x)
    x = keras.layers.BatchNormalization()(x)
    residual = keras.layers.Activation(activation)(x)
    output = residual
    output = squeeze_and_excite_1d(residual, se_ratio=se_ratio, activation=activation)
    
    if pool_size > 1:
        output = keras.layers.MaxPooling1D(pool_size=pool_size)(output)

    return output, residual


def Encoder(
        input_shape: tuple[int, int],
        block_args: dict[str, Any],
        activation=keras.activations.relu) -> keras.Model:
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    residuals = []

    for args in block_args:
        x, residual = Conv1DBlock(
            x,
            output_filters=args['output_filters'],
            kernel_size=args['kernel_size'],
            dilation_rate=args['dilation'],
            strides=args['strides'],
            se_ratio=args['se_ratio'],
            pool_size=args['pool_size'],
            activation=activation
        )
        
        residuals.append(residual)

    return keras.Model(inputs=inputs, outputs=[x, *residuals[:-1]], name='Encoder')


def Upsampling1DBlock(
        inputs: tf.Tensor,
        residuals: tf.Tensor,
        output_filters: int,
        kernel_size: int,
        se_ratio: float,
        pool_size: int,
        activation=keras.activations.relu) -> tf.Tensor:
    x = inputs
    
    x = keras.layers.UpSampling1D(size=pool_size)(x)
    x = keras.layers.Conv1D(
        output_filters, kernel_size,
        padding='same',
        kernel_initializer=CONV_KERNEL_INITIALIZER
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    # Pad if pooling has implicitly cropped the outputs along the encoder path
    x = PadNodesToMatch()(residuals, x)

    x = keras.layers.Concatenate(axis=-1)([x, residuals])

    x = keras.layers.Conv1D(
        output_filters, kernel_size,
        padding='same',
        kernel_initializer=CONV_KERNEL_INITIALIZER
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv1D(
        output_filters, kernel_size,
        padding='same',
        kernel_initializer=CONV_KERNEL_INITIALIZER
    )(x)
    x = keras.layers.BatchNormalization()(x)
    output = keras.layers.Activation(activation)(x)
    output = squeeze_and_excite_1d(output, se_ratio=se_ratio, activation=activation)

    return output


def Decoder(
        input_shape: tuple[int, int],
        residual_shapes: list[tuple[int, int]],
        block_args: dict[str, Any],
        activation=keras.activations.relu) -> keras.Model:
    inputs = keras.layers.Input(shape=input_shape)
    residuals = [keras.layers.Input(shape=res_shape) for res_shape in residual_shapes]
    x = inputs
    
    # Exclude the bottom block, reverse iterate the block args and residuals
    for args, res in zip(block_args[:-1][::-1], residuals[::-1]):
        x = Upsampling1DBlock(
            inputs=x,
            residuals=res,
            output_filters=args['output_filters'],
            kernel_size=args['kernel_size'],
            se_ratio=args['se_ratio'],
            pool_size=args['pool_size'],
            activation=activation
        )

    return keras.Model(inputs=[inputs, *residuals], outputs=x, name='decoder')


def UTime(
        input_names: list[str],
        block_args: list[dict[str, Any]],
        output_args: list[dict[str, Any]],
        aspp_depth: int = 128,
        activation=keras.activations.relu) -> keras.Model:
    inputs = []
    for inp_name in input_names:
        inp = keras.layers.Input(shape=(None, 1), name=inp_name)
        inputs.append(inp)
    
    if len(inputs) > 1:
        enc_inputs = keras.layers.Concatenate(name='input_concat')(inputs)
    else:
        enc_inputs = inputs[0]

    encoded, *residuals = Encoder(input_shape=(None, len(input_names)), block_args=block_args, activation=activation)(enc_inputs)

    if aspp_depth is not None:
        encoded = ASPP(encoded, depth=aspp_depth, activation=activation)

    residual_shapes = [res.shape[1:] for res in residuals]

    decoded = Decoder(encoded.shape[1:], residual_shapes, block_args, activation=activation)([encoded, *residuals])

    outputs = []
    for ocfg in output_args:
        ocfg = OutputArgs(**ocfg)
        out = keras.layers.Conv1D(
            filters=ocfg.n_classes,
            kernel_size=ocfg.dense_ksize,
            activation=ocfg.dense_activation,
            padding='same'
        )(decoded)

        out = keras.layers.AveragePooling1D(pool_size=ocfg.samples_per_segment)(out)
        out = keras.layers.Conv1D(
            filters=ocfg.n_classes,
            kernel_size=ocfg.segment_ksize,
            padding='same',
            activation=ocfg.segment_activation,
        )(out)
        out = keras.layers.Reshape([-1, out.shape[-1]], name=ocfg.output_name)(out)
        outputs.append(out)

    return keras.Model(inputs=inputs, outputs=outputs, name='u-time')