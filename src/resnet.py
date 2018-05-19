import tensorflow as tf
import input_data

_BATCH_NORM_MOMENTUM = 0.997
_BATCH_NORM_EPSILON = 1e-5


def pad_image(inputs, kernel_size, data_format):
    total = kernel_size - 1
    pad_beg = total // 2
    pad_end = total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def padded_conv2d(inputs, kernel_size, filters, strides, data_format):
    if strides > 1:
        inputs = pad_image(inputs, kernel_size, data_format)

    conv = tf.layers.conv2d(inputs, filters, kernel_size, strides, 'same' if strides == 1 else 'valid', data_format,
                            kernel_initializer=tf.variance_scaling_initializer(), use_bias=False)
    return conv


def batch_norm_relu(inputs, training, data_format):
    inputs = tf.layers.batch_normalization(
        inputs, axis=1 if data_format == 'channels_first' else 3, momentum=_BATCH_NORM_MOMENTUM,
        epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def normal_block(inputs, filters, strides, training, projection_shortcut, data_format):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = padded_conv2d(
        inputs, kernel_size=3, filters=filters, strides=strides, data_format=data_format
    )
    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = padded_conv2d(
        inputs, kernel_size=3, filters=filters, strides=1, data_format=data_format
    )
    inputs = batch_norm_relu(inputs, training, data_format)

    return inputs + shortcut


def normal_block_v2(inputs, filters, strides, training, projection_shortcut, data_format):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = padded_conv2d(inputs=inputs, kernel_size=3, filters=filters, strides=strides, data_format=data_format)
    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = padded_conv2d(inputs=inputs, kernel_size=3, filters=filters, strides=1, data_format=data_format)

    return inputs + shortcut


def bottleneck_block(inputs, filters, strides, training, projection_shortcut, data_format):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = padded_conv2d(
        inputs=inputs, kernel_size=1, filters=filters, strides=1, data_format=data_format)
    inputs = batch_norm_relu(inputs=inputs, data_format=data_format, training=training)
    inputs = padded_conv2d(
        inputs=inputs, kernel_size=3, filters=filters, strides=strides, data_format=data_format)
    inputs = batch_norm_relu(inputs=inputs, data_format=data_format, training=training)
    inputs = padded_conv2d(
        inputs=inputs, kernel_size=1, filters=4 * filters, strides=1, data_format=data_format)

    return inputs + shortcut


def bottleneck_block_v2(inputs, filters, strides, training, projection_shortcut, data_format):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = padded_conv2d(inputs=inputs, kernel_size=1, filters=filters, strides=1, data_format=data_format)
    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = padded_conv2d(inputs=inputs, kernel_size=3, filters=filters, strides=strides, data_format=data_format)
    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = padded_conv2d(inputs=inputs, kernel_size=1, filters=4 * filters, strides=1, data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, block_fn, n_blocks, filters, strides, training, data_format):

    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return padded_conv2d(inputs, kernel_size=1, filters=filters_out, strides=strides, data_format=data_format)

    inputs = block_fn(inputs, filters, strides, training, projection_shortcut, data_format)

    for i in range(1, n_blocks):
        inputs = block_fn(inputs, filters, 1, training, None, data_format)

    return inputs


def learning_rate_with_decay(num_images, batch_size, boundary_epochs, decay_rates):
    initial_learning_rate = 0.1
    batches_per_epoch = num_images // batch_size
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    rates = [initial_learning_rate * rate for rate in decay_rates]

    def learning_rate_fn(global_step):
        return tf.train.piecewise_constant(global_step, boundaries, rates)

    return learning_rate_fn


class Model:

    def __init__(self, resnet_size, n_classes, filters, kernel_size, conv_stride, first_pool_size, first_pool_stride,
                 second_pool_size, second_pool_stride, block_fn, block_sizes, block_strides, final_size, data_format):
        self.resnet_size = resnet_size
        self.n_classes = n_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.second_pool_size = second_pool_size
        self.second_pool_stride = second_pool_stride
        self.block_fn = block_fn
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size
        self.data_format = data_format

    def __call__(self, inputs, training):

        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = padded_conv2d(inputs, self.kernel_size, self.filters, self.conv_stride, self.data_format)
        if self.first_pool_size:
            inputs = tf.layers.max_pooling2d(
                inputs, self.first_pool_size, self.first_pool_stride, 'same', self.data_format, 'first_max_pool')

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.filters * (2 ** i)
            inputs = block_layer(
                inputs, self.block_fn, num_blocks, num_filters, self.block_strides[i], training, self.data_format)
        inputs = batch_norm_relu(inputs, training, self.data_format)
        if self.second_pool_size:
            inputs = tf.layers.average_pooling2d(
                inputs, self.second_pool_size, self.second_pool_stride, 'valid', self.data_format, 'second_max_pool')

        inputs = tf.reshape(inputs, [-1, self.final_size])
        inputs = tf.layers.dense(inputs, self.n_classes)

        return inputs


def resnet_main(model_function):

    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        params={
            'resnet_size': 20,
            'data_format': "channels_last"
        }
    )

    print("Starting a training cycle")

    classifier.train(
        input_fn=lambda : input_data.get_input_data(128, 100, True, input_data.parse_record_fn, 200)
    )

    print("Starting evaluation")

    eval_results = classifier.evaluate(
        input_fn=lambda : input_data.get_input_data(128, 100, False, input_data.parse_record_fn, 1)
    )

    print(eval_results)



