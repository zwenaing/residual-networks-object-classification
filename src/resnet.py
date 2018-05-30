import tensorflow as tf
import input_data
import argparse

_BATCH_NORM_MOMENTUM = 0.997
_BATCH_NORM_EPSILON = 1e-5


def pad_image(inputs, kernel_size, data_format):
    """
    Pad the image.
    :param inputs: Input images
    :param kernel_size: the kernel size to perform convolution
    :param data_format: the format of data, either channels_first or channels_last
    :return: padded images
    """
    total = kernel_size - 1
    pad_beg = total // 2
    pad_end = total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def padded_conv2d(inputs, kernel_size, filters, strides, data_format):
    """
    Perform the padding and then convolution
    :param inputs: Input images
    :param kernel_size: the kernel size to perform convolution
    :param filters: the number of filters
    :param strides: the strides of convolution
    :param data_format: the format of data, either channels_first or channels_last
    :return: the convolution operation
    """
    if strides > 1:
        inputs = pad_image(inputs, kernel_size, data_format)

    conv = tf.layers.conv2d(inputs, filters, kernel_size, strides, 'same' if strides == 1 else 'valid', data_format,
                            kernel_initializer=tf.variance_scaling_initializer(), use_bias=False)
    return conv


def batch_norm_relu(inputs, training, data_format):
    """
    Apply batch normalization and relu to the given input images.
    :param inputs: Input images
    :param training: a flag to indicate whether during training or not
    :param data_format: the format of data, either channels_first or channels_last
    :return: the data after operations
    """
    inputs = tf.layers.batch_normalization(
        inputs, axis=1 if data_format == 'channels_first' else 3, momentum=_BATCH_NORM_MOMENTUM,
        epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def normal_block(inputs, filters, strides, training, projection_shortcut, data_format):
    """
    2-layer residual block with batch normalization and relu after convolution layer.
    :param inputs: Input images
    :param filters: number of filters
    :param strides: strides of convolutions
    :param training: a flag to indicate whether during training or not
    :param projection_shortcut: a function if projection is necessary on shortcuts, None otherwise
    :param data_format: the format of data, either channels_first or channels_last
    :return: one 2-layer residual block
    """
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
    """
    2-layer residual block with batch normalization and relu before convolution layer.
    :param inputs: Input images
    :param filters: number of filters
    :param strides: strides of convolutions
    :param training: a flag to indicate whether during training or not
    :param projection_shortcut: a function if projection is necessary on shortcuts, None otherwise
    :param data_format: the format of data, either channels_first or channels_last
    :return: one 2-layer residual block
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = padded_conv2d(inputs=inputs, kernel_size=3, filters=filters, strides=strides, data_format=data_format)
    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = padded_conv2d(inputs=inputs, kernel_size=3, filters=filters, strides=1, data_format=data_format)

    return inputs + shortcut


def bottleneck_block(inputs, filters, strides, training, projection_shortcut, data_format):
    """
    3-layer bottleneck residual block with batch normalization and relu after convolution layer.
    :param inputs: Input images
    :param filters: number of filters
    :param strides: strides of convolutions
    :param training: a flag to indicate whether during training or not
    :param projection_shortcut: a function if projection is necessary on shortcuts, None otherwise
    :param data_format: the format of data, either channels_first or channels_last
    :return: one 3-layer bottleneck residual block
    """
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
    """
    3-layer bottleneck residual block with batch normalization and relu before convolution layer.
    :param inputs: Input images
    :param filters: number of filters
    :param strides: strides of convolutions
    :param training: a flag to indicate whether during training or not
    :param projection_shortcut: a function if projection is necessary on shortcuts, None otherwise
    :param data_format: the format of data, either channels_first or channels_last
    :return: one 3-layer bottleneck residual block
    """
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
    """

    :param inputs: Input images
    :param block_fn: the block function to be used in the block layer
    :param n_blocks: number of blocks
    :param filters: the number of filters in the first block
    :param strides: the strides used in the first block
    :param training: a boolean to indicate whether training or not
    :param data_format: format specifying channels_first or channels_last
    :return: the inputs
    """
    filters_out = 4 * filters if block_fn is bottleneck_block else filters  # last layers is 4x in bottleneck layer

    def projection_shortcut(inputs):
        # function to perform projection in the shortcut
        return padded_conv2d(inputs, kernel_size=1, filters=filters_out, strides=strides, data_format=data_format)

    inputs = block_fn(inputs, filters, strides, training, projection_shortcut, data_format)

    for i in range(1, n_blocks):
        inputs = block_fn(inputs, filters, 1, training, None, data_format)

    return inputs


def learning_rate_with_decay(num_images, batch_size, boundary_epochs, decay_rates):
    """
    Get the correct learning rate depending on the boundary epochs.
    :param num_images: number of images
    :param batch_size: the batch size
    :param boundary_epochs: the epochs at which to decay the learning rate
    :param decay_rates: the rate to decay the learning rate
    :return: the learning rate
    """
    initial_learning_rate = 0.1
    batches_per_epoch = num_images // batch_size
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    rates = [initial_learning_rate * rate for rate in decay_rates]

    def learning_rate_fn(global_step):
        return tf.train.piecewise_constant(global_step, boundaries, rates)

    return learning_rate_fn


def validate_batch_size_multi_gpu(batch_size):
    """
    Check if the batch size is divisible by the number of gpus available so that data can be separated among
    multiple gpus.
    :param batch_size: batch size to be used during training
    """
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_devices if d.device_type == 'GPU'])

    if not num_gpus:
        raise ValueError("Multi_gpus is specified, but no gpus are found")

    remainder = batch_size % num_gpus
    if remainder:
        raise ValueError("The batch size must be divisible by the number of GPUs available")


class Model:

    def __init__(self, resnet_size, n_classes, filters, kernel_size, conv_stride, first_pool_size, first_pool_stride,
                 second_pool_size, second_pool_stride, block_fn, block_sizes, block_strides, final_size, data_format):
        self.resnet_size = resnet_size      # the number of parameterized layers
        self.n_classes = n_classes
        self.filters = filters              # the number of filters in the first conv layer
        self.kernel_size = kernel_size      # kernel size of first conv layer
        self.conv_stride = conv_stride      # stride of first conv layer
        # first max pooling layer receptive field size and stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        # second average pooling layer receptive field size and stride
        self.second_pool_size = second_pool_size
        self.second_pool_stride = second_pool_stride
        # type of blocks to use in the network
        self.block_fn = block_fn
        # the number of layers in each block
        self.block_sizes = block_sizes
        # the strides in each block, must have same dimension with block_sizes
        self.block_strides = block_strides
        self.final_size = final_size    # the number of neurons after flattening and before fc layer
        self.data_format = data_format

    def __call__(self, inputs, training):

        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        # first conv layer
        inputs = padded_conv2d(inputs, self.kernel_size, self.filters, self.conv_stride, self.data_format)
        # max pooling layer if specified
        if self.first_pool_size:
            inputs = tf.layers.max_pooling2d(
                inputs, self.first_pool_size, self.first_pool_stride, 'same', self.data_format, 'first_max_pool')
        # consecutive residual blocks
        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.filters * (2 ** i)
            inputs = block_layer(
                inputs, self.block_fn, num_blocks, num_filters, self.block_strides[i], training, self.data_format)
        inputs = batch_norm_relu(inputs, training, self.data_format)
        # global average pooling layer if specified
        if self.second_pool_size:
            inputs = tf.layers.average_pooling2d(
                inputs, self.second_pool_size, self.second_pool_stride, 'valid', self.data_format, 'second_max_pool')
        # flatten the neurons
        inputs = tf.reshape(inputs, [-1, self.final_size])
        # fully connected layer
        inputs = tf.layers.dense(inputs, self.n_classes)

        return inputs


def resnet_main(flags, model_function):
    """
    The function for training the network using the given model function.
    :param flags: the arguments given from the command line
    :param model_function: the model to run
    """

    # replicate the model function for multiple gpus
    if flags.multi_gpu:
        validate_batch_size_multi_gpu(flags.batch_size)
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_function,
            loss_reduction=tf.losses.Reduction.MEAN
        )

    # initialize the estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=flags.model_dir,
        params={
            'resnet_size': flags.resnet_size,
            'data_format': flags.data_format,
            'multi_gpu': flags.multi_gpu
        }
    )

    print("Starting a training cycle")
    # train the network
    classifier.train(
        input_fn=lambda : input_data.get_input_data(flags.batch_size, flags.shuffle_buffer, True, input_data.parse_record_fn, flags.train_epochs)
    )

    print("Starting evaluation")
    # evaluate the network
    eval_results = classifier.evaluate(
        input_fn=lambda : input_data.get_input_data(flags.batch_size, flags.shuffle_buffer, False, input_data.parse_record_fn, flags.epochs_per_eval)
    )

    print(eval_results)


class ResnetArgParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument("--data_dir", type=str, default="tmp/resnet_data", help="The directory of input data")
        self.add_argument("--model_dir", type=str, default="log/resnet_model", help="The directory where model data is stored")
        self.add_argument("--resnet_size", type=int, default=50, help="The size of resnet model")
        self.add_argument("--train_epochs", type=int, default=100, help="Number of epochs to train")
        self.add_argument("--batch_size", type=int, default=32, help="The batch size to use during training")
        self.add_argument("--data_format", type=str, default=None, choices=["channels_first", "channels_last"],
                          help="A flag to override which data format to use in the model")
        self.add_argument("--epochs_per_eval", type=int, default=1, help="The number of training epochs to run in each evaluation")
        self.add_argument("--shuffle_buffer", type=int, default=100, help="The size of the buffer for shuffling input data")
        self.add_argument("--multi_gpu", action='store_true', help="Run across multpile gpus")






