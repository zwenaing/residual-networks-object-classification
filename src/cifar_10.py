import resnet
import resnet_model
import tensorflow as tf

_WIDTH = 32     # the width of each image in CIFAR-10 dataset
_HEIGHT = 32    # the height of each image in CIFAR-10 dataset
_CHANNELS = 3   # the channels of each image in CIFAR-10 dataset
_NUM_CLASSES = 10   # the number of classes in CIFAR-10 dataset

_NUM_IMAGES = {
    # size of train-test split in CIFAR-10 dataset
    'train': 50000,
    'validation': 10000
}

"""
This class represents a Resnet model and inherits Renet Model class. The total number of layers is 6n + 2 including
the very first 3 x 3 convolution layer, 6n 3 x 3 convolution layers with [16, 32, 64] filters in each 2n block and 
final 10 fully connected softmax layer. For more details, refer to the paper. 
"""
class Cifar10Model(resnet.Model):
    """
    The Resnet model for Cifar10 dataset.
    Args:
        resnet_size: The number of parameterized layers for the residual network
        num_classes: The number of unique classes in the labels
        data_format: The format of input data, either channels_first or channels_last
    """
    def __init__(self, resnet_size, num_classes=_NUM_CLASSES, data_format='channels_last'):

        # check if the given resnet size is 6n + 2, otherwise this violates the given architecture
        if resnet_size % 6 != 2:
            raise ValueError("Resnet size must be 6n + 2", resnet_size)

        num_blocks = (resnet_size - 2) // 6     # n

        super().__init__(
            resnet_size=resnet_size,
            n_classes=num_classes,
            filters=16,
            kernel_size=3,      # kernel size of first conv layer
            conv_stride=1,      # stride of first conv layer
            first_pool_size=None,
            first_pool_stride=None,
            second_pool_size=8,
            second_pool_stride=1,
            block_fn=resnet.normal_block,   # use 2-layer residual blocks
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2],    # strides in each block
            final_size=64,
            data_format=data_format
        )


def cifar_model_fn(features, labels, mode, params):
    """
    Returns the cifar-10 residual network model
    :param features: the image data
    :param labels: the labels of images
    :param mode: the mode, training, evaluation or prediction
    :param params: the parameters to be passed to the model
    :return: a resnet model fn
    """
    features = tf.reshape(features, [-1, _WIDTH, _HEIGHT, _CHANNELS])

    # the function that decays learning rate overtime
    learning_rate_fn = resnet.learning_rate_with_decay(
        num_images=_NUM_IMAGES['train'],
        batch_size=params['batch_size'],
        boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )
    weight_decay = 2e-4
    return resnet_model.resnet_model_fn(features, labels, mode, Cifar10Model, params['resnet_size'], weight_decay,
                                  learning_rate_fn, 0.9, params['data_format'], params['multi_gpu'])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    argparser = resnet.ResnetArgParser()
    argparser.set_defaults(
        data_dir='/tmp/cifar10_data',
        model_dir='/tmp/cifar10_model',
        resnet_size=32,
        train_epochs=250,
        epochs_per_eval=10,
        batch_size=128
    )
    FLAGS, unparse = argparser.parse_known_args()
    resnet.resnet_main(FLAGS, cifar_model_fn)
    tf.app.run()