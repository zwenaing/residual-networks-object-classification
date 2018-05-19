import resnet
import tensorflow as tf

_WIDTH = 32
_HEIGHT = 32
_CHANNELS = 3

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000
}

class Cifar10Model(resnet.Model):

    def __init__(self, resnet_size, num_classes, data_format='channel_last'):

        num_blocks = (resnet_size - 2) // 6

        super().__init__(
            resnet_size=resnet_size,
            n_classes=num_classes,
            filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            second_pool_size=8,
            second_pool_stride=1,
            block_fn=resnet.normal_block,
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2],
            final_size=64,
            data_format=data_format
        )

def cifar_model_fn(features, labels, mode, params):
    features = tf.reshape(features, [-1, _WIDTH, _HEIGHT, _CHANNELS])

    learning_rate_fn = resnet.learning_rate_with_decay(
        num_images=_NUM_IMAGES['train'],
        batch_size=128,
        boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )
    weight_decay = 2e-4
    return resnet.resnet_model_fn(features, labels, mode, Cifar10Model, params['resnet_size'], weight_decay,
                                  learning_rate_fn, 0.9, params['data_format'])


if __name__ == '__main__':
    resnet.resnet_main(cifar_model_fn)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()