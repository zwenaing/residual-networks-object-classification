import tensorflow as tf
import os
import numpy as np

_NUM_CLASSES = 10
_IMAGE_WIDTH = 32
_IMAGE_HEIGHT = 32
_IMAGE_CHANNEL = 3

data_path = '../data/cifar-10-batches-py/'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_input_data(batch_size, shuffle_buffer, is_training, parse_record_fn, num_epochs):

    if is_training:
        filenames = [x for x in os.listdir(data_path) if x.startswith('data_batch_')]
        train_labels = np.array([unpickle(data_path + f)[b'labels'] for f in filenames])
        train_labels = np.reshape(np.ravel(train_labels), [-1, 1])
        train_data = np.array([unpickle(data_path + f)[b'data'] for f in filenames])
        train_data = np.reshape(train_data, [-1, 3072])
    else:
        filename = data_path + 'test_batch'
        dict = unpickle(filename)
        train_labels = dict[b'labels']
        train_data = dict[b'data']

    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(lambda x, y: parse_record_fn(x, y, is_training))
    dataset = dataset.batch(batch_size)

    return dataset


def parse_record_fn(x, y, is_training):
    one_hot_label = tf.reshape(tf.one_hot(y, _NUM_CLASSES), [-1])
    x = tf.reshape(x, [_IMAGE_CHANNEL, _IMAGE_WIDTH, _IMAGE_HEIGHT])
    x = tf.transpose(x, [1, 2, 0])

    if is_training:
        x = tf.image.resize_image_with_crop_or_pad(x, _IMAGE_WIDTH + 8, _IMAGE_WIDTH + 8)
        x = tf.random_crop(x, [_IMAGE_WIDTH, _IMAGE_HEIGHT, _IMAGE_CHANNEL])
        x = tf.image.random_flip_left_right(x)

    x = tf.image.per_image_standardization(x)

    return x, one_hot_label
