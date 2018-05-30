import tensorflow as tf
import os
import numpy as np

_NUM_CLASSES = 10
_IMAGE_WIDTH = 32
_IMAGE_HEIGHT = 32
_IMAGE_CHANNEL = 3

data_path = '../data/cifar-10-batches-py/'

def unpickle(file):
    """
    Read the CIFAR-10 dataset and returns the dictionary
    :param file: file path
    :return: dictionary of data
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_input_data(batch_size, shuffle_buffer, is_training, parse_record_fn, num_epochs):
    """
    Construct a tensorflow dataset from the 5 batches of data.
    :param batch_size: the batch size to use during training
    :param shuffle_buffer: the size of buffer used for shuffling
    :param is_training: boolean to indicate whether training or not, use shuffling during training
    :param parse_record_fn: the function for preprocessing images
    :param num_epochs: the number of epochs to train
    :return: tensorflow dataset
    """
    if is_training:
        # read batches of train data
        filenames = [x for x in os.listdir(data_path) if x.startswith('data_batch_')]
        # get labels
        train_labels = np.array([unpickle(data_path + f)[b'labels'] for f in filenames])
        train_labels = np.reshape(np.ravel(train_labels), [-1, 1])
        # get pixel data
        train_data = np.array([unpickle(data_path + f)[b'data'] for f in filenames])
        train_data = np.reshape(train_data, [-1, 3072])
    else:
        # read test data
        filename = data_path + 'test_batch'
        dict = unpickle(filename)
        train_labels = dict[b'labels']
        train_data = dict[b'data']

    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(lambda x, y: parse_record_fn(x, y, is_training))  # preprocess data
    dataset = dataset.batch(batch_size)

    return dataset


def parse_record_fn(x, y, is_training):
    # convert to one hot encoding
    one_hot_label = tf.reshape(tf.one_hot(y, _NUM_CLASSES), [-1])
    # dataset is in channels_first format
    x = tf.reshape(x, [_IMAGE_CHANNEL, _IMAGE_WIDTH, _IMAGE_HEIGHT])
    # reshape to be in channels_last format
    x = tf.transpose(x, [1, 2, 0])

    if is_training:
        # pad image with 4 pixels on each side
        x = tf.image.resize_image_with_crop_or_pad(x, _IMAGE_WIDTH + 8, _IMAGE_WIDTH + 8)
        # take 32 x 32 crops random
        x = tf.random_crop(x, [_IMAGE_WIDTH, _IMAGE_HEIGHT, _IMAGE_CHANNEL])
        # flip the image horizontally
        x = tf.image.random_flip_left_right(x)
    # standardize across all images
    x = tf.image.per_image_standardization(x)

    return x, one_hot_label