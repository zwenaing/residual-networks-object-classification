import tensorflow as tf


def resnet_model_fn(features, labels, mode, model_class, resnet_size, weight_decay, learning_rate_fn, momentum, data_format, multi_gpu=False):
    """
    This function is the general resnet model function from which cifar 10 model is constructed
    :param features: Image data
    :param labels: the labels
    :param mode: mode, training, prediction or evaluation
    :param model_class: the class of the model
    :param resnet_size: the size of resnet
    :param weight_decay: the weight decay
    :param learning_rate_fn: the learning rate decay method
    :param momentum: the momentum of optimizer
    :param data_format: data format
    :param multi_gpu: boolean indicating whether multiple gpus will be used
    :return: an estimator
    """
    model = model_class(resnet_size)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, axis=1)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions)

    cross_entropy = tf.losses.softmax_cross_entropy(labels, logits)
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy + weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(global_step)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

        if multi_gpu:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_ops = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_ops = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {
        'accuracy': accuracy
    }
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode, predictions, loss, train_ops, metrics
    )
