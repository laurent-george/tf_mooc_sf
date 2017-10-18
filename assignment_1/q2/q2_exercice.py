"""
Problem 2: Logistic regression
You can choose to do one of the three following tasks:
Task 1: Improving the accuracy of our logistic regression on MNIST (objective 97 % accuracy on test set)

"""

import argparse
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
from tensorflow.python.training import training_util


def load_mnist(path='./data'):
    mnist = input_data.read_data_sets(path, one_hot=True)
    return mnist

def define_network():
    pass

def perso_mnist_net_model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):

    """
    a model_fn for Estimator class
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropout = 0.8
    else:
        dropout = 1.0
    learning_rate = params['learning_rate']

    n_classes = 10

    with tf.variable_scope('conv1') as scope:
         # first, reshape the image to [BATCH_SIZE, 28, 28, 1] to make it work with tf.nn.conv2d
        images = tf.reshape(features, shape=[-1, 28, 28, 1])
        kernel = tf.get_variable('kernel', [5, 5, 1, 32],
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [32],
                        initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv + biases, name=scope.name)

    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

    with tf.variable_scope('conv2') as scope:
        # similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
        kernel = tf.get_variable('kernels', [5, 5, 32, 64],
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [64],
                            initializer=tf.random_normal_initializer())

        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv + biases, name=scope.name)

    # output is of dimension BATCH_SIZE x 14 x 14 x 64
    # layers.conv2d(images, 64, 5, 1, activation_fn=tf.nn.relu, padding='SAME')
    with tf.variable_scope('pool2') as scope:
        # similar to pool1
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME')

        # output is of dimension BATCH_SIZE x 7 x 7 x 64

    with tf.variable_scope('fc') as scope:
        # use weight of dimension 7 * 7 * 64 x 1024
        input_features = 7 * 7 * 64
        w = tf.get_variable('weights', [input_features, 1024],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [1024],
                            initializer=tf.constant_initializer(0.0))

        # reshape pool2 to 2 dimensional
        pool2 = tf.reshape(pool2, [-1, input_features])
        fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')

        # pool2 = layers.flatten(pool2)
        # fc = layers.fully_connected(pool2, 1024, tf.nn.relu)

        fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

    with tf.variable_scope('softmax_linear') as scope:
        w = tf.get_variable('weights', [1024, n_classes],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [n_classes],
                            initializer=tf.random_normal_initializer())
        logits = tf.matmul(fc, w) + b

    with tf.name_scope('predictions'):
        predictions = tf.nn.softmax(logits)

    with tf.name_scope('loss'):
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(entropy, 'loss')

    #with tf.name_scope('metrics'):
    #    accuracy = tf.contrib.metrics.accuracy(predictions, labels)


    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=training_util.get_global_step())

    #print("Declaring {}".format("_".join([mode, "loss"])))
    tf.summary.scalar(loss.name, loss)
    #tf.summary.scalar("_".join([mode, "loss"]), accuracy)

    return tf.estimator.EstimatorSpec(predictions=predictions, loss=loss, train_op=train_op, mode=mode)


def main():
    mnist = load_mnist()
    batch_size = 1000

    images, labels = mnist.train.next_batch(batch_size=batch_size)

    config= tf.estimator.RunConfig()
    config = config.replace(save_summary_steps=1)
    estimator = tf.estimator.Estimator(model_fn=perso_mnist_net_model_fn,
                                       model_dir='mnist_test',
                                       params={'learning_rate':0.001,
                                               'dropout': 0.75},
                                       config=config)

    input_fn_train = partial(mnist.train.next_batch, batch_size=batch_size, shuffle=True)
    input_fn_validation = partial(mnist.validation.next_batch, batch_size=batch_size)
    input_fn_test = partial(mnist.test.next_batch, batch_size=batch_size)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=input_fn_train,  # First-class function
        eval_input_fn=input_fn_validation,  # First-class function
        min_eval_frequency=10,  # Eval frequency, each x steps  we run eval
    )
    #train_steps=1000,  # Minibatch steps

    experiment.continuous_train_and_eval()


    #print("start of training")
    #estimator.train(input_fn=input_fn_train, steps=20000)
    #print("end of training")
    #estimator.evaluate(input_fn=input_fn_validation)
    #estimator.predict(input_fn=input_fn_test)




if __name__ == "__main__":
    main()