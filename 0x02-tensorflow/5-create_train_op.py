#!/usr/bin/env python3
""" Training using TensorFlow """
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent
    """
    ct = tf.train.GradientDescentOptimizer(alpha)
    res = ct.minimize(loss)
    return res
