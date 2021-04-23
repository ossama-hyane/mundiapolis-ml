#!/usr/bin/env python3
"""
Module to calculate accuracy of prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
     a function that calculates the accuracy of a prediction
     y placeholders with the right labels of the input data
     y_pred tensor containing the network's predictions
     return tensor containing the decimal accuracy of the prediction
    """
    ca = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    res = tf.reduce_mean(tf.cast(ca, tf.float32))
    return res
