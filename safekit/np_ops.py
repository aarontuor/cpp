"""
Module of numpy operations for tensorflow models. Handles transformations on the input and output of
data to models.
"""

import tensorflow as tf
import numpy as np


def fan_scale(initrange, activation, tensor_in):
    """
    Creates a scaling factor for weight initialization according to best practices.

    :param initrange: Scaling in addition to fan_in scale.
    :param activation: A tensorflow non-linear activation function
    :param tensor_in: Input tensor to layer of network to scale weights for.
    :return: (float) scaling factor for weight initialization.
    """
    if activation == tf.nn.relu:
        initrange *= np.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
    else:
        initrange *= (1.0/np.sqrt(float(tensor_in.get_shape().as_list()[1])))
    return initrange