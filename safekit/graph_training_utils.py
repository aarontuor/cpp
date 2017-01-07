"""
Utilities for training the parameters of tensorflow computational graphs.
"""

import tensorflow as tf
import sys
import math

OPTIMIZERS = {'grad': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer}

class EarlyStop:
    """
    A class for determining when to stop a training while loop by a bad count criterion.
    If the data is exhausted or the model's performance hasn't improved for *badlimit* training
    steps, the __call__ function returns false. Otherwise it returns true.

    """
    def __init__(self, badlimit=20):
        """
        :param badlimit: Limit of for number of training steps without improvement for early stopping.
        """
        self.badlimit = badlimit
        self.badcount = 0
        self.current_loss = sys.float_info.max

    def __call__(self, mat, loss):
        """
        Returns a boolean for customizable stopping criterion.
        For first loop iteration set loss to sys.float_info.max.

        :param mat: Current batch of features for training.
        :param loss: Current loss during training.
        :return: boolean, True when mat is not None and self.badcount < self.badlimit and loss != inf, nan.
        """
        if mat is None:
            sys.stderr.write('Done Training. End of data stream.')
            cond = False
        elif math.isnan(loss) or math.isinf(loss):
            sys.stderr.write('Exiting due divergence: %s\n\n' % loss)
            cond = False
        elif loss > self.current_loss:
            self.badcount += 1
            if self.badcount >= self.badlimit:
                sys.stderr.write('Exiting. Exceeded max bad count.')
                cond = False
            else:
                cond = True
        else:
            self.badcount = 0
            cond = True
        self.current_loss = loss
        return cond


class ModelRunner:
    """
    A class for gradient descent training tensorflow models.
    """

    def __init__(self, loss, ph_dict, learnrate=0.01, opt='adam', debug=False):
        """

        :param loss: The objective function for optimization strategy.
        :param ph_dict: A dictionary of names (str) to tensorflow placeholders.
        :param learnrate: The step size for gradient descent.
        :param opt: A tensorflow op implementing the gradient descent optimization strategy.
        :param debug: Whether or not to print debugging info.
        """
        self.loss = loss
        self.ph_dict = ph_dict
        self.debug = debug
        self.train_op = OPTIMIZERS[opt](learnrate).minimize(loss)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def train_step(self, datadict):
        """
        Performs a training step of gradient descent with given optimization strategy.

        :param datadict: A dictionary of names (str) matching names in ph_dict to numpy matrices for this mini-batch.
        """
        self.sess.run(self.train_op, feed_dict=get_feed_dict(datadict, self.ph_dict, debug=self.debug))

    def eval(self, datadict, eval_tensors):
        """
        Evaluates tensors without effecting parameters of model.

        :param datadict: A dictionary of names (str) matching names in ph_dict to numpy matrices for this mini-batch.
        :param eval_tensors: Tensors from computational graph to evaluate as numpy matrices.
        :return: A list of evaluated tensors as numpy matrices.
        """
        return self.sess.run(eval_tensors, feed_dict=get_feed_dict(datadict, self.ph_dict, train=0, debug=self.debug))


def get_feed_dict(datadict, ph_dict, train=1, debug=False):

    """
    Function for pairing placeholders of a tensorflow computational graph with numpy arrays.

    :param datadict: A dictionary with keys matching keys in ph_dict, and values are numpy arrays.
    :param ph_dict: A dictionary where the keys match keys in datadict and values are placeholder tensors.
    :param train: {1,0}. Different values get fed to placeholders for dropout probability, and batch norm statistics
                depending on if model is training or evaluating.
    :param debug: (boolean) Whether or not to print dimensions of contents of placeholderdict, and datadict.
    :return: A feed dictionary with keys of placeholder tensors and values of numpy matrices.
    """
    fd = {ph_dict[key]: datadict[key] for key in ph_dict}
    dropouts = tf.get_collection('dropout_prob')
    bn_deciders = tf.get_collection('bn_deciders')
    if dropouts:
        for prob in dropouts:
            if train == 1:
                fd[prob[0]] = prob[1]
            else:
                fd[prob[0]] = 1.0
    if bn_deciders:
        fd.update({decider: [train] for decider in bn_deciders})
    if debug:
        for desc in ph_dict:
            print('%s\n\tph: %s\t%s\tdt: %s\t%s' % (desc,
                                                    ph_dict[desc].get_shape().as_list(),
                                                    ph_dict[desc].dtype,
                                                    datadict[desc].shape,
                                                    datadict[desc].dtype))
        print(fd.keys())
    return fd
