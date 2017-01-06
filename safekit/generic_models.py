import tensorflow as tf
import numpy as np
import sys
import math

OPTIMIZERS = {'grad': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer}

class Loop():
    def __init__(self, badlimit=20):
        """
        :param badlimit: limit of badcount for early stopping
        """
        self.badlimit = badlimit
        self.badcount = 0
        self.current_loss = sys.float_info.max

    def __call__(self, mat, loss):
        """
        Returns a boolean for customizable stopping criterion. For first loop set loss to sys.float_info.max.
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


class SimpleModel():
    """
    A class for gradient descent training arbitrary models.

    :param loss: Loss Tensor for gradient descent optimization (should evaluate to real number).
    :param pointloss: A tensor of individual losses for each datapoint in minibatch.
    :param ph_dict: A dictionary with string keys and tensorflow placeholder values.
    :param learnrate: step_size for gradient descent.
    :param resultfile: Where to print loss during training.
    :param debug: Whether to print debugging info.
    :param badlimit: Number of times to not improve during training before quitting.
    """

    def __init__(self, loss, pointloss, contrib, ph_dict,
                 learnrate=0.01, resultfile=None,
                 opt='adam', debug=False,
                 verbose=0):
        self.loss = loss
        self.pointloss = pointloss
        self.contrib = contrib
        self.ph_dict = ph_dict
        self.out = open(resultfile, 'w')
        self.debug = debug
        self.verbose = verbose
        self.train_step = OPTIMIZERS[opt](learnrate).minimize(loss)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)


    def train(self, train_data, loop):
        """
        :param train_data: A Batcher object that delivers batches of train data.
        :param loop: A function or callable object that returns a boolean depending on current data and current loss.
        """
        self.out.write('day user red loss\n')
        mat = train_data.next_batch()
        loss = sys.float_info.max
        while loop(mat, loss): #mat is not None and self.badcount < self.badlimit and loss != inf, nan:
            datadict = {'features': mat[:, 3:], 'red': mat[:, 2], 'user': mat[:, 1], 'day': mat[:, 0]}
            _, loss, pointloss, contrib = self.sess.run((self.train_step, self.loss, self.pointloss, self.contrib),
                                                        feed_dict=self.get_feed_dict(datadict, self.ph_dict))
            if self.verbose == 1:
                self.print_all_contrib(datadict, loss, pointloss, contrib, train_data.index)
            elif self.verbose == 0:
                self.print_results(datadict, loss, pointloss, train_data.index)

            mat = train_data.next_batch()
        self.out.close()

    def print_results(self, datadict, loss, pointloss, index):
        for d, u, t, l, in zip(datadict['day'].tolist(), datadict['user'].tolist(),
                               datadict['red'].tolist(), pointloss.flatten().tolist()):
            self.out.write('%s %s %s %s\n' % (d, u, t, l))
        print('index: %s loss: %.4f' % (index, loss))

    def print_all_contrib(self, datadict, loss, pointloss, contrib, index):
        for time, user, red, loss, contributor in zip(datadict['day'].tolist(),
                                                      datadict['user'].tolist(),
                                                      datadict['red'].tolist(),
                                                      pointloss.flatten().tolist(),
                                                      contrib.tolist()):
            self.out.write('%s %s %s %s ' % (time, user, red, loss))
            self.out.write(str(contributor).strip('[').strip(']').replace(',', ''))
            self.out.write('\n')
        print('index: %s loss: %.4f' % (index, loss))


def get_feed_dict(datadict, ph_dict, train=1, debug=False):

    """
    Function for pairing placeholders of a computational graph with numpy arrays.

    :param datadict: A dictionary with keys matching keys in ph_dict, and values are numpy matrices.
    :param ph_dict: A dictionary where the keys match keys in datadict and values are placeholder tensors.
    :param train: {1,0}. Different values get fed to placeholders for dropout probability, and batch norm statistics
                depending on if model is training or evaluating.
    :param debug: (boolean) Whether or not to print dimensions of contents of placeholderdict, and datadict.
    :return: A feed dictionary with keys of placeholder tensors and values of numpy matrices.
    """
    fd = {ph_dict[key]:datadict[key] for key in ph_dict}
    dropouts = tf.get_collection('dropout_prob')
    bn_deciders = tf.get_collection('bn_deciders')
    if dropouts:
        for prob in dropouts:
            if train == 1:
                fd[prob[0]] = prob[1]
            else:
                fd[prob[0]] = 1.0
    if bn_deciders:
        fd.update({decider:[train] for decider in bn_deciders})
    if debug:
        for desc in ph_dict:
            print('%s\n\tph: %s\t%s\tdt: %s\t%s' % (desc,
                                                    ph_dict[desc].get_shape().as_list(),
                                                    ph_dict[desc].dtype,
                                                    datadict[desc].shape,
                                                    datadict[desc].dtype))
            print(fd.keys())
    return fd


def print_datadict(datadict):
    for k, v in datadict.iteritems():
        print(k + str(v.shape))
