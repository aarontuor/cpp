#!/usr/bin/env python

"""
Deep Neural Network Autoencoder for multivariate Cert data. Anomaly detection is performed on model output
by ranking of loss scores within a time window from output of model.
"""

import tensorflow as tf
import sys
import argparse
import json
from safekit.batch import OnlineBatcher, split_batch
from safekit.graph_training_utils import ModelRunner, EarlyStop
from safekit.tf_ops import join_multivariate_inputs, dnn, softmax_dist_loss, diag_mvn_loss, multivariate_loss, eyed_mvn_loss


def return_parser():
    """
    Defines and returns argparse ArgumentParser object.

    :return: ArgumentParser
    """
    parser = argparse.ArgumentParser("Dnn auto-encoder for online unsupervised training.")
    parser.add_argument('datafile',
                        type=str,
                        help='The csv data file for our unsupervised training.'+\
                             'fields: day, user, redcount, [count1, count2, ...., count408]')
    parser.add_argument('results_folder', type=str, help='The folder to print results to.')
    parser.add_argument('dataspecs', type=str, help='Filename of json file with specification of feature indices.')
    parser.add_argument('-learnrate', type=float, default=0.001,
                        help='Step size for gradient descent.')
    parser.add_argument("-layers", nargs='+',
                        type=int, default=[100, 100, 408], help="A list of hidden layer sizes.")
    parser.add_argument('-mb', type=int, default=256, help='The mini batch size for stochastic gradient descent.')
    parser.add_argument('-act', type=str, default='tanh', help='May be "tanh" or "relu"')
    parser.add_argument('-bn', action='store_true', help='Use this flag if using batch normalization.')
    parser.add_argument('-keep_prob', type=float, default=None,
                        help='Percent of nodes to keep for dropout layers.')
    parser.add_argument('-debug', action='store_true',
                        help='Use this flag to print feed dictionary contents and dimensions.')
    parser.add_argument('-dist', type=str, default='diag',
                        help='"diag" or "ident". Describes whether to model multivariate guassian with identity, '
                             'or arbitrary diagonal covariance matrix.')
    parser.add_argument('-maxbadcount', type=str, default=20, help='Threshold for early stopping.')
    parser.add_argument('-embedding_ratio', type=float, default=.75, help='For determining size of embeddings for categorical features.')
    parser.add_argument('-min_embed', type=int, default=2, help='Minimum size for embeddings of categorical features.')
    parser.add_argument('-max_embed', type=int, default=1000, help='Maximum size for embeddings of categorical features.')
    parser.add_argument('-verbose', type=int, default=0, help='1 to print full loss contributors.')
    return parser


def write_results(datadict, pointloss, outfile):
    """
    Writes loss for each datapoint, along with meta-data to file.

    :param datadict: Dictionary of data names (str) keys to numpy matrix values for this mini-batch.
    :param pointloss: MB X 1 numpy array
    :param outfile: Where to write results.
    :return:
    """
    for d, u, t, l, in zip(datadict['time'].tolist(), datadict['user'].tolist(),
                           datadict['redteam'].tolist(), pointloss.flatten().tolist()):
        outfile.write('%s %s %s %s\n' % (d, u, t, l))


def write_all_contrib(datadict, pointloss, contrib, outfile):
    """
    Writes loss, broken down loss from all contributors, and metadata for each datapoint to file.

    :param datadict: Dictionary of data names (str) keys to numpy matrix values for this mini-batch.
    :param pointloss: MB X 1 numpy array.
    :param contrib: MB X total_num_loss_contributors nompy array.
    :param outfile: Where to write results.
    :return:
    """
    for time, user, red, loss, contributor in zip(datadict['time'].tolist(),
                                                  datadict['user'].tolist(),
                                                  datadict['redteam'].tolist(),
                                                  pointloss.flatten().tolist(),
                                                  contrib.tolist()):
        outfile.write('%s %s %s %s ' % (time, user, red, loss))
        outfile.write(str(contributor).strip('[').strip(']').replace(',', ''))
        outfile.write('\n')

if __name__ == '__main__':

    args = return_parser().parse_args()
    outfile_name = "cpp_mv_auto_lr_%s_nl_%s_hs_%s_mb_%s_bn_%s_kp_%s_ds_%s_bc_%s_em_%s" % (args.learnrate,
                                                                                          len(args.layers),
                                                                                          args.layers[0],
                                                                                          args.mb,
                                                                                          args.bn,
                                                                                          args.keep_prob,
                                                                                          args.dist,
                                                                                          args.maxbadcount,
                                                                                          args.embedding_ratio)
    if not args.results_folder.endswith('/'):
        args.results_folder += '/'
    outfile = open(args.results_folder + outfile_name, 'w')

    if args.act == 'tanh':
        activation = tf.tanh
    elif args.act == 'relu':
        activation = tf.nn.relu
    else:
        raise ValueError('Activation must be "relu", or "tanh"')

    if args.dist == "ident":
        mvn = eyed_mvn_loss
    else:
        mvn = diag_mvn_loss

    data = OnlineBatcher(args.datafile, args.mb)

    feature_spec = {'categorical': ['role', 'project', 'func', 'dep', 'team', 'sup'],
                    'continuous': ['ocean', 'counts']}

    dataspecs = json.load(open(args.dataspecs, 'r'))
    x, ph_dict = join_multivariate_inputs(feature_spec, dataspecs,
                                          args.embedding_ratio, args.max_embed, args.min_embed)

    h = dnn(x, layers=args.layers, act=activation, keep_prob=args.keep_prob, bn=args.bn)

    loss_spec = [('role', softmax_dist_loss, dataspecs['role']['num_classes']),
                 ('project', softmax_dist_loss, dataspecs['project']['num_classes']),
                 ('func', softmax_dist_loss, dataspecs['func']['num_classes']),
                 ('dep', softmax_dist_loss, dataspecs['dep']['num_classes']),
                 ('team', softmax_dist_loss, dataspecs['team']['num_classes']),
                 ('sup', softmax_dist_loss, dataspecs['sup']['num_classes']),
                 ('ocean', mvn, len(dataspecs['ocean'])),
                 ('counts', mvn, len(dataspecs['counts']))]

    loss_matrix = multivariate_loss(h, loss_spec, ph_dict)
    loss_vector = tf.reduce_sum(loss_matrix, reduction_indices=1)  # is MB x 1
    loss = tf.reduce_mean(loss_vector)  # is scalar

    eval_tensors = [loss, loss_vector, loss_matrix]
    model = ModelRunner(loss, ph_dict, learnrate=args.learnrate, opt='adam', debug=args.debug)
    raw_batch = data.next_batch()
    current_loss = sys.float_info.max
    not_early_stop = EarlyStop(args.maxbadcount)

    loss_feats = ['role', 'project', 'func', 'dep', 'team', 'sup', 'ocean', 'counts']
    # training loop
    while not_early_stop(raw_batch, current_loss):  # mat is not None and self.badcount < self.badlimit and loss != inf, nan:
        datadict = split_batch(raw_batch, dataspecs)
        targets = {'target_' + name: datadict[name] for name in loss_feats}
        datadict.update(targets)
        model.train_step(datadict)
        current_loss, pointloss, contrib = model.eval(datadict, eval_tensors)
        if args.verbose == 1:
            write_all_contrib(datadict, pointloss, contrib, outfile)
        elif args.verbose == 0:
            write_results(datadict, pointloss, outfile)
        print('index: %s loss: %.4f' % (data.index, current_loss))
        raw_batch = data.next_batch()
    outfile.close()
