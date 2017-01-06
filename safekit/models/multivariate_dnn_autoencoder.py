#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import argparse
import json
from safekit.batch import OnlineBatcher, split_batch
from safekit.generic_models import SimpleModel, Loop
from safekit.tf_ops import join_multivariate_inputs, dnn, softmax_dist_loss, diag_mvn_loss, multivariate_loss, eyed_mvn_loss

def return_parser():
    parser = argparse.ArgumentParser("Dnn auto-encoder for online unsupervised training.")
    parser.add_argument('datafile',
                        type=str,
                        help='The csv data file for our unsupervised training.'+\
                             'fields: day, user, redcount, [count1, count2, ...., count408]')
    parser.add_argument('results', type=str, help='The folder to print results to.')
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
                             'or abitrary diagonal covariance matrix.')
    parser.add_argument('-variance_floor', type=float, default=0.1,
                        help='For diagonal covariance matrix loss calculation.')
    parser.add_argument('-scalerange', type=float, default=1.0, help='Extra scaling on top of fan_in scaling.')
    parser.add_argument('-opt', type=str, default='adam', help='Optimization strategy. {"grad", "adam"}')
    parser.add_argument('-maxbadcount', type=str, default=20, help='Threshold for early stopping.')
    parser.add_argument('-embedding_ratio', type=float, default=.75, help='For determining size of embeddings for categorical features.')
    parser.add_argument('-min_embed', type=int, default=2, help='Minimum size for embeddings of categorical features.')
    parser.add_argument('-max_embed', type=int, default=1000, help='Maximum size for embeddings of categorical features.')
    parser.add_argument('-verbose', type=int, default=0, help='1 to print full loss contributors.')
    return parser

if __name__ == '__main__':

    args = return_parser().parse_args()

    if args.act == 'tanh':
        activation = tf.tanh
    elif args.act == 'relu':
        activation = tf.nn.relu
    else:
        raise ValueError('Activation must be "relu", or "tanh"')

    data = OnlineBatcher(args.datafile, args.mb)

    feature_spec = {'categorical': ['role', 'project', 'func', 'dep', 'team', 'sup'],
                'continuous': ['ocean', 'counts']}

    dataspecs = json.load(open(args.dataspecs, 'r'))
    x, placeholderdict = join_multivariate_inputs(feature_spec, dataspecs,
                                                  args.embedding_ratio, args.max_embed, args.min_embed)

    h = dnn(x, layers=args.layers, act=activation, keep_prob=args.keep_prob,
            scale_range=args.scalerange, bn=args.bn)

    loss_spec = [('role', softmax_dist_loss, dataspecs['role']['num_classes']),
                 ('project', softmax_dist_loss, dataspecs['project']['num_classes']),
                 ('func', softmax_dist_loss, dataspecs['func']['num_classes']),
                 ('dep', softmax_dist_loss, dataspecs['dep']['num_classes']),
                 ('team', softmax_dist_loss, dataspecs['team']['num_classes']),
                 ('sup', softmax_dist_loss, dataspecs['sup']['num_classes']),
                 ('ocean', eyed_mvn_loss, len(dataspecs['ocean'])),
                 ('counts', diag_mvn_loss, len(dataspecs['counts']))]

    log_det_list, log_det_names, loss_list, loss_names = multivariate_loss(h, loss_spec, placeholderdict)

    # if args.dist == 'ident':
    #     contrib = tf.square(x-h)
    #     pointloss = tf.reduce_sum(contrib, reduction_indices=[1])
    # elif args.dist == 'diag':
    #     pointloss, contrib = mvn(x, h, scale_range=args.scalerange, variance_floor=args.variance_floor)
    # else:
    #     raise ValueError('Argument dist must be "ident" or "diag".')
    # loss = tf.reduce_mean(pointloss)
    #
    # placeholderdict = {'features': x}
    # model = SimpleModel(loss, pointloss, contrib, placeholderdict, learnrate=args.learnrate,
    #                     resultfile=args.results, opt=args.opt, debug=args.debug, verbose=args.verbose)
    # loop = Loop(badlimit=args.maxbadcount)
    # model.train(data, loop)
