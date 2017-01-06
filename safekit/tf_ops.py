import tensorflow as tf
import math
from np_ops import fan_scale


def weights(distribution, shape, dtype=tf.float32, initrange=1e-5,
            seed=None, l2=0.0, name='weights'):
    """
    Wrapper parameterizing common constructions of tf.Variables.

    :param distribution: A string identifying distribution 'tnorm' for truncated normal, 'rnorm' for random normal, 'constant' for constant, 'uniform' for uniform.
    :param shape: Shape of weight tensor.
    :param dtype: dtype for weights
    :param initrange: Scales standard normal and truncated normal, value of constant dist., and range of uniform dist. [-initrange, initrange].
    :param seed: For reproducible results.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param name: For variable scope.
    :return: A tf.Variable.
    """

    if distribution == 'norm':
        wghts = tf.Variable(initrange*tf.random_normal(shape, 0, 1, dtype, seed))
    elif distribution == 'tnorm':
        wghts = tf.Variable(initrange*tf.truncated_normal(shape, 0, 1, dtype, seed))
    elif distribution == 'uniform':
        wghts = tf.Variable(tf.random_uniform(shape, -initrange, initrange, dtype, seed))
    elif distribution == 'constant':
        wghts = tf.Variable(tf.constant(initrange, dtype=dtype, shape=shape))
    else:
        raise ValueError("Argument 'distribution takes values 'norm', 'tnorm', 'uniform', 'constant', "
                          "Received %s" % distribution)
    if l2 != 0.0:
        tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(wghts), l2, name=name + 'weight_loss'))
    return wghts


def batch_normalize(tensor_in, epsilon=1e-5, decay=0.999, name="batch_norm"):
    """
    Batch Normalization:
    `Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift`_

    An exponential moving average of means and variances in calculated to estimate sample mean
    and sample variance for evaluations. For testing pair placeholder is_training
    with [0] in feed_dict. For training pair placeholder is_training
    with [1] in feed_dict. Example:

    Let **train = 1** for training and **train = 0** for evaluation

    .. code-block:: python
        bn_deciders = {decider:[train] for decider in tf.get_collection('bn_deciders')}
        feed_dict.update(bn_deciders)

    :param tensor_in: input Tensor_
    :param epsilon: A float number to avoid being divided by 0.
    :param name: For variable_scope_
    :return: Tensor with variance bounded by a unit and mean of zero according to the batch.
    """

    is_training = tf.placeholder(tf.int32, shape=[None]) # [1] or [0], Using a placeholder to decide which
                                          # statistics to use for normalization allows
                                          # either the running stats or the batch stats to
                                          # be used without rebuilding the graph.
    tf.add_to_collection('bn_deciders', is_training)

    pop_mean = tf.Variable(tf.zeros([tensor_in.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([tensor_in.get_shape()[-1]]), trainable=False)

    # calculate batch mean/var and running mean/var
    batch_mean, batch_variance = tf.nn.moments(tensor_in, [0], name=name)

    # The running mean/variance is updated when is_training == 1.
    running_mean = tf.assign(pop_mean,
                             pop_mean * (decay + (1.0 - decay)*(1.0 - tf.to_float(is_training))) +
                             batch_mean * (1.0 - decay) * tf.to_float(is_training))
    running_var = tf.assign(pop_var,
                            pop_var * (decay + (1.0 - decay)*(1.0 - tf.to_float(is_training))) +
                            batch_variance * (1.0 - decay) * tf.to_float(is_training))

    # Choose statistic
    mean = tf.nn.embedding_lookup(tf.pack([running_mean, batch_mean]), is_training)
    variance = tf.nn.embedding_lookup(tf.pack([running_var, batch_variance]), is_training)

    shape = tensor_in.get_shape().as_list()
    gamma = weights('constant', [shape[1]], initrange=0.0, name=name + '_gamma')
    beta = weights('constant', [shape[1]], initrange=1.0, name=name + '_beta')

    # Batch Norm Transform
    inv = tf.rsqrt(epsilon + variance, name=name)
    tensor_in = beta * (tensor_in - mean) * inv + gamma

    return tensor_in


def dropout(tensor_in, prob, name='Dropout'):
    """
    Adds dropout node.
    :param tensor_in: Input tensor_.
    :param prob: The percent of units to keep.
    :param name: A name for the tensor.
    :return: Tensor_ of the same shape of *tensor_in*.
    """
    if isinstance(prob, float):
        keep_prob = tf.placeholder(tf.float32)
        tf.add_to_collection('dropout_prob', (keep_prob, prob))
    return tf.nn.dropout(tensor_in, keep_prob)


def dnn(x, layers=[100, 408], act=tf.nn.relu, scale_range=1.0, bn=False, keep_prob=None, name='nnet'):
    """
    An arbitrarily deep neural network. Output has non-linear activation.

    :param x: Input to the network.
    :param layers: List of sizes of network layers.
    :param act: Activation function to produce hidden layers of neural network.
    :param scale_range: Scaling factor for initial range of weights (Set to 1/sqrt(fan_in).
    :param bn: Whether to use batch normalization.
    :param keep_prob: The percent of nodes to keep in dropout layers.
    :param name: For naming and variable scope.

    :return: (tf.Tensor) Output of neural net. This will be just following a linear transform,
             so that final activation has not been applied.
    """

    for ind, hidden_size in enumerate(layers):
        with tf.variable_scope('layer_%s' % ind):

            fan_in = x.get_shape().as_list()[1]
            W = tf.Variable(fan_scale(scale_range, act, x)*tf.truncated_normal([fan_in, hidden_size],
                                                     mean=0.0, stddev=1.0,
                                                     dtype=tf.float32, seed=None, name='W'))
            tf.add_to_collection(name + '_weights', W)
            b = tf.Variable(tf.zeros([hidden_size]))
            tf.add_to_collection(name + '_bias', b)
            x = tf.matmul(x,W) + b
            if bn:
                x = batch_normalize(x, name=name + '_bn')
            x = act(x, name='h' + str(ind)) # The hidden layer
            tf.add_to_collection(name + '_activation', x)
            if keep_prob:
                x = dropout(x, keep_prob, name=name + '_dropouts')
    return x


def softmax_dist_loss(truth, h, dimension, scale_range=1.0):
    """

    :param truth:
    :param h:
    :param scale_range:
    :return:
    """
    fan_in = h.get_shape().as_list()[1]
    U = tf.Variable(fan_scale(scale_range, tf.tanh, h) * tf.truncated_normal([fan_in, dimension],
                                                                             dtype=tf.float32,
                                                                             name='W'))
    b = tf.Variable(tf.zeros([dimension]))
    y = tf.nn.softmax(tf.matmul(h, U) + b)
    loss_column = -tf.log(tf.diag_part(tf.nn.embedding_lookup(tf.transpose(y), truth)))
    loss_column = tf.reshape(loss_column, [-1, 1])

    return loss_column


def eyed_mvn_loss(truth, h, scale_range=1.0):
    """

    :param truth:
    :param h:
    :param scale_range:
    :return:
    """
    fan_in = h.get_shape().as_list()[1]
    dim = truth.get_shape().as_list()[1]
    U = tf.Variable(fan_scale(scale_range, tf.tanh, h) * tf.truncated_normal([fan_in, dim],
                                                                             dtype=tf.float32, name='U'))
    b = tf.Variable(tf.zeros([dim]))
    y = tf.matmul(h, U) + b
    loss_columns = tf.square(y-truth)
    return loss_columns, None


def diag_mvn_loss(truth, h, scale_range=1.0, variance_floor=0.1):
    """

    :param truth: (tf.Tensor) The truth for this minibatch.
    :param h:(tf.Tensor) The output of dnn.
             (Here the output of dnn , h, is assumed to be the same dimension as truth)
    :param variance_floor: (float, positive) To ensure model doesn't find trivial optimization.
    :return: (tf.Tensor) A vector of losses for each pair of vectors in truth, pair.
    """
    fan_in = h.get_shape().as_list()[1]
    dim = truth.get_shape().as_list()[1]
    U = tf.Variable(
        fan_scale(scale_range, tf.tanh, h) * tf.truncated_normal([fan_in, 2 * dim],
                                                                 dtype=tf.float32,
                                                                 name='U'))
    b = tf.Variable(tf.zeros([2 * dim]))
    y = tf.matmul(h, U) + b

    mu, var = tf.split(1, 2, y)  # split y into two even sized matrices, each with half the columns
    var = tf.maximum(tf.exp(var),  # make the variance non-negative
                     tf.constant(variance_floor, shape=[dim], dtype=tf.float32))
    logdet = tf.reduce_sum(tf.log(var), 1)  # MB x 1
    loss_columns = tf.square(truth - mu) / var  # is MB x D
    return loss_columns, tf.reshape(logdet, [-1, 1])


def full_mvn_loss(truth, h, scale_range=1.0):
    """

    :param truth: Actual datapoints to compare against learned distribution
    :param h: output of neural network (after last non-linear transform)
    :param scale_range:
    :return:
    """
    fan_in = h.get_shape().as_list()[1]
    dimension = truth.get_shape().as_list()
    U = tf.Variable(
        fan_scale(scale_range, tf.tanh, h) * tf.truncated_normal([fan_in, dimension + dimension**2],
                                                                 dtype=tf.float32,
                                                                 name='U'))
    b = tf.Variable(tf.zeros([2 * dimension + dimension**2]))
    y = tf.matmul(h, U) + b
    mu = tf.slice(y, [0, 0], [-1, dimension])  # is MB x dimension
    var = tf.slice(y, [0, dimension], [-1, -1])  # is MB x dimension^2
    var = tf.reshape(var, [-1, dimension, dimension])  # make it a MB x D x D tensor (var is a superset of the lower triangular part of a Cholesky decomp)
    z = tf.batch_matrix_triangular_solve(var, truth - mu, lower=True, adjoint=False) # z should be MB x D
    inner_prods = tf.reduce_sum(tf.square(z), 1)  # take row-wise inner products of z, leaving MB x 1 vector
    logdet = tf.reduce_sum(tf.log(tf.square(tf.batch_matrix_diag_part(var))), 1) # diag_part converts MB x D x D to MB x D, square and log preserve, then sum makes MB x 1
    loss_column = inner_prods  # is MB x 1 ... hard to track of individual features' contributions due to correlations
    return loss_column, tf.reshape(logdet, [-1, 1])


def join_multivariate_inputs(feature_spec, specs, embedding_ratio, max_embedding, min_embedding):
    """
    This function makes placeholders for all input data, performs a lookup on an embedding matrix for each categorical feature,
    and concatenates the resulting real-valued vectors from individual features into a single vector for each data point in the batch.

    :param placeholderdict: A dict to store tf placeholders for feature inputs.
    :param feature_spec: A dict {categorical: [c1, c2, ..., cp], continuous:[f1, f2, ...,fk]
                        which lists which features to use as categorical and continuous inputs to the model.
                        c1, ..., cp, f1, ...,fk should match a key in specs.
    :param specs: A python dict containing information about which indices in the incoming data point correspond to which features.
                  Entries for continuous features list the indices for the feature, while entries for categorical features
                  contain a dictionary- {'index': i, 'num_classes': c}, where i and c are the index into the datapoint, and number of distinct
                  categories for the category in question.
    :param embedding_ratio: Determines size of embedding vectors for each categorical feature: num_classes*embedding_ratio (within limits below)
    :param max_embedding: A limit on how large an embedding vector can be.
    :param min_embedding: A limit on how small an embedding vector can be.
    :return: A tuple (x, placeholderdict):
            (tensor with shape [None, Sum_of_lengths_of_all_continuous_feature_vecs_and_embedding_vecs],
            dict to store tf placeholders to pair with data, )
    """

    placeholderdict, embeddings, continuous_features, targets = {}, {}, {}, {}

    # Make placeholders for all input data and select embeddings for categorical data
    for dataname in feature_spec['categorical']:
        embedding_size = math.ceil(embedding_ratio * specs[dataname]['num_classes'])
        embedding_size = int(max(min(max_embedding, embedding_size), min_embedding))
        with tf.variable_scope(dataname):
            placeholderdict[dataname] = tf.placeholder(tf.int32, [None])
            embeddings[dataname] = tf.nn.embedding_lookup(weights('tnorm',
                                                                  (specs[dataname]['num_classes'],
                                                                   embedding_size)),
                                                          placeholderdict[dataname])

    for dataname in feature_spec['continuous']:
        placeholderdict[dataname] = tf.placeholder(tf.float32, [None, len(specs[dataname])])
        continuous_features[dataname] = placeholderdict[dataname]

    # concatenate all features
    return tf.concat(1, continuous_features.values() + embeddings.values(), name='features'), placeholderdict


def multivariate_loss(h, loss_spec, placeholder_dict):
    """
    This function computes a multivariate loss according to loss_spec.

    :param h: Final hidden layer of dnn or rnn. (Post-activation)
    :param loss_spec: A tuple of 3-tuples of the form (input_name, loss_function, dimension) where
                        input_name is the same as a target in datadict,
                         loss_function takes two parameters, a target and prediction,
                         and dimension is the dimension of the target.
    :param placeholder_dict: A dictionary to store placeholder tensors for target values.
    :return: (loss, loss_vector, loss_matrix, loss_names) where:
            loss is a scalar value (average loss over loss contributors and mini-batch)
            loss_vector is MB X 1 tensor of average loss per data point in mini-batch.
            loss_matrix is MB X concatenated_feature_size tensor containing loss for all contributors for each data point.
            loss_names is a list concatenated_feature_size long with names of all loss contributors.
    """
    log_det_list, log_det_names, loss_list, loss_names = [], [], [], []
    print("loss distributions:")
    for i, (input_name, loss_func, dimension) in enumerate(loss_spec):
        print("%s\t%s\t%s\t%s" %(i, input_name, loss_func, dimension))
        with tf.variable_scope(input_name):
            # this input will be a (classification or regression) target - need to define a placeholder for it
            if loss_func == softmax_dist_loss:
                x = tf.placeholder(tf.int32, [None])
            else:
                x = tf.placeholder(tf.float32, [None, dimension])
            placeholder_dict["target_%s" % input_name] = x

            # predict this input from the current hidden state
            if loss_func == softmax_dist_loss: # discrete
                component_wise_point_loss = loss_func(x, h, dimension)# MB X 1
                loss_names.append("loss_%s" % input_name)
            else: # continuous
                component_wise_point_loss, logdet = loss_func(x, h)# MB X DIM_MULTIVARIATE, MB X 1
                if logdet is not None:
                    log_det_list.append(logdet)
                    log_det_names.append("loss_%s.logdet" % (input_name))
                for k in range(dimension):
                    loss_names.append("loss_%s.%d" % (input_name, k))
            loss_list.append(component_wise_point_loss)

    loss_list.extend(log_det_list)
    loss_names.extend(log_det_names)
    loss_matrix = tf.concat(1, loss_list)  # is MB x (total num contributors)
    loss_vector = tf.reduce_sum(loss_matrix, reduction_indices=1)  # is MB x 1
    loss = tf.reduce_mean(loss_vector)  # is scalar

    return loss, loss_vector, loss_matrix, loss_names
