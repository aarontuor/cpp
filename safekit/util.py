"""
Python utility functions.
"""

from safekit.tf_ops import softmax_dist_loss, diag_mvn_loss, full_mvn_loss


def get_multivariate_loss_names(loss_spec):
    """
    For use in conjunction with `tf_ops.multivariate_loss`. Gives the names of all contributors (columns) of the loss matrix.

    :param loss_spec: A tuple of 3-tuples of the form (input_name, loss_function, dimension) where
                        input_name is the same as a target in datadict,
                         loss_function takes two parameters, a target and prediction,
                         and dimension is the dimension of the target.
    :param placeholder_dict: A dictionary to store placeholder tensors for target values.
    :return: loss_names is a list concatenated_feature_size long with names of all loss contributors.
    """

    loss_names, log_det_names = []
    for i, (input_name, loss_func, dimension) in enumerate(loss_spec):
        if loss_func == softmax_dist_loss:  # discrete
            loss_names.append("loss_%s" % input_name)
        else:  # continuous
            if loss_func == diag_mvn_loss or loss_func == full_mvn_loss:
                log_det_names.append("loss_%s.logdet" % input_name)
            for k in range(dimension):
                loss_names.append("loss_%s.%d" % (input_name, k))

    loss_names.extend(log_det_names)

    return loss_names
