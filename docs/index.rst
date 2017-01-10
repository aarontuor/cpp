.. safekit documentation master file, created by
   sphinx-quickstart on Thu Jan  5 17:42:22 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. papers

.. _Deep Learning for Unsupervised Insider Threat Detection in Structured Cybersecurity Data Streams: https://studentweb.cs.wwu.edu/~tuora/aarontuor/materials/anomaly_detection.pdf


About Safekit
=============
Safekit is a python software distribution for anomaly detection from multivariate streams,
developed for the AIMSAFE initiative at Pacific Northwest National Laboratory. It should have a better name, so any suggestions are welcome.
An exposition of the models in this package can be found in the paper:

`Deep Learning for Unsupervised Insider Threat Detection in Structured Cybersecurity Data Streams`_

The code of the toolkit is written in python using the tensorflow deep learning
toolkit and numpy.

Core Modules
------------
    :any:`tf_ops`: Functions for building tensorflow computational graph models.

    :any:`np_ops`: Module of numpy operations for tensorflow models. Handles transformations on the input and output of
data to models.

    :any:`batch`: Module for mini-batching data.

    :any:`graph_training_utils`: Utilities for training the parameters of tensorflow computational graphs.

    :any:`util`: Python utility functions.

Supplementary packages
----------------------

    :any:`features`: Modules for feature derivation of various data sets.

    :any:`models`: Models implemented for various data sets.

Dependencies
===============

This toolkit was written using tensorflow version 12 (the latest) and numpy. The latest version of tensorflow should
be installed in a virtual environment. I recommend labelling the virtualenv with the tensorflow version number as
the tensorflow distribution is volatile as of late.

Installation
=============

Clone the repo from github. From the top level directory run:

.. code-block:: bash

    $ python setup.py develop

To test your build acquire a copy of `no_weekends_reordered.csv` place it in the cpp/safekit/models directory
and from the cpp/safekit/models directory run:

.. code-block:: bash

    $ mkdir cert_dnn_auto_results
    $ python cert_agg_dnn_autoencoder.py no_weekends_reordered.csv cert_dnn_auto_results ../features/cert_agg_no_weekends.json

Documentation
=============


.. toctree::
   :maxdepth: 2

   tf_ops.rst
   np_ops.rst
   batch.rst
   graph_training_utils.rst
   models.rst
   features.rst

Contributing
============


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
