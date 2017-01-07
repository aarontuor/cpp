.. safekit documentation master file, created by
   sphinx-quickstart on Thu Jan  5 17:42:22 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About Safekit
=============
Safekit is a python software distribution for anomaly detection from multivariate streams,
developed for the AIMSAFE initiative at Pacific Northwest National Laboratory.
It should have a better name, so any suggestions are welcome.
The code of the toolkit is written in python using the tensorflow deep learning
toolkit and numpy.
The toolkit is comprised of four core modules and two supplementary packages.
In this introduction we will discuss each of these components.

Core Modules
------------
    :any:`tf_ops`

    :any:`np_ops`

    :any:`batch`

    :any:`graph_training_utils`

Supplementary packages
----------------------

    :any:`features`

    :any:`models`

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
