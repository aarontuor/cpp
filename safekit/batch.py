import numpy as np
import json

class OnlineBatcher():
    """
    For batching data too large to fit into memory. Written for one pass on data!!!
    """

    def __init__(self, datafile, batch_size):
        """
        :param datafile: File to read lines from.
        :param batch_size: Mini-batch size.
        """
        self.f = open(datafile, 'r')
        self.batch_size = batch_size
        self.index = 0

    def next_batch(self):
        """
        :return: until end of datafile, each time called,
                 returns mini-batch number of lines from csv file
                 as a numpy array. Returns shorter than mini-batch
                 end of contents as a smaller than batch size array.
                 Returns None when no more data is available(one pass batcher!!).
        """
        matlist = []
        l = self.f.readline()
        if l == '':
            return None
        rowtext = np.array([float(k) for k in l.strip().split(',')])
        matlist.append(rowtext)
        for i in range(self.batch_size - 1):
            l = self.f.readline()
            if l == '':
                break
            rowtext = np.array([float(k) for k in l.strip().split(',')])
            matlist.append(rowtext)
        data = np.array(matlist)
        self.index += self.batch_size
        return data

def split_batch(batch, spec):
    """
    :param batch: An assembled 3 way array of data collected from the stream with shape (num_time_steps, num_users, num_features)
    :param specs: A python dict containing information about which indices in the incoming data point correspond to which features.
                  Entries for continuous features list the indices for the feature, while entries for categorical features
                  contain a dictionary- {'index': i, 'num_classes': c}, where i and c are the index into the datapoint, and number of distinct
                  categories for the category in question.
    :return: A dictionary of numpy arrays of the split 2d feature array.
    """
    assert spec['num_features'] == batch.shape[1], "Wrong number of features: spec/%s\tbatch/%s" % (spec['num_features'], batch.shape[1])
    datadict = {}
    for dataname, value in spec.iteritems():
        if dataname != 'num_features':
            if type(value) is dict:
                datadict[dataname] = batch[:, value['index']].astype(int)
            else:
                datadict[dataname] = batch[:, value]
    return datadict
