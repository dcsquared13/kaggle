__author__ = 'dan'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import cPickle as pickle
import sys
import os

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def load_train_data(path):
    # Read in the train cvs file.  It should be of the format image_name, label
    df = pd.read_csv(path)
    # Read in the image and append to image array with label.
    #X = df.values.copy()
    #np.random.shuffle(X)

    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print "Wrote submission to file {}.".format(name)

sys.setrecursionlimit(10000)
train_files = '/Users/dan/Google Drive/kaggle/retinopathy/train_normalized'
test_files = '/Users/dan/Google Drive/kaggle/retinopathy/test_normalized'
X, y, encoder, scaler = load_train_data('csv/trainLabels.csv')
X_test, ids = load_test_data('csv/testLabels.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

print "Number of classes: {}".format(num_classes)
print "Number of features: {}".format(num_features)

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,

                 input_shape=(None, num_features),
                 dense0_num_units=200,
                 dropout_p=0.5,
                 dense1_num_units=200,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,

                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,

                 eval_size=0.2,
                 verbose=1,
                 max_epochs=20,
                 )

layers2 = [('input', InputLayer),
           ('conv1', Conv2DLayer),
           ('pool1', MaxPool2DLayer),
           ('conv2', Conv2DLayer),
           ('pool2', MaxPool2DLayer),
           ('conv3', Conv2DLayer),
           ('pool3', MaxPool2DLayer),
           ('hidden4', DenseLayer),
           ('hidden5', DenseLayer),
           ('output', DenseLayer)]

net2 = NeuralNet(
    layers=layers2,
    input_shape=(None, num_features),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2), pool1_pool_stride=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2), pool2_pool_stride=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2), pool3_pool_stride=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=num_classes, output_nonlinearity=softmax,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,
    max_epochs=500,
    verbose=1,
    )

#net0.fit(X, y)

net2.fit(X, y)

make_submission(net2, X_test, ids, encoder)

with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)
