__author__ = 'dan'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import lasagne.layers as layers
from lasagne.nonlinearities import softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import cPickle as pickle
import sys
import os
import cv2
import theano
import numpy.random as nrandom

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def float32(k):
    return np.cast['float32'](k)

def shuffle_in_unison(a, b):
    rng_state = nrandom.get_state()
    nrandom.shuffle(a)
    nrandom.set_state(rng_state)
    nrandom.shuffle(b)
    return a, b

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def load_train_data(datafile, datadir):
    f = open(datafile)
    lines = []
    for i, l in enumerate(f, 1):
        lines.append(l)

    for j in range(1, i):
        filename, level = lines[j].split(',')
        print "Processing train image {}: {}".format(j, filename)
        img = cv2.imread(datadir + "/" + filename + ".jpeg").astype(np.float64)
        img_r = cv2.resize(img, (150, 150))
        if j == 1:
            X = np.zeros((i-1, img_r.shape[2], img_r.shape[0], img_r.shape[1]))
            labels = np.empty((i-1), dtype=np.int32)
        X[j-1] = img_r.reshape(3, 150, 150)
        labels[j-1] = level.rstrip('\n')
    f.close()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
#    scaler = StandardScaler()
#    X = scaler.fit_transform(X)
    return X, y, encoder

def load_test_data(datafile, datadir):
    f = open(datafile)
    lines = []
    for i, l in enumerate(f, 1):
        lines.append(l)

    for j in range(1, i):
        filename = lines[j].rstrip('\n')
        print "Processing test image {}: {}".format(j, filename)
        img = cv2.imread(datadir + "/" + filename + ".jpeg").astype(np.float64)
        img_r = cv2.resize(img, (150, 150))
        if j == 1:
            X = np.zeros((i-1, img_r.shape[2], img_r.shape[0], img_r.shape[1]))
            ids = np.empty((i-1), dtype=object)
        X[j-1] = img_r.reshape(3, 150, 150)
        ids[j-1] = filename
    f.close()
    return X, ids

def make_submission(ids, preds, name='my_neural_net_submission.csv'):
    with open(name, 'w') as f:
        f.write('image, level')
        f.write('\n')
        for id, pred in zip(ids, preds):
            f.write(id + ", " + str(pred))
            f.write('\n')
    print "Wrote submission to file {}.".format(name)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": range(1,len(preds)+1), "Label": preds}).to_csv(fname, index=False, header=True)

sys.setrecursionlimit(10000)
train_files = '/Users/dan/Google Drive/kaggle/retinopathy/train'
#test_files = '/Users/dan/Google Drive/kaggle/retinopathy/test'
test_files = '/Users/dan/Google Drive/kaggle/retinopathy/train'

print "Getting data"
X, y, encoder = load_train_data('csv/trainLabelsBal_1.csv', train_files)

print "X len: {}".format(len(X))
print "y len: {}".format(len(y))
X_test, ids = load_test_data('csv/testLabels_0.csv', test_files)
print "Getting number of classes and features"
num_classes = len(encoder.classes_)
num_features = X.shape[1]
# shuffle X and y
X, y = shuffle_in_unison(X, y)

print "Number of classes: {}".format(num_classes)
print "Number of features: {}".format(num_features)

net0 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 3, 150, 150),

    conv1_num_filters=8, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.2,
    conv2_num_filters=16, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.3,
    conv3_num_filters=32, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    hidden1_num_units=500,
    dropout4_p=0.5,
    hidden2_num_units=500,

    output_num_units=num_classes,
    output_nonlinearity=softmax,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=False,
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=25),
    ],
    max_epochs=100,
    verbose=1,
    eval_size=0.2,
)

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 3, 150, 150),
    hidden1_num_units=128,  # number of units in hidden layer
    output_nonlinearity=softmax,  # output layer uses identity function
    output_num_units=num_classes,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
#        EarlyStopping(patience=25),
    ],
    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    eval_size=0.2,
    verbose=1,
    )

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 3, 150, 150),

    conv1_num_filters=8, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    hidden1_num_units=50,
    output_num_units=num_classes,
    output_nonlinearity=softmax,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,
    on_epoch_finished=[
#        AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
#        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=20),
    ],
    max_epochs=100,
    verbose=1,
    eval_size=0.2,
)

net3 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 3, 150, 150),
    hidden1_num_units=200,
    output_nonlinearity=softmax,
    output_num_units=num_classes,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
    ],
    regression=False,
    max_epochs=200,
    verbose=1,
    )



print "Running NN fit"

print "X shape: {}".format(X.shape)
print "Y shape: {}".format(y.shape)
print "Y: {}".format(y)
net3.fit(X, y)

predictions = net3.predict(X_test)

print "Predictions: {}".format(predictions)
#convolution_based_preds = fit_convolutional_model(train_x_reshaped, train_y, image_width, image_height, test_x_reshaped)
#write_preds(convolution_based_preds, "convolutional_nn.csv")

print "Make submission"
make_submission(ids, predictions)

print "Pickling..."
with open('net0.pickle', 'wb') as f:
    pickle.dump(net0, f, -1)

