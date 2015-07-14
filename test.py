__author__ = 'dan'

import os
import cv2
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def load_train_data(datafile, datadir):
    labels = []
    f = open(datafile)
    lines = []
    for i, l in enumerate(f, 1):
        lines.append(l)

    for j in range(1, i):
        filename, level = lines[j].split(',')
        print "Processing train image {}: {}".format(j, filename)
        img = cv2.imread(datadir + "/" + filename + ".jpeg").astype(np.float32)
        img_r = cv2.resize(img, (150, 150))
        if j == 1:
            X = np.zeros((i, img_r.shape[0], img_r.shape[1], img_r.shape[2]))
        X[j-1] = img_r
        labels.append(int(level.rstrip('\n')))
    f.close()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
#    scaler = StandardScaler()
#    X = scaler.fit_transform(X)
    return X, y, encoder

def load_test_data(datafile, datadir):
    ids = []
    f = open(datafile)
    lines = []
    for i, l in enumerate(f, 1):
        lines.append(l)

    for j in range(1, i):
        filename = lines[j].rstrip('\n')
        print "Processing test image {}: {}".format(j, filename)
        img = cv2.imread(datadir + "/" + filename + ".jpeg").astype(np.float32)
        img_r = cv2.resize(img, (150, 150))
        if j == 1:
            X = np.zeros((i, img_r.shape[0], img_r.shape[1], img_r.shape[2]))
        X[j-1] = img_r
        ids.append(filename)
    f.close()
    return X, ids

sys.setrecursionlimit(10000)
train_files = '/Users/dan/Google Drive/kaggle/retinopathy/train'
test_files = '/Users/dan/Google Drive/kaggle/retinopathy/test'

print "Getting data"
X, y, encoder = load_train_data('data/trainSample.csv', train_files)

print "X: {}".format(X)
print "y: {}".format(y)
print "encoder: {}".format(encoder)

print "Get Test Data"
X_test, ids = load_test_data('data/testSample.csv', test_files)
print "Getting number of classes and features"
num_classes = len(encoder.classes_)
num_features = X.shape[1]
print "Number of classes: {}".format(num_classes)
print "Number of features: {}".format(num_features)

