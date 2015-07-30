__author__ = 'dan'

import os
import cv2

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

train_source_dir = '/Users/dan/Google Drive/kaggle/retinopathy/train'
train_out_dir = '/Users/dan/Google Drive/kaggle/retinopathy/train_normalized'

train_files = mylistdir(train_source_dir)

print "Number of training files to process: ", str(len(train_files))

num = 1
for f in train_files:
    print "Processing training file number: ", str(num)
    file_in = train_source_dir + "/" + f
    print "Reading in: ", file_in
    img = cv2.imread(file_in)
    img_r = cv2.resize(img, (400, 400))
    file_out = train_out_dir + "/" + f
    print "Writing out: ", file_out
    cv2.imwrite(file_out, img_r)
    num += 1

test_source_dir = '/Users/dan/Google Drive/kaggle/retinopathy/test'
test_out_dir = '/Users/dan/Google Drive/kaggle/retinopathy/test_normalized'

test_files = mylistdir(test_source_dir)

print "Number of test files to process: ", str(len(test_files))

num = 1
for f in test_files:
    print "Processing test file number: ", str(num)
    file_in = test_source_dir + "/" + f
    print "Reading in: ", file_in
    img = cv2.imread(file_in)
    img_r = cv2.resize(img, (400, 400))
    file_out = test_out_dir + "/" + f
    print "Writing out: ", file_out
    cv2.imwrite(file_out, img_r)
    num += 1
